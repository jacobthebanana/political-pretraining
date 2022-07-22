"""
Logic for generating per-user average embeddings of a given dataset.
"""
from dataclasses import dataclass, field
from typing import Iterator, Dict, List, Union
import json
import numpy as np
import jax
import jax.numpy as jnp
from flax.training.common_utils import shard
from flax.jax_utils import replicate
import datasets
from datasets import load_from_disk
from transformers import FlaxAutoModel, FlaxRobertaModel, HfArgumentParser
from tqdm.auto import tqdm

from ..config import (
    DataConfig,
    ModelConfig,
    PipelineConfig,
    BatchTokenKeys,
    BatchInfoKeys,
)

Dataset = datasets.arrow_dataset.Dataset
Array = jax.numpy.ndarray
NUM_COLUMNS = 80


@dataclass
class Batch:
    """
    Batch yielded from the dataloader.
    """

    info: Dict[BatchInfoKeys, str]
    tokens: Dict[BatchTokenKeys, Array]


@dataclass
class TweetUser:
    """
    Dataclass for holding the sum of embedding vectors for each user
    before calculating per-user avg. embeddings.
    """

    # Embedding sum is not initialized here since embedding_dim isn't specified.
    embedding_sum: Array
    num_tweets_processed: int = field(default=0)


def _reshape_batch(batch: Dict[str, Union[Array, str]]) -> Batch:
    batch_info: Dict[str, str] = {k: batch[k] for k in ["uid", "tid"]}  # type: ignore
    batch_tokens: Dict[str, Array] = {
        k: np.array(batch[k]) for k in ["input_ids", "attention_mask"]
    }  # type: ignore

    # Only tokens are processed on the accelerators.
    # Batch_info is for the CPUs and doesn't require sharding.
    sharded_batch_tokens = shard(batch_tokens)

    return Batch(tokens=sharded_batch_tokens, info=batch_info)


def get_dataloader(
    dataset: Dataset, pipeline_args: PipelineConfig, include_leftovers: bool = False
) -> Iterator[Batch]:
    """
    Return sharded fix-sized token batches from the dataset.
    Returns batches of identical sizes.

    If include_leftovers=False (default), data that don't fit into
    a batch would not be included.

    Note that if include_leftovers=True, the leftover batch would be
    padded with entries that were previously returned.

    Args:
     dataset: HuggingFace dataset to iterate over.
    """
    batch_size = pipeline_args.eval_per_device_batch_size * jax.device_count()
    num_batches = len(dataset) // batch_size

    for j in range(num_batches):
        batch = dataset[j * batch_size : (j + 1) * batch_size]
        formatted_batch = _reshape_batch(batch)

        yield formatted_batch

    if include_leftovers:
        leftover_batch = dataset[-batch_size:]

        formatted_leftover_batch = _reshape_batch(leftover_batch)
        yield formatted_leftover_batch


def _run_batch_inference_single_shard(
    batch: Dict[BatchTokenKeys, Array],
    model: FlaxRobertaModel,
    model_params: Dict,
) -> Array:
    """
    Return the sentence-level embeddings for texts in this batch for one shard.

    Args:
     batch: Dict, from tokenizer.
     model: HuggingFace sequence classification FLAX model.
     model_params: parameters of the said model.

    Returns:
     ndarray of (batch_size, embedding_dimension).
    """
    outputs = model(**batch, params=model_params)
    embeddings = outputs.pooler_output  # type: ignore
    return embeddings


_run_batch_inference_sharded = jax.pmap(
    _run_batch_inference_single_shard, static_broadcasted_argnums=1
)


def run_batch_inference(
    batch_tokens: Dict[BatchTokenKeys, Array],
    model: FlaxRobertaModel,
) -> Array:
    """
    Return the sentence-level embeddings for texts in this batch.
    This function runs data parallelism across all accelerator cores.

    Args:
     batch: Dict, from tokenizer.
     model: HuggingFace sequence classification FLAX model.
     model_params: parameters of the said model.

    Returns:
     ndarray of (batch_size, embedding_dimension).
    """
    sharded_model_params = replicate(model.params)
    sharded_embeddings: Array = _run_batch_inference_sharded(
        batch_tokens, model, sharded_model_params
    )
    embeddings = sharded_embeddings.reshape((-1, sharded_embeddings.shape[-1]))
    return embeddings


def _get_uid_tally_dict(
    model: FlaxRobertaModel, dataloader: Iterator[Batch], num_batches: int
) -> Dict[str, TweetUser]:

    uid_lookup_dict: Dict[str, TweetUser] = {}

    for batch in tqdm(dataloader, total=num_batches, ncols=NUM_COLUMNS):
        embeddings = run_batch_inference(batch.tokens, model)
        uids = batch.info["uid"]

        for embedding, uid in tqdm(
            zip(embeddings, uids), leave=False, ncols=NUM_COLUMNS
        ):
            tweet_user = uid_lookup_dict.get(uid)  # type: ignore
            if not tweet_user:
                tweet_user = TweetUser(embedding_sum=jnp.zeros_like(embedding))

            tweet_user: TweetUser

            # JAX does not allow in-place update.
            tweet_user.embedding_sum = tweet_user.embedding_sum + embedding
            tweet_user.num_tweets_processed += 1

            uid_lookup_dict[uid] = tweet_user

    return uid_lookup_dict


def _compute_mean_embeddings(
    uid_lookup_dict: Dict[str, TweetUser]
) -> Dict[str, List[float]]:
    uid_output: Dict[str, List[float]] = {}
    for uid, tweet_user in tqdm(uid_lookup_dict.items(), ncols=NUM_COLUMNS):
        avg_embedding = tweet_user.embedding_sum / tweet_user.num_tweets_processed
        avg_embedding_py: List[float] = avg_embedding.tolist()
        uid_output[uid] = avg_embedding_py

    return uid_output


def main():
    """
    Given a dataset of tweets, generate the average text embedding for each
    user's tweets. Save the result as a json dictionary from uid to avg. embeddings.
    """
    parser = HfArgumentParser((ModelConfig, DataConfig, PipelineConfig))
    model_args, data_args, pipeline_args = parser.parse_args_into_dataclasses()
    model_args: ModelConfig
    data_args: DataConfig
    pipeline_args: PipelineConfig

    processed_dataset = load_from_disk(data_args.processed_dataset_path)  # type: ignore
    processed_dataset: Dataset
    dataloader = get_dataloader(
        processed_dataset, pipeline_args, include_leftovers=True
    )
    batch_size = pipeline_args.eval_per_device_batch_size * jax.device_count()
    num_batches = len(processed_dataset) // batch_size
    if len(processed_dataset) % batch_size >= 1:
        num_batches += 1  # Leftover batch.

    model: FlaxRobertaModel = FlaxAutoModel.from_pretrained(model_args.base_model_name)

    uid_lookup_dict = _get_uid_tally_dict(model, dataloader, num_batches)
    uid_lookup_output = _compute_mean_embeddings(uid_lookup_dict)

    with open(data_args.output_embeddings_json_path, "w") as output_json_file:
        json.dump(uid_lookup_output, output_json_file, indent=2)


if __name__ == "__main__":
    main()
