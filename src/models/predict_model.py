"""
Logic for generating per-user average embeddings of a given dataset.
"""
from typing import Iterator, Dict
import numpy as np
import jax
from flax.training.common_utils import shard
from flax.jax_utils import replicate
import datasets
from transformers import FlaxRobertaModel

from ..config import PipelineConfig, BatchKeys

Dataset = datasets.arrow_dataset.Dataset
Array = jax.numpy.ndarray


def get_dataloader(
    dataset: Dataset, pipeline_args: PipelineConfig
) -> Iterator[Dict[BatchKeys, Array]]:
    """
    Return sharded fix-sized token batches from the dataset.
    Returns batches of identical sizes. Data that don't fit into a batch
    won't be included.

    Args:
     dataset: HuggingFace dataset to iterate over.
    """
    batch_size = pipeline_args.eval_per_device_batch_size * jax.device_count()
    num_batches = len(dataset) // batch_size

    for j in range(num_batches):
        batch = dataset[j * batch_size : (j + 1) * batch_size]
        batch = {k: np.array(batch[k]) for k in ["input_ids", "attention_mask"]}
        batch = shard(batch)

        yield batch


def _run_batch_inference_single_shard(
    batch: Dict[BatchKeys, Array],
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
    batch: Dict[BatchKeys, Array],
    model: FlaxRobertaModel,
) -> Array:
    """
    Return the sentence-level embeddings for texts in this batch.
    This function runs in parallel across all accelerator cores.

    Args:
     batch: Dict, from tokenizer.
     model: HuggingFace sequence classification FLAX model.
     model_params: parameters of the said model.

    Returns:
     ndarray of (batch_size, embedding_dimension).
    """
    sharded_model_params = replicate(model.params)
    sharded_embeddings: Array = _run_batch_inference_sharded(
        batch, model, sharded_model_params
    )
    embeddings = sharded_embeddings.reshape((-1, sharded_embeddings.shape[-1]))
    return embeddings
