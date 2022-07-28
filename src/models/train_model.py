"""
Functions required for the training pipeline, including dataloaders.
"""
from typing import Iterator, Dict, Callable
import jax
from jax import numpy as jnp
import numpy as np
from datasets.arrow_dataset import Dataset
from transformers import (
    FlaxAutoModel,
    FlaxRobertaModel,
    HfArgumentParser,
)
from transformers.modeling_flax_outputs import FlaxBaseModelOutputWithPooling

from .model_utils import (
    Batch,
    BatchForMining,
    TweetUser,
    BatchWithEmbeddings,
    BatchForMiningWithEmbeddings,
    TokenizerOutput,
    TokenBatch,
    ShardedTokenBatch,
    ShardedBatchEmbeddings,
    BatchEmbeddings,
    Embeddings,
    reshape_batch,
    get_pooling_fn,
    ShardedModelParams,
    ModelParams,
    squared_l2_distance,
)
from ..config import PipelineConfig, ModelConfig, UserID, LookupByUID

Array = jnp.ndarray
PRNGKey = jax.random.KeyArray


def get_train_dataloader(
    dataset: Dataset,
    lookup_by_uid: LookupByUID,
    prng_key: PRNGKey,
    pipeline_args: PipelineConfig,
) -> Iterator[BatchForMining]:
    """
    Iterate through the dataset with on-the-fly shuffling.
    Note that neg candidates might be repeated if
    eval batch size > train batch size.

    Args:
     dataset: HuggingFace Dataset.
     lookup_by_uid: Mapping from uid strings to indices of the dataset.
     prng_key: JAX PRNG key.
     pipeline_args: PipelineConfig with batch sizes details.

    Yields:
     batch: BatchForTripletMining dataclass, each include train_batch_size anchors,
     train_batch_size positives, and eval_batch_size neg candidates.
    """
    dataset_indices = jnp.arange(0, len(dataset))
    anc_prng_key, pos_prng_key, neg_candidate_prng_key = jax.random.split(
        prng_key, num=3
    )

    anc_perms = jax.random.permutation(anc_prng_key, dataset_indices)
    neg_candidate_perms = jax.random.permutation(
        neg_candidate_prng_key, dataset_indices
    )

    anc_batch_size = pipeline_args.train_per_device_batch_size * jax.device_count()
    neg_batch_size = pipeline_args.eval_per_device_batch_size * jax.device_count()

    num_anc_batches = len(dataset) // anc_batch_size
    for j in range(num_anc_batches):
        anc_indices = anc_perms[j * anc_batch_size : (j + 1) * anc_batch_size]

        # The batch size of anc_batch is smaller than that of neg_batch.
        # Hence, sampling from the neg_batch might need to be repeated.
        neg_candidate_indices = jnp.take(
            neg_candidate_perms,
            jnp.arange(j * neg_batch_size, (j + 1) * neg_batch_size),
            mode="wrap",
        )

        anc_batch: Batch = reshape_batch(dataset[anc_indices])
        neg_candidate_batch: Batch = reshape_batch(dataset[neg_candidate_indices])

        anc_uids = anc_batch.info["uid"]
        pos_indices = np.zeros_like(anc_indices)
        for anc_index, uid in enumerate(anc_uids):
            # Indices of tweets from the same user.
            pos_index_choices = jnp.array(lookup_by_uid.get(uid))
            pos_index = jax.random.choice(pos_prng_key, pos_index_choices)
            pos_indices[anc_index] = pos_index

        pos_batch: Batch = reshape_batch(dataset[pos_indices])

        yield BatchForMining(
            anchor_batch=anc_batch,
            positive_batch=pos_batch,
            negative_candidate_batch=neg_candidate_batch,
        )


def _embed_mining_batch_single_shard(
    token_batch: TokenBatch,
    model: FlaxRobertaModel,
    model_args: ModelConfig,
    model_params: ModelParams,
) -> BatchEmbeddings:
    """
    Embed sharded_batch.

    Args:
     token_batch: batch of tokens (only), where for each type of token,
      "input_ids" and "attention_mask" are of shape
     (example, num_tokens).
     model: HuggingFace model template.
     model_args: specifies pooling strategy.
     model_params: HuggingFace model weights, from model.params,
      might need to be replicated across devices.

    Returns:
     embedding_batch, where each type of embeddings is of shape
     (example, embedding_dim)
    """

    def apply_model(
        tokens: TokenizerOutput,
    ) -> FlaxBaseModelOutputWithPooling:
        outputs = model(**tokens, params=model_params)
        return outputs  # type: ignore

    outputs = map(apply_model, token_batch)

    pooling_fn = get_pooling_fn(model_args)
    embeddings = BatchEmbeddings(*map(pooling_fn, outputs))

    return embeddings


_embed_mining_batch: Callable[
    [ShardedTokenBatch, FlaxRobertaModel, ModelConfig, ShardedModelParams],
    ShardedBatchEmbeddings,
] = jax.pmap(_embed_mining_batch_single_shard, static_broadcasted_argnums=(1, 2))


def get_anc_neg_distance(batch_embeddings: BatchEmbeddings) -> Array:
    """
    Return the pairwise distance between anchors and
    negative examples. Also see notebooks/jjt-anc-neg-distance-matrix.ipynb.

    Args:
     batch_embeddings: from _embed_mining_batch, but with the device dimension
      and the batch dimension flattened into one. In other words, for anc and pos,
      the embeddings is of shape (train_batch_size * num_devices, embedding_dim),
      denoted as (n_anc, embedding_dim).
      For neg, the embeddings is of shape
      (eval_batch_size * num_devices, embedding_dim), denoted as
      (n_neg, embedding_dim).

    Returns:
     If there are n_anc anchors and n_neg negative examples,
     the output of this function would be (n_neg, n_anc),
     where the (j, k) entry is the distance between the j-th neg embedding
     and the k-th anc embedding.
    """

    # (n_anc, embedding_dim)
    anc_embeddings = batch_embeddings.anchor_embeddings
    n_anc = anc_embeddings.shape[0]

    # (n_neg, embedding_dim)
    neg_embeddings = batch_embeddings.negative_embeddings
    n_neg = neg_embeddings.shape[0]

    embedding_dim = anc_embeddings.shape[-1]

    # (n_anc, n_neg, embedding_dim)
    anc_embeddings_repeated = jnp.repeat(anc_embeddings, n_neg, axis=-1).reshape(
        (n_anc, n_neg, embedding_dim)
    )

    # (n_neg, n_anc, embedding_dim)
    anc_embeddings_repeated_transposed = jnp.transpose(
        anc_embeddings_repeated, axes=(1, 0, 2)
    )

    # (n_neg, n_anc, embedding_dim)
    neg_embeddings_repeated = jnp.repeat(neg_embeddings, n_anc, axis=-1).reshape(
        (n_neg, n_anc, embedding_dim)
    )

    # (n_anc, n_neg)
    l2_difference = squared_l2_distance(
        anc_embeddings_repeated_transposed, neg_embeddings_repeated
    )

    return l2_difference


def get_margins(batch_embeddings: BatchEmbeddings) -> Array:
    """
    Returns:
     the pairwise margins d(a - n) - d(a - p).
     If there are n_anc = n_pos anchors and positive examples, and
     n_neg negative examples, the output of this function would be (n_anc, n_neg).
     The (j, k) entry would be the margin d(a_j, p_j) - d(a_j, n_k).
    """
    anc_embeddings = batch_embeddings.anchor_embeddings
    pos_embeddings = batch_embeddings.positive_embeddings

    n_anc = anc_embeddings.shape[0]
    n_neg = batch_embeddings.negative_embeddings.shape[0]

    # (n_anc,)
    anc_pos_distances = squared_l2_distance(anc_embeddings, pos_embeddings)

    # TODO: Explain why transposing is required here.
    # (n_anc, n_neg)
    anc_pos_distances_repeated = jnp.repeat(
        anc_pos_distances,
        repeats=n_neg,
        axis=-1,  # Along the n_anc axis.
    ).reshape((n_anc, n_neg))

    # (n_neg, n_anc)
    anc_pos_distances_repeated_transposed = jnp.transpose(
        anc_pos_distances_repeated, axes=(1, 0)
    )

    # (n_neg, n_anc)
    anc_neg_distances = get_anc_neg_distance(batch_embeddings)

    return anc_neg_distances - anc_pos_distances_repeated_transposed


# def get_triplet_masks

# def get_top_triplet_pairs
