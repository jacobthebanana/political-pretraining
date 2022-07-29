"""
Functions required for the training pipeline, including dataloaders.
"""
from typing import Iterator, Tuple, Dict, Callable
import jax
from jax import numpy as jnp
from flax.training.common_utils import shard
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
    ShardedBatchForMining,
    TweetUser,
    BatchWithEmbeddings,
    BatchForMiningWithEmbeddings,
    get_token_batch,
    gather_shards,
    TripletEligibilityMask,
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
    ShardedFilteredTokenBatch,
    FilteredTokenBatch,
    array_index_tokenizer_output,
)
from ..config import PipelineConfig, ModelConfig, UserID, LookupByUID

Array = jnp.ndarray
PRNGKey = jax.random.KeyArray


def get_train_dataloader(
    dataset: Dataset,
    lookup_by_uid: LookupByUID,
    prng_key: PRNGKey,
    pipeline_args: PipelineConfig,
) -> Iterator[ShardedBatchForMining]:
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

        yield ShardedBatchForMining(
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


def _get_neg_anc_indices(num_neg: int, num_anc: int) -> Tuple[Array, Array]:
    """
    Get retrieval keys for neg and anc/pos tokens.

    Recall that the (j, k) entry of margins is
    d(a_k - n_j) - d(a_k - p_k).

    The (j, k) entry of neg_indices_repeated is thus j, while
    the (j, k) entry of anc_indices_repeated is k.

    Args:
     num_neg: number of neg examples.
     num_anc: number of anchor/positive examples.

    Returns:
     neg_indices_repeated: (num_neg, num_anc), with the
     (j, k) entry of neg_indices_repeated being j.

     anc_indices_repeated: (num_neg, num_anc), with the
     (j, k) entry of anc_indices_repeated being k.
    """
    neg_indices = jnp.arange(num_neg)
    anc_indices = jnp.arange(num_anc)
    neg_indices_repeated = jnp.repeat(neg_indices, num_anc).reshape((num_neg, num_anc))
    anc_indices_repeated = (
        jnp.repeat(anc_indices, num_neg).reshape((num_anc, num_neg)).T
    )

    return neg_indices_repeated, anc_indices_repeated


def get_top_triplet_pairs(
    mining_batch: ShardedBatchForMining,
    model: FlaxRobertaModel,
    model_args: ModelConfig,
    sharded_model_params: ShardedModelParams,
) -> FilteredTokenBatch:
    """
    Given a mining batch of tokens with n_anc examples of n_anc,
    n_anc examples of pos, and n_neg examples of neg, return the
    top n_anc (anc_k, pos_k, neg_j) combinations, sorted by margin
    d(anc_k - pos_k) - d(anc_j, neg_j).

    This function isn't sharded so as to maximize the effective
    batch size and maximize the amount of information gained from
    the batch. Margins computed on each accelerator are gathered for
    comparison, sharded again, and sent to the devices.

    Args:
     mining_batch: BatchForMining with n_anc examples of anc, n_pos
     examples of pos, and n_neg examples of neg.

    Returns:
     FilteredTokenBatch. anchor_tokens and positive_tokens are paired
     as they were in the mining_batch input. However, these anc_pos pairs might
     be repeated to maximize the margin d(anc_k - pos_k) - d(anc_j, neg_j).
    """
    batch_tokens: ShardedTokenBatch = get_token_batch(mining_batch)
    sharded_batch_embeddings: ShardedBatchEmbeddings = _embed_mining_batch(
        batch_tokens, model, model_args, sharded_model_params
    )
    gathered_batch_embeddings: BatchEmbeddings = gather_shards(sharded_batch_embeddings)
    gathered_batch_tokens: TokenBatch = gather_shards(batch_tokens)

    margins = get_margins(gathered_batch_embeddings)
    num_neg = margins.shape[0]
    num_anc = margins.shape[1]

    neg_indices_repeated, anc_indices_repeated = _get_neg_anc_indices(num_neg, num_anc)

    sorted_margins, neg_indices_sorted = jax.lax.sort_key_val(
        margins.flatten(), neg_indices_repeated.flatten()
    )
    sorted_margins, anc_indices_sorted = jax.lax.sort_key_val(
        margins.flatten(), anc_indices_repeated.flatten()
    )

    num_train_triplets = num_anc
    neg_indices_selected = neg_indices_sorted[:num_train_triplets]
    anc_indices_selected = anc_indices_sorted[:num_train_triplets]
    triplet_margins_selected = sorted_margins[:num_train_triplets]

    output = FilteredTokenBatch(
        anchor_tokens=array_index_tokenizer_output(
            gathered_batch_tokens.anchor_tokens, anc_indices_selected
        ),
        positive_tokens=array_index_tokenizer_output(
            gathered_batch_tokens.positive_tokens, anc_indices_selected
        ),
        negative_tokens=array_index_tokenizer_output(
            gathered_batch_tokens.negative_tokens, neg_indices_selected
        ),
        triplet_margin=triplet_margins_selected,
    )

    return shard(output)
