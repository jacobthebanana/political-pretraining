from typing import Dict, Union, Tuple, Iterable, NamedTuple, Callable, overload
from dataclasses import dataclass, field

import numpy as np
from flax.training.common_utils import shard
import jax
import chex
import optax
import jax.numpy as jnp
from transformers.modeling_flax_outputs import FlaxBaseModelOutputWithPooling
from ..config import (
    BatchInfoKeys,
    BatchTokenKeys,
    MetricKeys,
    ModelConfig,
    PoolingStrategy,
)

Array = jax.numpy.ndarray

# (n_anc, n_pos) boolean. True if eligible.
TripletEligibilityMask = jax.numpy.ndarray
Embeddings = jax.numpy.ndarray

TokenizerOutput = Dict[BatchTokenKeys, Array]
ModelParams = optax.Params
ReplicatedModelParams = optax.Params


class Batch(NamedTuple):
    """
    Batch yielded from the dataloader.
    """

    info: Dict[BatchInfoKeys, str]
    tokens: TokenizerOutput


class BatchWithEmbeddings(NamedTuple):
    """
    Batch with embeddings and additional info.
    """

    info: Dict[BatchInfoKeys, str]
    tokens: Dict[BatchTokenKeys, Array]
    embeddings: Array


class ShardedTrainStepOutput(NamedTuple):
    """
    Output of the training step,
    with metrics, updated model params, and updated optimizer state.
    """

    metrics: Dict[MetricKeys, Array]
    model_params: ModelParams
    optimizer_state: optax.OptState


TrainStepOutput = ShardedTrainStepOutput


@dataclass
class TweetUser:
    """
    Dataclass for holding the sum of embedding vectors for each user
    before calculating per-user avg. embeddings.
    """

    # Embedding sum is not initialized here since embedding_dim isn't specified.
    embedding_sum: Array
    num_tweets_processed: int = field(default=0)


class ShardedBatchForMining(NamedTuple):
    """
    NamedTuple of anchor entries, positive entries,
    as well as candidates of negative entries.
    """

    anchor_batch: Batch
    positive_batch: Batch
    negative_candidate_batch: Batch


class BatchForMiningWithEmbeddings(NamedTuple):
    anchor_batch: BatchWithEmbeddings
    positive_batch: BatchWithEmbeddings
    negative_candidate_batch: BatchWithEmbeddings


def reshape_batch(batch: Dict[str, Union[Array, str]]) -> Batch:
    """
    Convert a batch from the HuggingFace dataset into a "Batch"
    compatible with the dataloading pipeline. Apply sharding to
    enable parallel processing.
    """
    batch_info: Dict[BatchInfoKeys, str] = {
        k: batch[k] for k in ["uid", "tid"]
    }  # type: ignore
    batch_tokens: Dict[str, Array] = {
        k: np.array(batch[k]) for k in ["input_ids", "attention_mask"]
    }  # type: ignore

    # Only tokens are processed on the accelerators.
    # Batch_info is for the CPUs and doesn't require sharding.
    sharded_batch_tokens = shard(batch_tokens)
    return Batch(tokens=sharded_batch_tokens, info=batch_info)


class TokenBatch(NamedTuple):
    """
    NamedTuple of anchor tokens, positive tokens,
    as well as tokens of candidates of negative entries.
    Each leaf is a JAX Numpy Array and could be sharded.
    """

    anchor_tokens: TokenizerOutput
    positive_tokens: TokenizerOutput
    negative_tokens: TokenizerOutput


class FilteredTokenBatch(NamedTuple):
    """
    Based on TokenBatch, but with all token types
    (anc, pos, neg) of the same shape, plus triplet_margin array.
    """

    anchor_tokens: TokenizerOutput
    positive_tokens: TokenizerOutput
    negative_tokens: TokenizerOutput

    triplet_margin: Array

    def to_token_batch(self) -> TokenBatch:
        return TokenBatch(
            anchor_tokens=self.anchor_tokens,
            positive_tokens=self.positive_tokens,
            negative_tokens=self.negative_tokens,
        )


class ShardedBatchEmbeddings(NamedTuple):
    """
    Embeddings for ShardedTokenBatch.
    """

    anchor_embeddings: Embeddings
    positive_embeddings: Embeddings
    negative_embeddings: Embeddings


# Same data format, but different shape
# (arrays without the device dimension.)
BatchEmbeddings = ShardedBatchEmbeddings
ShardedTokenBatch = TokenBatch

# TokenBatch, but where anc, pos, and neg each includes
# n_anc examples.
ShardedFilteredTokenBatch = TokenBatch


def get_pooling_fn(
    model_args: ModelConfig,
) -> Callable[[FlaxBaseModelOutputWithPooling], Embeddings]:
    """
    Return pooling function
    for the given pooling strategy (set in model_args).

    Args:
     model_args: specifies pooling_strategy.

    Returns:
     Pooling function for handling FlaxBaseModelOutputWithPooling,
    """
    pooling_strategy = model_args.pooling_strategy

    if (
        PoolingStrategy(pooling_strategy)
        is PoolingStrategy.CLS_EMBEDDING_WITH_DENSE_LAYER
    ):

        def cls_dense_pooler(outputs: FlaxBaseModelOutputWithPooling) -> Embeddings:
            pooled_embeddings = outputs.pooler_output  # type: ignore
            return pooled_embeddings

        return cls_dense_pooler

    elif PoolingStrategy(pooling_strategy) is PoolingStrategy.CLS_EMBEDDING_ONLY:

        def cls_passthrough_pooler(
            outputs: FlaxBaseModelOutputWithPooling,
        ) -> Embeddings:
            word_embeddings: Array = outputs.last_hidden_state
            pooled_embeddings = word_embeddings[:, 0, :]
            return pooled_embeddings

        return cls_passthrough_pooler

    else:
        assert PoolingStrategy(pooling_strategy) is PoolingStrategy.WORD_EMBEDDING_MEAN

        def word_mean_pooler(outputs: FlaxBaseModelOutputWithPooling) -> Embeddings:
            word_embeddings: Array = outputs.last_hidden_state
            pooled_embeddings = jnp.mean(word_embeddings, axis=1)
            return pooled_embeddings

        return word_mean_pooler


def get_token_batch(mining_batch: ShardedBatchForMining) -> TokenBatch:
    """
    Extract from a mining batch the tokens (and only tokens.)

    Args:
     mining_batch.

    Returns:
     token_batch.
    """

    def get_tokens(batch: Batch) -> TokenizerOutput:
        return batch.tokens

    return TokenBatch(*map(get_tokens, mining_batch))


def squared_l2_distance(x_1: Array, x_2: Array) -> Array:
    """
    Compute squared L2 distance along axis (-1).

    Args:
     x_1: (a, b, n)
     x_2: (a, b, n)

    Returns:
     (a, b). || x_1 - x_2 ||^{2}.
    """
    chex.assert_equal_shape((x_1, x_2))
    squared_difference = (x_1 - x_2) * (x_1 - x_2)

    l2_difference: Array = jnp.sum(squared_difference, axis=-1)
    return l2_difference


def _gather(sharded_array: Array) -> Array:
    device_count = jax.device_count()

    # Device dimension must be the first dimension.
    chex.assert_axis_dimension(sharded_array, 0, device_count)
    assert len(sharded_array.shape) >= 2

    gathered_shape: Tuple[int, ...] = (
        sharded_array.shape[0] * sharded_array.shape[1],
        *sharded_array.shape[2:],
    )
    output_array = sharded_array.reshape(gathered_shape)

    return output_array


@overload
def gather_shards(sharded_tree: ShardedTokenBatch) -> TokenBatch:
    ...


@overload
def gather_shards(sharded_tree: ShardedBatchEmbeddings) -> BatchEmbeddings:
    ...


def gather_shards(sharded_tree):
    """
    Un-shard the tree (undo flax.common_utils.shard).

    Args:
     sharded_array: (device_count, batch_size, ...)

    Returns:
     (device_count * batch_size, ...)
    """

    return jax.tree_map(_gather, sharded_tree)


def array_index_tokenizer_output(
    tokenizer_output: TokenizerOutput, indices: Iterable[int]
) -> TokenizerOutput:
    """
    Index the given tokenizer output with an iterable of indices.
    Reusable (unlike lambda functions.)

    Args:
     tokenizer_output: from a HuggingFace tokenizer.
     indices: Indices of entries to select from token_batch,
     relatively-indexed within this token batch.

    Returns:
     array-indexed tokenizer_output.
    """
    output: TokenizerOutput = {
        "attention_mask": tokenizer_output["attention_mask"][indices, :],
        "input_ids": tokenizer_output["input_ids"][indices, :],
    }

    return output
