from typing import Dict, Union, NamedTuple, Callable
from dataclasses import dataclass, field

import numpy as np
from flax.training.common_utils import shard
import jax
import jax.numpy as jnp
from transformers.modeling_flax_outputs import FlaxBaseModelOutputWithPooling
from ..config import BatchInfoKeys, BatchTokenKeys, ModelConfig, PoolingStrategy

Array = jax.numpy.ndarray
Embeddings = jax.numpy.ndarray
TokenizerOutput = Dict[BatchTokenKeys, Array]
ModelParams = Dict
ShardedModelParams = Dict


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


@dataclass
class TweetUser:
    """
    Dataclass for holding the sum of embedding vectors for each user
    before calculating per-user avg. embeddings.
    """

    # Embedding sum is not initialized here since embedding_dim isn't specified.
    embedding_sum: Array
    num_tweets_processed: int = field(default=0)


class BatchForMining(NamedTuple):
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


class ShardedTokenBatch(NamedTuple):
    """
    NamedTuple of anchor tokens, positive tokens,
    as well as tokens of candidates of negative entries.
    Each leaf is a JAX Numpy Array and could be sharded.
    """

    anchor_tokens: TokenizerOutput
    positive_tokens: TokenizerOutput
    negative_tokens: TokenizerOutput


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
TokenBatch = ShardedTokenBatch


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


def get_token_batch(mining_batch: BatchForMining) -> TokenBatch:
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


# def get_squared_l2_distance(x_1: Array, x_2: Array) -> Array:
#     """
#     Compute L2 distance along axis (-1).
#     """
