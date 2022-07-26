from typing import Dict, Union
from dataclasses import dataclass, field

import numpy as np
from flax.training.common_utils import shard
import jax
from ..config import BatchInfoKeys, BatchTokenKeys

Array = jax.numpy.ndarray


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


@dataclass
class BatchForTripletMining:
    """
    Dataclass of anchor entries, positive entries,
    as well as candidates of negative entries.
    """

    anchor_batch: Batch
    positive_batch: Batch
    negative_candidate_batch: Batch


def reshape_batch(batch: Dict[str, Union[Array, str]]) -> Batch:
    """
    Convert a batch from the HuggingFace dataset into a "Batch"
    compatible with the dataloading pipeline. Apply sharding to
    enable parallel processing.
    """
    batch_info: Dict[str, str] = {k: batch[k] for k in ["uid", "tid"]}  # type: ignore
    batch_tokens: Dict[str, Array] = {
        k: np.array(batch[k]) for k in ["input_ids", "attention_mask"]
    }  # type: ignore

    # Only tokens are processed on the accelerators.
    # Batch_info is for the CPUs and doesn't require sharding.
    sharded_batch_tokens = shard(batch_tokens)

    return Batch(tokens=sharded_batch_tokens, info=batch_info)
