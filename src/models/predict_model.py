"""
Logic for generating per-user average embeddings of a given dataset.
"""
from typing import Iterator, Dict, Literal
import numpy as np
import jax
from flax.training.common_utils import shard
import datasets

from ..config import PipelineConfig, BatchKeys

Dataset = datasets.arrow_dataset.Dataset


def get_dataloader(
    dataset: Dataset, pipeline_args: PipelineConfig
) -> Iterator[Dict[BatchKeys, jax.numpy.ndarray]]:
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
