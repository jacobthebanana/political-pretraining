"""
Functions required for the training pipeline, including dataloaders.
"""
from typing import Iterator
import jax
from jax import numpy as jnp
import numpy as np
from datasets.arrow_dataset import Dataset
from .model_utils import Batch, BatchForTripletMining, TweetUser, reshape_batch
from ..config import PipelineConfig, UserID, LookupByUID

PRNGKey = jax.random.KeyArray


def get_train_dataloader(
    dataset: Dataset,
    lookup_by_uid: LookupByUID,
    prng_key: PRNGKey,
    pipeline_args: PipelineConfig,
) -> Iterator[BatchForTripletMining]:
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

    anc_perms = jax.random.shuffle(anc_prng_key, dataset_indices)
    neg_candidate_perms = jax.random.shuffle(neg_candidate_prng_key, dataset_indices)

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

        yield BatchForTripletMining(
            anchor_batch=anc_batch,
            positive_batch=pos_batch,
            negative_candidate_batch=neg_candidate_batch,
        )
