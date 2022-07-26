import unittest
import json
from collections import Counter

import jax
import chex
import datasets
from datasets import load_from_disk

from ..models.train_model import get_train_dataloader
from ..config import (
    DataConfig,
    ModelConfig,
    PipelineConfig,
    LookupByUID,
    BatchTokenKeys,
)

Dataset = datasets.arrow_dataset.Dataset


data_args = DataConfig(
    processed_dataset_path="data/testing/processed/tweets",
    processed_lookup_by_uid_json_path="data/testing/processed/tweets/lookup_by_uid.json",
)
num_devices = jax.device_count()

pipeline_args = PipelineConfig(
    train_per_device_batch_size=2, eval_per_device_batch_size=7
)

model_args = ModelConfig()


class GetTrainDataLoaderFromProcessedDataset(unittest.TestCase):
    def setUp(self):
        self.preprocessed_dataset: Dataset = load_from_disk(data_args.processed_dataset_path)  # type: ignore

        with open(
            data_args.processed_lookup_by_uid_json_path, "r"
        ) as lookup_by_uid_json_file:
            self.lookup_by_uid: LookupByUID = json.load(lookup_by_uid_json_file)

        for feature in ["uid", "input_ids", "attention_mask"]:
            self.assertIn(feature, self.preprocessed_dataset.features)  # type: ignore

    def test_batch_shape(self):
        dataloader = get_train_dataloader(
            dataset=self.preprocessed_dataset,
            lookup_by_uid=self.lookup_by_uid,
            prng_key=jax.random.PRNGKey(0),
            pipeline_args=pipeline_args,
        )

        for batch in dataloader:
            self.assertSetEqual(
                set(batch.anchor_batch.info.keys()), set(["uid", "tid"])
            )
            self.assertSetEqual(
                set(batch.anchor_batch.tokens.keys()),
                set(["attention_mask", "input_ids"]),
            )
            for key in ["attention_mask", "input_ids"]:  # type: ignore
                key: BatchTokenKeys
                value = batch.anchor_batch.tokens[key]
                chex.assert_axis_dimension(value, 0, jax.device_count())
                chex.assert_axis_dimension(
                    value, 1, pipeline_args.train_per_device_batch_size
                )
                chex.assert_axis_dimension(value, 2, model_args.max_seq_length)

            for key in ["attention_mask", "input_ids"]:  # type: ignore
                key: BatchTokenKeys
                value = batch.positive_batch.tokens[key]
                chex.assert_axis_dimension(value, 0, jax.device_count())
                chex.assert_axis_dimension(
                    value, 1, pipeline_args.train_per_device_batch_size
                )
                chex.assert_axis_dimension(value, 2, model_args.max_seq_length)

            for key in ["attention_mask", "input_ids"]:  # type: ignore
                key: BatchTokenKeys
                value = batch.negative_candidate_batch.tokens[key]
                chex.assert_axis_dimension(value, 0, jax.device_count())
                chex.assert_axis_dimension(
                    value, 1, pipeline_args.eval_per_device_batch_size
                )
                chex.assert_axis_dimension(value, 2, model_args.max_seq_length)

    def test_sampling_repetition(self):
        dataloader = get_train_dataloader(
            dataset=self.preprocessed_dataset,
            lookup_by_uid=self.lookup_by_uid,
            prng_key=jax.random.PRNGKey(0),
            pipeline_args=pipeline_args,
        )

        anc_tid_counter = Counter()
        pos_tid_counter = Counter()
        neg_tid_counter = Counter()

        for batch in dataloader:
            for anc_tid in batch.anchor_batch.info["tid"]:
                anc_tid_counter[anc_tid] += 1

            for pos_tid in batch.positive_batch.info["tid"]:
                pos_tid_counter[pos_tid] += 1

            for neg_tid in batch.negative_candidate_batch.info["tid"]:
                neg_tid_counter[neg_tid] += 1

        # Repetitions are not allowed except for neg_tids.
        self.assertEqual(max(anc_tid_counter.values()), 1)
        self.assertEqual(min(anc_tid_counter.values()), 1)
        self.assertLessEqual(
            max(neg_tid_counter.values()) - min(neg_tid_counter.values()), 1
        )
