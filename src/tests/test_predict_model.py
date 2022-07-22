import unittest

import jax
import chex
import datasets
from datasets import load_from_disk

from ..models.predict_model import get_dataloader

from ..config import PipelineConfig, ModelConfig, BatchKeys

Dataset = datasets.arrow_dataset.Dataset

test_processed_dataset_path = "data/testing/processed/tweets"
num_devices = len(jax.devices())
eval_per_device_batch_size = 2

model_args = ModelConfig()


class GetDataLoaderFromProcessedDataset(unittest.TestCase):
    def setUp(self):
        self.preprocessed_dataset: Dataset = load_from_disk(test_processed_dataset_path)  # type: ignore
        for feature in ["input_ids", "attention_mask"]:
            self.assertIn(feature, self.preprocessed_dataset.features)  # type: ignore

    def test_batch_shape(self):
        pipeline_args = PipelineConfig(
            eval_per_device_batch_size=eval_per_device_batch_size
        )
        dataloader = get_dataloader(self.preprocessed_dataset, pipeline_args)

        for batch in dataloader:
            self.assertSetEqual(
                set(batch.keys()), set(["attention_mask", "input_ids"])
            )
            for key in ["attention_mask", "input_ids"]:  # type: ignore
                key: BatchKeys
                value = batch[key]
                chex.assert_axis_dimension(value, 0, jax.device_count())
                chex.assert_axis_dimension(value, 1, eval_per_device_batch_size)
                chex.assert_axis_dimension(value, 2, model_args.max_seq_length)
