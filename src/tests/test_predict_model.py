import unittest

import jax
import chex
from flax.jax_utils import replicate
import datasets
from datasets import load_from_disk
from transformers import (
    FlaxAutoModel,
    AutoConfig,
    RobertaConfig,
    FlaxRobertaModel,
)

from ..models.predict_model import get_dataloader, run_batch_inference

from ..config import PipelineConfig, ModelConfig, BatchKeys

Dataset = datasets.arrow_dataset.Dataset

test_processed_dataset_path = "data/testing/processed/tweets"
num_devices = len(jax.devices())
eval_per_device_batch_size = 2

model_args = ModelConfig()
pipeline_args = PipelineConfig(eval_per_device_batch_size=eval_per_device_batch_size)


class GetDataLoaderFromProcessedDataset(unittest.TestCase):
    def setUp(self):
        self.preprocessed_dataset: Dataset = load_from_disk(test_processed_dataset_path)  # type: ignore
        for feature in ["input_ids", "attention_mask"]:
            self.assertIn(feature, self.preprocessed_dataset.features)  # type: ignore

    def test_batch_shape(self):
        dataloader = get_dataloader(self.preprocessed_dataset, pipeline_args)

        for batch in dataloader:
            self.assertSetEqual(set(batch.keys()), set(["attention_mask", "input_ids"]))
            for key in ["attention_mask", "input_ids"]:  # type: ignore
                key: BatchKeys
                value = batch[key]
                chex.assert_axis_dimension(value, 0, jax.device_count())
                chex.assert_axis_dimension(value, 1, eval_per_device_batch_size)
                chex.assert_axis_dimension(value, 2, model_args.max_seq_length)


class RunInferenceOnTextBatch(unittest.TestCase):
    def setUp(self):
        preprocessed_dataset: Dataset = load_from_disk(test_processed_dataset_path)  # type: ignore

        self.dataloader = get_dataloader(preprocessed_dataset, pipeline_args)
        self.model: FlaxRobertaModel = FlaxAutoModel.from_pretrained(
            model_args.base_model_name
        )
        self.model_params = replicate(self.model.params)
        self.hf_model_config: RobertaConfig = AutoConfig.from_pretrained(
            model_args.base_model_name
        )

    def test_run_batch_inference_shape(self):
        for batch in self.dataloader:
            embeddings: jax.numpy.ndarray = run_batch_inference(batch, self.model)

            self.assertEqual(
                embeddings.shape[0],
                num_devices * eval_per_device_batch_size,
                "Number of outputs should match actual batch size.",
            )
            self.assertEqual(
                embeddings.shape[-1],
                self.hf_model_config.hidden_size,
                "Embedding dim should match model specs.",
            )
