import unittest
from os import environ

import json
import jax
import chex
import datasets
from datasets import load_from_disk
from transformers import (
    FlaxAutoModel,
    AutoConfig,
    RobertaConfig,
    FlaxRobertaModel,
)

from ..models.predict_model import (
    get_dataloader,
    run_batch_inference,
    _get_uid_tally_dict,
    _compute_mean_embeddings,
)

from ..config import PipelineConfig, ModelConfig, BatchTokenKeys, PoolingStrategy

Dataset = datasets.arrow_dataset.Dataset

test_processed_dataset_path = "data/testing/processed/tweets"
num_devices = jax.device_count()
eval_per_device_batch_size = 2
effective_batch_size = num_devices * eval_per_device_batch_size


base_model_name = environ.get("unittest_base_model_name")
if base_model_name:
    model_args = ModelConfig(base_model_name=base_model_name)
else:
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
            self.assertSetEqual(set(batch.info.keys()), set(["uid", "tid"]))
            self.assertSetEqual(
                set(batch.tokens.keys()), set(["attention_mask", "input_ids"])
            )
            for key in ["attention_mask", "input_ids"]:  # type: ignore
                key: BatchTokenKeys
                value = batch.tokens[key]
                chex.assert_axis_dimension(value, 0, jax.device_count())
                chex.assert_axis_dimension(value, 1, eval_per_device_batch_size)
                chex.assert_axis_dimension(value, 2, model_args.max_seq_length)

    def test_leftover_handling(self):
        dataloader_with_leftovers = get_dataloader(
            self.preprocessed_dataset, pipeline_args, include_leftovers=True
        )
        tid_list = []
        uid_list = []
        for batch in dataloader_with_leftovers:
            tid_list.extend(batch.info["tid"])
            uid_list.extend(batch.info["uid"])

        num_examples_yielded = len(tid_list)

        # num_examples_yielded might exceed len(self.preprocessed_dataset) as some
        # examples were repeated to pad the leftover batch.
        self.assertGreaterEqual(num_examples_yielded, len(self.preprocessed_dataset))
        self.assertLessEqual(
            num_examples_yielded, len(self.preprocessed_dataset) + effective_batch_size
        )

        self.assertSetEqual(set(self.preprocessed_dataset["uid"]), set(uid_list))


class RunInferenceOnTextBatch(unittest.TestCase):
    def setUp(self):
        self.preprocessed_dataset: Dataset = load_from_disk(test_processed_dataset_path)  # type: ignore

        self.dataloader = get_dataloader(self.preprocessed_dataset, pipeline_args)  # type: ignore
        self.model: FlaxRobertaModel = FlaxAutoModel.from_pretrained(
            model_args.base_model_name, from_pt=True
        )
        self.hf_model_config: RobertaConfig = AutoConfig.from_pretrained(
            model_args.base_model_name, from_pt=True
        )

    def test_run_batch_inference_shape(self):
        for pooling_strategy in PoolingStrategy:
            model_args_with_pooling_strategy = ModelConfig(
                pooling_strategy=pooling_strategy
            )
            dataloader = get_dataloader(self.preprocessed_dataset, pipeline_args)
            for batch in dataloader:
                embeddings: jax.numpy.ndarray = run_batch_inference(
                    batch.tokens, self.model, model_args_with_pooling_strategy
                )

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

    def test_run_batch_inference_shape_(self):
        for batch in self.dataloader:
            embeddings: jax.numpy.ndarray = run_batch_inference(
                batch.tokens, self.model, model_args
            )

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


class ComputeUserEmbeddingTallyAndMean(unittest.TestCase):
    def setUp(self):
        self.model: FlaxRobertaModel = FlaxAutoModel.from_pretrained(
            model_args.base_model_name, from_pt=True
        )

        preprocessed_dataset: Dataset = load_from_disk(test_processed_dataset_path)  # type: ignore
        self.preprocessed_dataset = preprocessed_dataset

        self.dataloader = get_dataloader(
            preprocessed_dataset, pipeline_args, include_leftovers=True
        )
        batch_size = pipeline_args.eval_per_device_batch_size * jax.device_count()
        self.num_batches = len(self.preprocessed_dataset) // batch_size

        self.user_embedding_tally = _get_uid_tally_dict(
            self.model, self.dataloader, self.num_batches, model_args
        )

        self.hf_model_config: RobertaConfig = AutoConfig.from_pretrained(
            model_args.base_model_name, from_pt=True
        )

        self.user_embedding_mean = _compute_mean_embeddings(self.user_embedding_tally)

    def test_user_embedding_tally_completeness(self):
        self.assertSetEqual(
            set(self.preprocessed_dataset["uid"]), set(self.user_embedding_tally.keys())
        )

    def test_user_embedding_tally_data_format(self):
        for uid, tweet_user in self.user_embedding_tally.items():
            self.assertIsInstance(uid, str)
            self.assertGreaterEqual(tweet_user.num_tweets_processed, 1)
            chex.assert_axis_dimension(
                tweet_user.embedding_sum, 0, self.hf_model_config.hidden_size
            )

    def test_user_embedding_mean_data_format(self):
        for uid, mean_tweet_embeddings in self.user_embedding_mean.items():
            self.assertIsInstance(uid, str)
            self.assertEqual(
                len(mean_tweet_embeddings), self.hf_model_config.hidden_size
            )

    def test_user_embedding_mean_json_compatibility(self):
        json_output = json.dumps(self.user_embedding_mean)
        self.assertIsInstance(json_output, str)
        self.assertGreater(len(json_output), 0)
