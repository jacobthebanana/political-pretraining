import unittest
from typing import Dict, Union
import json

import datasets

from ..data.make_dataset import (
    create_raw_hf_dataset,
    filter_hf_dataset_by_uid,
    preprocess_and_tokenize_dataset,
)
from ..config import ModelConfig, DataConfig

Dataset = datasets.arrow_dataset.Dataset


test_tweet_json_path = "data/testing/raw/tweets.json"
test_uid_set = set(["1180684225097289729", "20011085", "0"])

model_args = ModelConfig()
data_args = DataConfig(source_format="json", source_path=test_tweet_json_path)

# Regression test: handle examples where text is None.
with open(test_tweet_json_path, "a") as test_tweet_csv_file:
    example_null_entry: Dict[str, Union[str, int, None]] = {
        "tweet_id": "7",
        "user_id": "11",
        "text": None,
        "created_at": 1658552020000,
    }
    example_null_entry_string = json.dumps(example_null_entry)
    test_tweet_csv_file.write(example_null_entry_string + "\n")


class CreateDatasetFromRawText(unittest.TestCase):
    def setUp(self):
        self.dataset = create_raw_hf_dataset(data_args)

    def test_dataset_features(self):
        for feature in ["tid", "uid", "text"]:
            self.assertIn(feature, self.dataset.features)

    def test_dataset_datatypes(self):
        example_entry = self.dataset[0]
        self.assertIsInstance(example_entry["text"], str)
        self.assertIsInstance(example_entry["tid"], str)
        self.assertIsInstance(
            example_entry["uid"], str, "Must be string to avoid truncation."
        )

    def test_dataset_length(self):
        with open(test_tweet_json_path, "r") as test_tweet_json_file:
            num_test_entries = len(test_tweet_json_file.readlines())

        self.assertEqual(len(self.dataset), num_test_entries)


class CreateDatasetShardFromRawText(unittest.TestCase):
    def setUp(self):
        data_args_with_sharding = DataConfig(
            source_format="json", source_path=test_tweet_json_path, shard_denominator=2
        )
        self.sharded_dataset = create_raw_hf_dataset(data_args_with_sharding)
        self.full_dataset = create_raw_hf_dataset(data_args)

    def test_sharded_dataset_length(self):
        self.assertEqual(len(self.sharded_dataset), len(self.full_dataset) // 2)


class FilterDatasetByUID(unittest.TestCase):
    def setUp(self):
        self.dataset: Dataset = create_raw_hf_dataset(data_args)
        self.filtered_dataset: Dataset = filter_hf_dataset_by_uid(
            self.dataset, test_uid_set, data_args
        )

    def test_filtered_dataset_length(self):
        self.assertEqual(len(self.filtered_dataset), 181)
        self.assertIn("1180684225097289729", self.filtered_dataset["uid"])
        self.assertNotIn("0", self.filtered_dataset["uid"])


class PreprocessAndTokenizeDataset(unittest.TestCase):
    def setUp(self):
        self.dataset = create_raw_hf_dataset(data_args)
        self.preprocessed_dataset = preprocess_and_tokenize_dataset(
            self.dataset, model_args, data_args
        )

    def test_preprocessed_dataset_length(self):
        self.assertEqual(
            len(self.dataset),
            len(self.preprocessed_dataset),
            "Tokenizer shouldn't change dataset length.",
        )

        for feature in ["input_ids", "attention_mask"]:
            self.assertIn(feature, self.preprocessed_dataset.features)
