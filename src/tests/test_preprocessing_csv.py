import os.path
import unittest

import json
import datasets
from tqdm.auto import tqdm

from ..data.make_dataset import (
    create_raw_hf_dataset,
    filter_hf_dataset_by_uid,
    preprocess_and_tokenize_dataset,
    create_uid_lookup,
)
from ..config import ModelConfig, DataConfig

Dataset = datasets.arrow_dataset.Dataset


test_tweet_csv_path = "data/testing/raw/tweets.csv"
test_uid_set = set(["611113833", "1284651818782535680", "0"])

model_args = ModelConfig()
data_args = DataConfig(source_path=test_tweet_csv_path)

# Regression test: handle examples where text is None.
if os.path.isfile(test_tweet_csv_path):
    with open(test_tweet_csv_path, "a") as test_tweet_csv_file:
        test_tweet_csv_file.write("7,11,2022-07-21T09:35:15.000Z,,\n")


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


class CreateDatasetShardFromRawText(unittest.TestCase):
    def setUp(self):
        data_args_with_sharding = DataConfig(
            source_path=test_tweet_csv_path, shard_denominator=2
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
        self.assertEqual(len(self.filtered_dataset), 2)
        self.assertIn("611113833", self.filtered_dataset["uid"])
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


class CreateUidLookupTable(unittest.TestCase):
    def setUp(self):
        self.dataset = create_raw_hf_dataset(data_args)
        self.uid_lookup = create_uid_lookup(self.dataset, data_args)

    def test_lookup_table_completeness(self):
        num_lookup_entries = sum(map(len, self.uid_lookup.values()))
        self.assertEqual(num_lookup_entries, len(self.dataset))

    def test_lookup_table_accuracy(self):
        for uid, dataset_indices in tqdm(self.uid_lookup.items(), leave=False):
            matched_entries = self.dataset[dataset_indices]
            matched_uids = matched_entries["uid"]
            for matched_uid in matched_uids:
                self.assertEqual(matched_uid, uid)

    def test_lookup_table_json_compatibility(self):
        lookup_table_json: str = json.dumps(self.uid_lookup)
        self.assertIsNotNone(lookup_table_json)
