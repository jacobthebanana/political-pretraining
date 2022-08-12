import unittest
from typing import Dict, Union, Any
import json

import datasets
from datasets import load_from_disk

from ..data.make_dataset import (
    create_raw_hf_dataset,
    filter_hf_dataset_by_uid,
    preprocess_and_tokenize_dataset,
    create_uid_lookup,
    _concatenate_by_uid_on_shard,
    concatenate_by_uid,
    load_labels,
    split_dataset_by_uid,
    label_dataset,
    _LOOKUP_DICT_SHARD_FOR_CONCATENATION,
)
from ..data.filter_label_file import shuffle_labels, split_labels
from ..config import ModelConfig, DataConfig, DatasetFeatures

Dataset = datasets.arrow_dataset.Dataset


test_tweet_json_path = "data/testing/raw/tweets.json"
test_uid_set = set(["1180684225097289729", "20011085", "0"])

model_args = ModelConfig(max_seq_length=512)
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


class LoadUserLabels(unittest.TestCase):
    def setUp(self):
        self.user_labels = load_labels(data_args.train_filtered_label_path)

    def test_user_label_data_types(self):
        for key, value in self.user_labels.items():
            self.assertIsInstance(key, str)
            self.assertIsInstance(value, int)


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


class SplitDatasetByUID(unittest.TestCase):
    def setUp(self):
        self.train_lookup = load_labels(data_args.train_filtered_label_path)
        self.test_lookup = load_labels(data_args.test_filtered_label_path)
        self.dataset: Dataset = create_raw_hf_dataset(data_args)

        self.split_dataset = split_dataset_by_uid(
            self.dataset, self.train_lookup.keys(), self.test_lookup.keys(), data_args
        )

    def test_split_dataset_key_and_members(self):
        for split_key, split_uids in [
            ("train", self.train_lookup),
            ("test", self.test_lookup),
        ]:
            self.assertIn(split_key, self.split_dataset.keys())
            dataset_split = self.split_dataset[split_key]

            for entry in dataset_split:  # type: ignore
                entry: Dict[DatasetFeatures, Any]
                uid = entry["uid"]
                self.assertIn(uid, split_uids)


class LabelDataset(unittest.TestCase):
    def setUp(self):
        self.full_labels = load_labels(data_args.filtered_label_path)
        self.split_dataset = load_from_disk(data_args.processed_dataset_path)  # type: ignore
        self.split_dataset: datasets.dataset_dict.DatasetDict

    def test_label_dataset(self):
        labelled_dataset = label_dataset(
            self.split_dataset, self.full_labels, data_args
        )
        for entry in labelled_dataset:
            entry: Dict[DatasetFeatures, Any]
            label = entry["label"]
            uid = entry["uid"]

            reference_label = self.full_labels.get(uid)
            self.assertEqual(label, reference_label)


class ShuffleAndSplitLabels(unittest.TestCase):
    def setUp(self):
        with open(data_args.filtered_label_path, "r") as filtered_label_file:
            self.filtered_labels = filtered_label_file.readlines()

        self.shuffled_labels = shuffle_labels(self.filtered_labels, data_args)
        self.train_labels, self.test_labels = split_labels(
            self.filtered_labels, data_args
        )

    def test_output_length(self):
        self.assertEqual(len(self.shuffled_labels), len(self.filtered_labels))
        self.assertEqual(
            len(self.train_labels) + len(self.test_labels), len(self.filtered_labels)
        )


class ConcatenateByUID(unittest.TestCase):
    def setUp(self):
        self.dataset = create_raw_hf_dataset(data_args)
        self.lookup_by_uid = create_uid_lookup(self.dataset, data_args)

    def test_concatenate_by_uid_shard(self):
        lookup_shard = _LOOKUP_DICT_SHARD_FOR_CONCATENATION(
            dataset=self.dataset,
            lookup_shard=self.lookup_by_uid,
            model_args=model_args,
            data_args=data_args,
            shard_index=0,
            num_shards=1,
        )
        dataset_output = _concatenate_by_uid_on_shard(lookup_shard)
        for text in dataset_output["text"]:
            self.assertLessEqual(len(text.split()), model_args.max_seq_length)

    def test_concatenate_by_uid(self):
        dataset_output = concatenate_by_uid(
            self.dataset, self.lookup_by_uid, model_args, data_args
        )
        for text in dataset_output["text"]:
            self.assertLessEqual(len(text.split()), model_args.max_seq_length)


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
