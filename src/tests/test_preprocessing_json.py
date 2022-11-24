from os import path
import unittest
from typing import Dict, Union, Any
import json

import datasets
from datasets import load_from_disk
from transformers import AutoTokenizer

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
    generate_keyword_counts,
    load_keyword_list,
    _LOOKUP_DICT_SHARD_FOR_CONCATENATION,
)
from ..data.filter_label_file import shuffle_labels, split_labels, exclude_users
from ..config import (
    ModelConfig,
    DataConfig,
    DatasetFeatures,
    DEFAULT_HF_MODEL,
    CONCATENATION_DELIMITER_MAP,
)

Dataset = datasets.arrow_dataset.Dataset


test_tweet_json_path = "data/testing/raw/tweets.json"
test_uid_set = set(["1180684225097289729", "20011085", "0"])

model_args = ModelConfig(max_seq_length=512)
data_args = DataConfig(source_format="json", source_path=test_tweet_json_path)

# Regression test: handle examples where text is None.
if path.isfile(test_tweet_json_path):
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
        self.validation_lookup = load_labels(data_args.validation_filtered_label_path)
        self.test_lookup = load_labels(data_args.test_filtered_label_path)
        self.dataset: Dataset = create_raw_hf_dataset(data_args)

        self.split_dataset = split_dataset_by_uid(
            self.dataset,
            self.train_lookup.keys(),
            self.validation_lookup.keys(),
            self.test_lookup.keys(),
            data_args,
        )

    def test_split_dataset_key_and_members(self):
        for split_key, split_uids in [
            ("train", self.train_lookup),
            ("validation", self.train_lookup),
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
        labelled_dataset = label_dataset(self.split_dataset, self.full_labels)
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
            self.filtered_labels, data_args.test_ratio
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

    def test_tokenizer_sep_token_handling(self):
        tokenizer = AutoTokenizer.from_pretrained(DEFAULT_HF_MODEL)
        example_words = ["apple", "banana"]

        sep_concatenated_input = CONCATENATION_DELIMITER_MAP["sep"].join(example_words)

        tokenizer_output = tokenizer(sep_concatenated_input)
        tokenizer_output_ref = tokenizer(example_words[0], example_words[1])

        self.assertListEqual(
            tokenizer_output.input_ids,
            tokenizer_output_ref.input_ids,
            "Tokenizer output mismatch: "
            + str((tokenizer_output.tokens(), tokenizer_output_ref.tokens())),
        )


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


class FilterLabels(unittest.TestCase):
    def setUp(self):
        example_full_labels = [
            "0,username_a,7",
            "1,username_a,11",
            "0,username_b,5",
        ]
        example_excluded_labels = ["1,username_b,5", "0,username_a,6"]

        self.filtered_examples = exclude_users(
            example_full_labels, example_excluded_labels
        )

    def test_excluded_users(self):
        self.assertEqual(len(self.filtered_examples), 2)

        self.assertIn("0,username_a,7", self.filtered_examples)
        self.assertIn("1,username_a,11", self.filtered_examples)
        self.assertNotIn("0,username_b,5", self.filtered_examples)


class LoadKeywords(unittest.TestCase):
    def setUp(self):
        self.keyword_list = load_keyword_list(data_args)

    def test_keyword_list_length(self):
        for keyword in self.keyword_list:
            self.assertNotIn('"', keyword)


class GenerateKeywordCounts(unittest.TestCase):
    def setUp(self):
        self.example_texts = [
            "pineapple apple banana",
            "apple banana peach",
            "pineapple apple pineapple",
            "banana peach peach",
            "apple peach banana",
        ]
        self.keywords = ["apple", "pineapple"]

    def test_count_array_shape(self):
        keyword_count_output = generate_keyword_counts(
            self.example_texts, self.keywords
        )
        count_array = keyword_count_output["input_ids"]
        self.assertEqual(count_array.shape[0], len(self.example_texts))
        self.assertEqual(count_array.shape[1], len(self.keywords))

    def test_count_without_cap(self):
        keyword_count_output = generate_keyword_counts(
            self.example_texts, self.keywords
        )
        count_array = keyword_count_output["input_ids"]
        self.assertListEqual(
            count_array[:, 0].tolist(), [1, 1, 1, 0, 1], "count of 'apple' without cap"
        )
        self.assertListEqual(
            count_array[:, 1].tolist(),
            [1, 0, 2, 0, 0],
            "count of 'pineapple without cap",
        )

    def test_count_with_cap_enabled(self):
        keyword_count_output = generate_keyword_counts(
            self.example_texts, self.keywords
        )
        count_array = keyword_count_output["input_ids"]
        self.assertListEqual(
            count_array[:, 0].tolist(), [1, 1, 1, 0, 1], "count of 'apple' with cap 1"
        )
        self.assertListEqual(
            count_array[:, 1].tolist(),
            [1, 0, 1, 0, 0],
            "count of 'pineapple' with cap 1",
        )
