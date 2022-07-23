from typing import Any, Container, Dict

import pandas as pd
import datasets
from datasets import Dataset, load_dataset, Value, Features
from transformers import AutoTokenizer, HfArgumentParser

from ..config import ModelConfig, DataConfig

Dataset = datasets.arrow_dataset.Dataset


def create_raw_hf_dataset(data_args: DataConfig) -> Dataset:
    """
    Create HuggingFace dataset from tweet CSV.

    output:
     raw dataset.
    """
    if data_args.source_format == "csv":
        csv_features = Features(
            {
                "uid": Value(dtype="string"),
                "tid": Value(dtype="string"),
                "text": Value(dtype="string"),
                "created_at": Value(dtype="string"),
            }
        )
        dataset_dict = load_dataset(
            "csv", data_files=data_args.source_path, features=csv_features
        )
    else:  # json
        dataset_dict = load_dataset("json", data_files=data_args.source_path)
        dataset_dict = dataset_dict.rename_columns(
            {"user_id": "uid", "tweet_id": "tid"}
        )

    dataset: Dataset = dataset_dict["train"]  # type: ignore

    num_shards = data_args.shard_denominator
    shard_index = num_shards - 1
    sharded_dataset = dataset.shard(num_shards=num_shards, index=shard_index)

    return sharded_dataset


def filter_hf_dataset_by_uid(
    dataset: Dataset, uid_set: Container[str], data_args: DataConfig
) -> Dataset:
    def filter_function(example: Dict[str, Any]) -> bool:
        uid = example["uid"]
        return uid in uid_set

    return dataset.filter(filter_function, num_proc=data_args.num_procs)


def preprocess_and_tokenize_dataset(
    dataset: datasets.arrow_dataset.Dataset,
    model_args: ModelConfig,
    data_args: DataConfig,
) -> datasets.arrow_dataset.Dataset:
    """
    Preprocess and tokenize the raw dataset

    params:
     dataset: raw HF dataset.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_args.base_model_name)

    def preprocess_function(examples):
        texts = list(examples["text"])

        # Handle instances where text might be None.
        for text_index in range(len(texts)):
            text = texts[text_index]
            if not isinstance(text, str):
                texts[text_index] = " "

        return tokenizer(
            texts,
            padding="max_length",
            max_length=model_args.max_seq_length,
            truncation=True,
        )

    processed_dataset = dataset.map(
        preprocess_function,
        batched=True,
        num_proc=data_args.num_procs,
    )
    return processed_dataset


def main():
    parser = HfArgumentParser((ModelConfig, DataConfig))
    model_args, data_args = parser.parse_args_into_dataclasses()
    model_args: ModelConfig
    data_args: DataConfig

    raw_dataset = create_raw_hf_dataset(data_args)
    processed_dataset = preprocess_and_tokenize_dataset(
        raw_dataset, model_args, data_args
    )
    processed_dataset.save_to_disk(data_args.processed_dataset_path)


if __name__ == "__main__":
    main()
