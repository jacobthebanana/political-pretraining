from typing import Any, Container, Dict

import pandas as pd
import datasets
from datasets import Dataset
from transformers import AutoTokenizer, HfArgumentParser

from ..config import ModelConfig, DataConfig

Dataset = datasets.arrow_dataset.Dataset


def create_raw_hf_dataset(data_args: DataConfig) -> Dataset:
    """
    Create HuggingFace dataset from tweet CSV.

    params:
     csv_path: path to tweet csv file.

    output:
     raw dataset.
    """
    df = pd.read_csv(data_args.csv_path, dtype={"uid": str, "tid": str})
    dataset = Dataset.from_pandas(df)
    return dataset


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
        return tokenizer(
            examples["text"],
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
