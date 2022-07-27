from typing import Any, Container, Dict, List, Tuple, Iterable
from collections import defaultdict
import multiprocessing
import json
from dataclasses import dataclass

import datasets
from datasets import Dataset, load_dataset, Value, Features
from transformers import AutoTokenizer, HfArgumentParser
from tqdm.auto import tqdm

from ..config import ModelConfig, DataConfig, LookupByUID, UserID

Dataset = datasets.arrow_dataset.Dataset


@dataclass
class _DATASET_SHARD_FOR_INDEXING:
    dataset_shard: datasets.arrow_dataset.Dataset
    offset: int
    num_shards: int


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


def _create_uid_lookup_on_shard(
    shard: _DATASET_SHARD_FOR_INDEXING,
) -> LookupByUID:
    """
    Return a dictionary for looking up tweet indices in the given dataset
    shard by the uid of the author.

    Args:
     dataset: HuggingFace dataset, either raw or processed.
     base_index: Index of the first entry in the shard in the full dataset.

    Returns:
     Dictionary mapping uid strings to arrays of dataset indices.
    """
    lookup_dictionary: defaultdict[str, List[int]] = defaultdict(list)
    for index, dataset_entry in enumerate(
        tqdm(
            shard.dataset_shard,
            desc=f"Lookup {shard.offset}/{shard.num_shards}",
            ncols=80,
        )
    ):
        uid = dataset_entry["uid"]  # type: ignore
        lookup_dictionary[uid].append(index * shard.num_shards + shard.offset)

    output: Dict[UserID, Tuple[int, ...]] = {}
    for uid, dataset_indices in tqdm(lookup_dictionary.items(), leave=False):
        output[uid] = tuple(dataset_indices)

    return output


def _merge_lookup_shards(lookup_shards: List[LookupByUID]) -> LookupByUID:
    """
    Merge lookup dictionaries by UID.
    """
    lookup_merged: Dict[str, List[int]] = defaultdict(list)
    for lookup_shard in tqdm(lookup_shards, desc="Merging lookups"):
        for uid, indices in lookup_shard.items():
            lookup_merged[uid].extend(indices)

    lookup_output: LookupByUID = {}
    for uid, indices in lookup_merged.items():
        lookup_output[uid] = tuple(indices)

    return lookup_output


def create_uid_lookup(
    dataset: datasets.arrow_dataset.Dataset,
    data_args: DataConfig,
) -> LookupByUID:
    """
    Return a dictionary for looking up tweet indices in the given dataset
    by the uid of the author. Runs in parallel.

    Args:
     dataset: HuggingFace dataset, either raw or processed.
     data_args: Provides num_procs details.

    Returns:
     Dictionary mapping uid strings to arrays of dataset indices.
    """
    shards: List[_DATASET_SHARD_FOR_INDEXING] = []
    num_shards = data_args.num_procs
    base_entry_index = 0
    for shard_index in tqdm(range(num_shards), desc="Creating shards."):
        dataset_shard = dataset.shard(num_shards=num_shards, index=shard_index)
        shard = _DATASET_SHARD_FOR_INDEXING(
            dataset_shard=dataset_shard, offset=shard_index, num_shards=num_shards
        )
        shards.append(shard)
        base_entry_index += len(dataset_shard)

    with multiprocessing.Pool(data_args.num_procs) as pool:
        lookup_by_uid_sharded: List[LookupByUID] = pool.map(
            _create_uid_lookup_on_shard, shards
        )

    return _merge_lookup_shards(lookup_by_uid_sharded)


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

    if data_args.enable_indexing:
        dataset_for_indexing = processed_dataset.remove_columns(
            ["text", "input_ids", "attention_mask"]
        )
        uid_lookup = create_uid_lookup(dataset_for_indexing, data_args)
        with open(
            data_args.processed_lookup_by_uid_json_path, "w"
        ) as uid_lookup_json_file:
            json.dump(uid_lookup, uid_lookup_json_file, indent=2)


if __name__ == "__main__":
    main()
