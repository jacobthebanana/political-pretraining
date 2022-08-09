from typing import (
    Any,
    Container,
    Dict,
    List,
    Tuple,
    Iterable,
    Union,
    Optional,
)
from collections import defaultdict
import multiprocessing
import json
from dataclasses import dataclass, field

import datasets
from datasets import (
    Dataset,
    load_dataset,
    load_from_disk,
    concatenate_datasets,
    Value,
    Features,
)
from transformers import AutoTokenizer, HfArgumentParser
from tqdm.auto import tqdm

from ..config import (
    ModelConfig,
    DataConfig,
    LookupByUID,
    LabelByUID,
    UserID,
    CONCATENATION_DELIMITER_MAP,
)

Dataset = datasets.arrow_dataset.Dataset


@dataclass
class _DATASET_SHARD_FOR_INDEXING:
    dataset_shard: datasets.arrow_dataset.Dataset
    offset: int
    num_shards: int


@dataclass
class _LOOKUP_DICT_SHARD_FOR_CONCATENATION:
    dataset: datasets.arrow_dataset.Dataset
    lookup_shard: LookupByUID
    model_args: ModelConfig
    data_args: DataConfig
    shard_index: int
    num_shards: int
    label_lookup: Optional[LabelByUID] = field(default=None)


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


def load_user_labels(data_args: DataConfig) -> LabelByUID:
    """
    Load user labels from the *filtered* user label csv.

    Args:
     data_args: specifies path to the filtered user label csv.
    """
    with open(data_args.filtered_label_path, "r") as filtered_label_file:
        filtered_labels = filtered_label_file.readlines()

    output: Dict[UserID, int] = {}
    for filtered_label_entry in tqdm(
        filtered_labels[1:], ncols=80, desc="Loading labels"
    ):
        entry_fields = filtered_label_entry.split(",")
        entry_label = int(entry_fields[0])
        entry_user_id = entry_fields[-1].rstrip("\n")

        output[entry_user_id] = entry_label

    return output


def filter_hf_dataset_by_uid(
    dataset: Dataset, uid_set: Container[str], data_args: DataConfig
) -> Dataset:
    def filter_function(example: Dict[str, Any]) -> bool:
        uid = example["uid"]
        return uid in uid_set

    return dataset.filter(filter_function, num_proc=data_args.num_procs)


def _concatenate_by_uid_on_shard(
    shard: _LOOKUP_DICT_SHARD_FOR_CONCATENATION,
) -> Dataset:
    tokenizer = AutoTokenizer.from_pretrained(shard.model_args.base_model_name)

    output = {"uid": [], "tid": [], "text": [], "label": []}
    features = Features(
        {
            "uid": Value(dtype="string"),
            "tid": Value(dtype="string"),
            "text": Value(dtype="string"),
            "label": Value(dtype="int8"),
        }
    )
    delimiter = CONCATENATION_DELIMITER_MAP.get(
        shard.data_args.concatenation_delimiter, ""  # type: ignore
    )
    for uid, indices in tqdm(
        shard.lookup_shard.items(),
        ncols=80,
        total=len(shard.lookup_shard.keys()),
        desc=f"Concatenating {shard.shard_index}/{shard.num_shards}",
    ):
        if shard.label_lookup:
            user_label = shard.label_lookup.get("uid", -1)
        else:
            user_label = None

        text_buffer = ""
        texts: Iterable[Union[str, None]] = shard.dataset[indices]["text"]
        leading_tid: str = shard.dataset[indices]["tid"][0]
        for text in texts:
            if text:
                user_text = text + delimiter
                new_length = len(tokenizer(text_buffer + user_text)["input_ids"])

                if new_length >= shard.model_args.max_seq_length:
                    output["uid"].append(uid)
                    output["tid"].append(leading_tid)
                    output["text"].append(text_buffer)
                    output["label"].extend([user_label])
                    text_buffer = ""
                else:
                    text_buffer += user_text

        output["uid"].append(uid)
        output["tid"].append(leading_tid)
        output["text"].append(text_buffer)
        output["label"].extend([user_label])

    return Dataset.from_dict(output, features=features)


def concatenate_by_uid(
    dataset: Dataset,
    lookup_by_uid: LookupByUID,
    model_args: ModelConfig,
    data_args: DataConfig,
    label_lookup: Optional[LabelByUID] = None,
) -> Dataset:
    """
    Given a raw text dataset, concatenate sentences from each uid,
    starting a new entry whenever the number of tokens in the batch
    exceeds model_args.max_seq_length.

    If data_args.filtered_label_path is non-empty, attach to each entry
    the label of the author.

    params:
     dataset: raw HF dataset (with both `text` and `uid`)
     lookup_by_uid: uid-to-index map. See `create_uid_lookup`.
     model_args: specifies max_seq_length.
     data_args: specifies num_procs.
     label_lookup: Optional; user-level labels to include in dataset.
    """
    uids = list(lookup_by_uid.keys())
    num_shards = data_args.num_procs
    shards: List[_LOOKUP_DICT_SHARD_FOR_CONCATENATION] = []
    shard_length: int = len(uids) // num_shards + 1

    for shard_index in tqdm(range(num_shards), ncols=80, desc="Creating shards"):
        lower_index: int = shard_index * shard_length
        upper_index: int = min(len(uids), lower_index + shard_length)
        uids_in_shard = uids[lower_index:upper_index]
        lookup_dict_shard: LookupByUID = {}
        for uid in uids_in_shard:
            lookup_dict_shard[uid] = lookup_by_uid[uid]

        lookup_shard = _LOOKUP_DICT_SHARD_FOR_CONCATENATION(
            dataset,
            lookup_dict_shard,
            model_args,
            data_args,
            shard_index=shard_index,
            num_shards=num_shards,
            label_lookup=label_lookup,
        )
        shards.append(lookup_shard)

    with multiprocessing.Pool(data_args.num_procs) as pool:
        sharded_datasets: List[Dataset] = pool.map(_concatenate_by_uid_on_shard, shards)

    return concatenate_datasets(sharded_datasets)


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
            desc=f"Indexing {shard.offset:2d}/{shard.num_shards}",
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
    for shard_index in tqdm(range(num_shards), desc="Creating shards"):
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

    if data_args.rerun_tokenization:
        raw_dataset = create_raw_hf_dataset(data_args)

        if data_args.require_labels or (data_args.filtered_label_path != ""):
            label_lookup = load_user_labels(data_args)
        else:
            label_lookup = None

        if data_args.require_labels:
            assert label_lookup is not None, "Labels are required for filtering."
            print("Unfiltered", raw_dataset)
            raw_dataset = filter_hf_dataset_by_uid(
                raw_dataset, label_lookup.keys(), data_args
            )
            print("Filtered", raw_dataset)

        if data_args.per_user_concatenation:
            dataset_for_indexing = raw_dataset.remove_columns(["text"])
            lookup_by_uid = create_uid_lookup(dataset_for_indexing, data_args)
            raw_dataset_concatenated = concatenate_by_uid(
                raw_dataset,
                lookup_by_uid,
                model_args,
                data_args,
                label_lookup=label_lookup,
            )
            processed_dataset = preprocess_and_tokenize_dataset(
                raw_dataset_concatenated, model_args, data_args
            )

        else:
            processed_dataset = preprocess_and_tokenize_dataset(
                raw_dataset, model_args, data_args
            )
        processed_dataset.save_to_disk(data_args.processed_dataset_path)
        print(processed_dataset)
    else:
        processed_dataset: Dataset = load_from_disk(
            data_args.processed_dataset_path
        )  # type: ignore
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
