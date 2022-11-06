from typing import (
    Any,
    Container,
    Dict,
    List,
    Tuple,
    Iterable,
    Union,
    Optional,
    overload,
)
from collections import defaultdict
import multiprocessing
import json
from dataclasses import dataclass, field

import numpy as np
import datasets
from datasets import (
    Dataset,
    load_dataset,
    load_from_disk,
    concatenate_datasets,
    Value,
    Features,
    dataset_dict,
)
from transformers import AutoTokenizer, HfArgumentParser
from tqdm.auto import tqdm

from ..config import (
    ModelConfig,
    DataConfig,
    DatasetFeatures,
    LookupByUID,
    LabelByUID,
    UserID,
    CONCATENATION_DELIMITER_MAP,
    NUM_TQDM_COLUMNS,
)

Dataset = datasets.arrow_dataset.Dataset
Array = np.ndarray


@dataclass
class _DATASET_SHARD_FOR_INDEXING:
    dataset_shard: datasets.arrow_dataset.Dataset
    offset: int
    num_shards: int


@dataclass
class _DATASET_SHARD_FOR_FILTERING:
    dataset_shard: datasets.arrow_dataset.Dataset
    uids_to_include: Iterable[UserID]
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


def load_labels(label_csv_path: str) -> LabelByUID:
    """
    Load user labels from the *filtered* user label csv.

    Args:
     label_csv_path: path to the label file in csv format.
    """
    with open(label_csv_path, "r") as filtered_label_file:
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


def _filter_hf_dataset_by_uid_shard(
    shard: _DATASET_SHARD_FOR_FILTERING,
) -> Optional[Dataset]:
    dataset_shard = shard.dataset_shard
    uids_to_include = shard.uids_to_include
    output = {key: [] for key in dataset_shard.column_names}

    for _, entry in enumerate(  # type: ignore
        tqdm(
            dataset_shard,
            desc="Filtering ({:2d}/{:2d})".format(shard.offset, shard.num_shards),
            ncols=80,
        )
    ):
        entry: Dict[DatasetFeatures, Any]
        uid = entry["uid"]
        if uid in uids_to_include:
            for key, value in entry.items():
                output[key].append(value)

    if len(output["uid"]) == 0:
        return None

    return Dataset.from_dict(output, features=dataset_shard.features)


def filter_hf_dataset_by_uid(
    dataset: Dataset, uid_set: Iterable[UserID], data_args: DataConfig
) -> Dataset:
    """
    Keep only dataset entries where "uid" is in the given uid_set.

    Args:
     dataset: must include the "uid" (str) feature.
     uid_set: supports membership "in".
     data_args: specifies num_procs.

    Returns:
     Dataset.
    """
    num_shards = data_args.num_procs
    shards: List[_DATASET_SHARD_FOR_FILTERING] = []
    num_shards = data_args.num_procs
    base_entry_index = 0

    for shard_index in tqdm(
        range(num_shards), ncols=80, desc="Creating filtering shards"
    ):
        dataset_shard = dataset.shard(num_shards=num_shards, index=shard_index)
        shard = _DATASET_SHARD_FOR_FILTERING(
            dataset_shard=dataset_shard,
            offset=shard_index,
            num_shards=num_shards,
            uids_to_include=set(uid_set),
        )
        shards.append(shard)
        base_entry_index += len(dataset_shard)

    sharded_filtered_datasets: List[Optional[Dataset]] = list(
        map(_filter_hf_dataset_by_uid_shard, shards)
    )

    shards_to_concatenated = []
    for dataset in sharded_filtered_datasets:  # type: ignore
        if dataset is not None:
            shards_to_concatenated.append(dataset)

    filtered_dataset = concatenate_datasets(shards_to_concatenated)  # type: ignore
    filtered_dataset: Dataset
    return filtered_dataset


def split_dataset_by_uid(
    dataset: Dataset,
    train_uid_container: Iterable[UserID],
    validation_uid_container: Iterable[UserID],
    test_uid_container: Iterable[UserID],
    data_args: DataConfig,
) -> dataset_dict.DatasetDict:
    """
    Apply train-test split to dataset based on uid lists.

    Args:
     dataset: HuggingFace dataset.
     train_uid_container: uid container supporting the "in" lookup feature.
     validation_uid_container: uid container supporting the "in" lookup feature.
     test_uid_container: uid container supporting the "in" lookup feature.
     data_args: specifies num_procs.

    Returns:
     DatasetDict: with "train" and "test" as keys.
    """
    train_dataset = filter_hf_dataset_by_uid(dataset, train_uid_container, data_args)
    validation_dataset = filter_hf_dataset_by_uid(
        dataset, validation_uid_container, data_args
    )
    test_dataset = filter_hf_dataset_by_uid(dataset, test_uid_container, data_args)

    return dataset_dict.DatasetDict(
        train=train_dataset, validation=validation_dataset, test=test_dataset
    )


def _concatenate_by_uid_on_shard(
    shard: _LOOKUP_DICT_SHARD_FOR_CONCATENATION,
) -> Dataset:
    tokenizer = AutoTokenizer.from_pretrained(shard.model_args.base_model_name)

    output = {"uid": [], "tid": [], "text": []}
    features = Features(
        {
            "uid": Value(dtype="string"),
            "tid": Value(dtype="string"),
            "text": Value(dtype="string"),
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
        text_buffer = ""
        texts: Iterable[Union[str, None]] = shard.dataset[indices]["text"]
        leading_tid: str = shard.dataset[indices]["tid"][0]
        for text in texts:
            if text:
                user_text = text + delimiter

                if shard.data_args.bag_of_words_baseline_enabled:
                    new_length = -1
                else:
                    new_length = len(tokenizer(text_buffer + user_text).input_ids)

                if new_length >= shard.model_args.max_seq_length:
                    output["uid"].append(uid)
                    output["tid"].append(leading_tid)
                    output["text"].append(text_buffer)
                    text_buffer = ""
                else:
                    text_buffer += user_text

        output["uid"].append(uid)
        output["tid"].append(leading_tid)
        output["text"].append(text_buffer)

    return Dataset.from_dict(output, features=features)


def concatenate_by_uid(
    dataset: Dataset,
    lookup_by_uid: LookupByUID,
    model_args: ModelConfig,
    data_args: DataConfig,
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
        )
        shards.append(lookup_shard)

    with multiprocessing.Pool(data_args.num_procs) as pool:
        sharded_datasets: List[Dataset] = pool.map(_concatenate_by_uid_on_shard, shards)

    return concatenate_datasets(sharded_datasets)  # type: ignore


def generate_keyword_counts(
    texts: List[str], keywords: List[str], cap: int = -1
) -> Dict[str, Array]:
    """
    Given a list of n paragraphs and m keywords,
    return the count of each keyword in each paragraph as an (n, m) array).

    Args:
     texts: list of n paragraphs, where each paragraph is a string.
     keywords: list of m words.
     cap: maximum count. Set to a negative value to turn off the cap.


    Returns:
     Array of shape (n, m), where the (j, k) entry is the number of times
     the k-th word (delimited with space) appeared in the j-th paragraph,
     capped at `cap`.
    """
    output = np.zeros((len(texts), len(keywords)), dtype=int)
    for text_index, text in enumerate(texts):
        words = text.lower().split()
        for keyword_index, keyword in enumerate(keywords):
            count = words.count(keyword)
            if cap >= 0:
                count = min(count, cap)

            output[text_index, keyword_index] = count

    return {"input_ids": output, "attention_mask": np.ones_like(output)}


def load_keyword_list(data_args: DataConfig) -> List[str]:
    """
    Load the list of bag-of-word baseline keywords
    specified in data_args.bag_of_words_keyword_csv_path.

    Args:
     data_args: specifies data_args.bag_of_words_keyword_csv_path,
     a csv file where each row is of the format "keyword","category".

    Returns:
     List[str]: list of keywords.
    """
    keywords = []
    with open(
        data_args.bag_of_words_keyword_csv_path, "r"
    ) as bag_of_words_keyword_csv_file:
        bag_of_words_keyword_lines = bag_of_words_keyword_csv_file.readlines()

    for keyword_entry in bag_of_words_keyword_lines:
        keyword = keyword_entry.split(",")[0]
        keyword = keyword.lstrip('"').rstrip('"')
        keywords.append(keyword)

    assert len(keywords) >= 1
    return keywords


@overload
def preprocess_and_tokenize_dataset(
    dataset: datasets.arrow_dataset.Dataset,
    model_args: ModelConfig,
    data_args: DataConfig,
) -> datasets.arrow_dataset.Dataset:
    ...


@overload
def preprocess_and_tokenize_dataset(
    dataset: datasets.dataset_dict.DatasetDict,
    model_args: ModelConfig,
    data_args: DataConfig,
) -> datasets.dataset_dict.DatasetDict:
    ...


def preprocess_and_tokenize_dataset(
    dataset, model_args: ModelConfig, data_args: DataConfig
):
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

        if data_args.bag_of_words_baseline_enabled:
            keyword_list = load_keyword_list(data_args)
            return generate_keyword_counts(
                texts, keyword_list, cap=data_args.bag_of_words_count_cap
            )

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


def label_dataset(
    dataset: dataset_dict.DatasetDict,
    labels: LabelByUID,
) -> dataset_dict.DatasetDict:
    """
    Add labels to a HuggingFace dataset.
    Users are matched by the "uid" feature.
    Missing users would be labelled (-1).

    Each shard is saved to a temporary dataset to reduce
    memory footprint.

    Args:
     dataset: Input dataset. Must include the "uid" feature.
     labels: User labels.

    Returns:
     DatasetDict: dataset with an additional "label" column.
    """
    output = {}
    unique_uid_stats: Dict[str, int] = {}

    for split_key, split in dataset.items():
        label_column = np.zeros(len(split))
        unique_uids = set()

        for index, entry in enumerate(
            tqdm(split, desc=f"labelling {split_key} split", ncols=80)
        ):
            uid = entry["uid"]
            unique_uids.add(uid)
            label_column[index] = int(labels.get(uid, -1))

        unique_uid_stats[split_key] = len(unique_uids)
        output[split_key] = split.add_column("label", label_column)

    print("Unique uids in each split:", unique_uid_stats)
    return dataset_dict.DatasetDict(**output)


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
    num_shards = min(data_args.num_procs, len(dataset))
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

    full_user_labels = load_labels(data_args.filtered_label_path)
    train_user_labels = load_labels(data_args.train_filtered_label_path)
    validation_user_labels = load_labels(data_args.validation_filtered_label_path)
    test_user_labels = load_labels(data_args.test_filtered_label_path)

    if data_args.rerun_tokenization:
        raw_dataset = create_raw_hf_dataset(data_args)

        if data_args.per_user_concatenation:
            dataset_for_indexing = raw_dataset.remove_columns(["text"])
            lookup_by_uid = create_uid_lookup(dataset_for_indexing, data_args)
            raw_dataset = concatenate_by_uid(
                raw_dataset,
                lookup_by_uid,
                model_args,
                data_args,
            )

        split_raw_dataset: dataset_dict.DatasetDict = split_dataset_by_uid(
            raw_dataset,
            train_user_labels.keys(),
            validation_user_labels.keys(),
            test_user_labels.keys(),
            data_args,
        )

        for label_lookup in (validation_user_labels, test_user_labels):
            for user_id, label in tqdm(label_lookup.items(), ncols=NUM_TQDM_COLUMNS):
                full_user_labels[user_id] = label

        print("Labelling")
        labelled_dataset = label_dataset(split_raw_dataset, full_user_labels)
        processed_dataset = preprocess_and_tokenize_dataset(
            labelled_dataset, model_args, data_args
        )

        processed_dataset.save_to_disk(data_args.processed_dataset_path)
        print(processed_dataset)
    else:
        processed_dataset: dataset_dict.DatasetDict = load_from_disk(
            data_args.processed_dataset_path
        )  # type: ignore

    if data_args.enable_indexing:
        dataset_for_indexing = processed_dataset["train"].remove_columns(
            ["text", "input_ids", "attention_mask"]
        )
        uid_lookup = create_uid_lookup(dataset_for_indexing, data_args)
        with open(
            data_args.processed_lookup_by_uid_json_path, "w"
        ) as uid_lookup_json_file:
            json.dump(uid_lookup, uid_lookup_json_file, indent=2)


if __name__ == "__main__":
    main()
