"""
Utilities for filtering out user_label entries where 
classifier_label is unavailable and splitting labelled data. 
"""
from typing import List, Dict, Tuple
import json
import os

import numpy as np
import jax
from transformers.hf_argparser import HfArgumentParser
from tqdm.auto import tqdm

from ..config import DataConfig


def shuffle_labels(labels: List[str], data_args: DataConfig) -> List[str]:
    """
    Shuffle labels using JAX prng key.

    Args:
     labels: list of label file rows, excluding header row.
     data_args: specifies JAX Threefry prng key.

    Returns:
     shuffled list of labels.
    """
    num_entries = len(labels)

    prng_key = jax.random.PRNGKey(data_args.train_test_split_prng_seed)
    indices = jax.numpy.arange(num_entries)
    shuffled_indices = jax.random.permutation(prng_key, indices, independent=True)

    print("Shuffling")
    label_array = np.array(labels)
    shuffled_labels = label_array[shuffled_indices]

    return list(shuffled_labels)


def split_labels(labels: List[str], test_ratio: float) -> Tuple[List[str], List[str]]:
    """
    Split labels into training set and testing set.

    Args:
     labels: list of label file rows, excluding header row.
     test_ratio: specifies ratio of the test split.

    Returns:
     Tuple[List[str], List[str]]: train_rows, test_rows.
    """
    num_entries = len(labels)
    num_test_entries = int(num_entries * test_ratio)
    num_train_entries = num_entries - num_test_entries

    return labels[:num_train_entries], labels[num_train_entries:]


def get_user_id(label_entry: str) -> str:
    """
    Return the user id (str)field of the given label entry.
    """
    fields = label_entry.split(",")
    return fields[-1].rstrip("\n")


def exclude_users(labels: List[str], labels_to_exclude: List[str]) -> List[str]:
    """
    Exclude users that are in the labels_to_exclude list of labels.

    Args:
     labels: list of full label file rows, excluding header row.
     labels_to_exclude: list of label file rows with users to exclude,
        also without the header row.

    Returns:
     List[str]: list of full label file rows minus ones that are excluded.
    """
    excluded_users = set()
    for label_row in tqdm(labels_to_exclude, ncols=80, desc="Loading excluded users"):
        user_id = get_user_id(label_row)
        excluded_users.add(user_id)

    output: List[str] = []
    for label_row in tqdm(labels, ncols=80, desc="Filtering users"):
        user_id = get_user_id(label_row)

        if user_id not in excluded_users:
            output.append(label_row)

    return output


def main():
    """
    Create new label csv file where the first field
    (classifier label) of all entries must be non-empty.
    Apply train-test split to the labels.

    Args:
     data_args.raw_label_path: path to input label csv.
     data_args.filtered_label_path: path to output label csv.
     data_args.train_filtered_label_path: path to output train label csv.
     data_args.test_filtered_label_path: path to output test label csv.
     data_args.test_ratio: test ratio.
     data_args.split_prng_seed: JAX Threefry PRNG key seed.
     data_args.use_true_label_for_test_split: If set, use true labels for test
        and split the non-true label users into train and validation.

    """
    parser = HfArgumentParser((DataConfig,))
    (data_args,) = parser.parse_args_into_dataclasses()
    data_args: DataConfig

    print("Loading raw label file input.")
    with open(data_args.raw_label_path, "r") as raw_label_file:
        raw_labels = raw_label_file.readlines()

    filtered_labels: List[str] = []

    label_id_to_label_text_path = data_args.label_text_to_label_id_path
    if os.path.exists(label_id_to_label_text_path):
        with open(label_id_to_label_text_path, "r") as label_id_to_label_text_file:
            label_text_to_label_id = json.load(label_id_to_label_text_file)
    else:
        label_text_to_label_id: Dict[str, int] = {}
    label_id: int = -1

    for raw_label_entry in tqdm(raw_labels[1:], desc="Filtering labels", ncols=80):
        entry_fields: List[str] = raw_label_entry.split(",")
        entry_label = entry_fields[0]

        if entry_label != "":
            label_id = label_text_to_label_id.get(entry_label, label_id + 1)
            label_text_to_label_id[entry_label] = label_id
            updated_entry_fields: Tuple[str, ...] = (
                str(label_id),
                *entry_fields[1:],
            )
            filtered_labels.append(",".join(updated_entry_fields))

    num_entries = len(filtered_labels)
    print(
        f"Writing filtered label file output; number of entries: {num_entries}.",
        data_args.filtered_label_path,
    )
    with open(data_args.filtered_label_path, "w") as filtered_label_file:
        filtered_label_file.writelines([raw_labels[0]] + filtered_labels)

    if data_args.use_true_label_for_test_split:
        with open(data_args.processed_true_label_path, "r") as true_label_file:
            true_labels = true_label_file.readlines()[1:]

        train_val_labels = exclude_users(filtered_labels, true_labels)
        train_labels, validation_labels = split_labels(
            train_val_labels, data_args.validation_ratio
        )
        test_labels = true_labels
    else:
        test_ratio = data_args.test_ratio

        # Ensure that validation ratio refers to the fraction of the entire data set.
        validation_ratio = data_args.validation_ratio / (1 - test_ratio)

        shuffled_labels = shuffle_labels(filtered_labels, data_args)
        train_val_labels, test_labels = split_labels(shuffled_labels, test_ratio)
        train_labels, validation_labels = split_labels(
            train_val_labels, validation_ratio
        )

    print(
        f"Writing training label file output; number of entries: {len(train_labels)}.",
        data_args.train_filtered_label_path,
    )
    with open(data_args.train_filtered_label_path, "w") as train_label_file:
        train_label_file.writelines([raw_labels[0]] + train_labels)

    print(
        "Writing validation label file output; number of entries:",
        f"{len(validation_labels)}.",
        data_args.validation_filtered_label_path,
    )
    with open(data_args.validation_filtered_label_path, "w") as validation_label_file:
        validation_label_file.writelines([raw_labels[0]] + validation_labels)

    print(
        f"Writing testing label file output; number of entries: {len(test_labels)}.",
        data_args.test_filtered_label_path,
    )
    with open(data_args.test_filtered_label_path, "w") as test_label_file:
        test_label_file.writelines([raw_labels[0]] + test_labels)

    label_id_to_label_text: Dict[int, str] = {}
    for label_text, label_id in label_text_to_label_id.items():
        label_id_to_label_text[label_id] = label_text

    with open(
        data_args.label_text_to_label_id_path, "w"
    ) as label_label_text_to_id_file:
        json.dump(label_text_to_label_id, label_label_text_to_id_file, indent=2)


if __name__ == "__main__":
    main()
