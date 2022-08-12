"""
Utilities for filtering out user_label entries where 
classifier_label is unavailable.
"""
from typing import List, Dict, Tuple, Union
import json

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


def split_labels(
    labels: List[str], data_args: DataConfig
) -> Tuple[List[str], List[str]]:
    """
    Split labels into training set and testing set.

    Args:
     labels: list of label file rows, excluding header row.
     data_args: specifies test_ratio.

    Returns:
     Tuple[List[str], List[str]]: train_rows, test_rows.
    """
    num_entries = len(labels)
    num_test_entries = int(num_entries * data_args.test_ratio)
    num_train_entries = num_entries - num_test_entries

    return labels[:num_train_entries], labels[num_train_entries:]


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

    """
    parser = HfArgumentParser((DataConfig,))
    (data_args,) = parser.parse_args_into_dataclasses()
    data_args: DataConfig

    print("Loading raw label file input.")
    with open(data_args.raw_label_path, "r") as raw_label_file:
        raw_labels = raw_label_file.readlines()

    filtered_labels: List[str] = []
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
    print(f"Writing filtered label file output; number of entries: {num_entries}.")
    with open(data_args.filtered_label_path, "w") as filtered_label_file:
        filtered_label_file.writelines([raw_labels[0]] + filtered_labels)

    shuffled_labels = shuffle_labels(filtered_labels, data_args)
    train_labels, test_labels = split_labels(shuffled_labels, data_args)

    print(
        f"Writing training label file output; number of entries: {len(train_labels)}."
    )
    with open(data_args.train_filtered_label_path, "w") as train_label_file:
        train_label_file.writelines([raw_labels[0]] + train_labels)

    print(f"Writing testing label file output; number of entries: {len(test_labels)}.")
    with open(data_args.test_filtered_label_path, "w") as test_label_file:
        test_label_file.writelines([raw_labels[0]] + test_labels)

    label_id_to_label_text: Dict[int, str] = {}
    for label_text, label_id in label_text_to_label_id.items():
        label_id_to_label_text[label_id] = label_text

    with open(
        data_args.label_id_to_label_text_path, "w"
    ) as label_id_to_label_text_file:
        json.dump(label_id_to_label_text, label_id_to_label_text_file, indent=2)


if __name__ == "__main__":
    main()
