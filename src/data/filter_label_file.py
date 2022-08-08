"""
Utilities for filtering out user_label entries where 
classifier_label is unavailable.
"""
from typing import List, Dict, Tuple, Union
import json

from transformers.hf_argparser import HfArgumentParser
from tqdm.auto import tqdm

from ..config import DataConfig


def main():
    """
    Create new label csv file where the first field
    (classifier label) of all entries must be non-empty.

    Args:
     data_args.raw_label_path: path to input label csv.
     data_args.filtered_label_path: path to output label csv.
    """
    parser = HfArgumentParser((DataConfig,))
    (data_args,) = parser.parse_args_into_dataclasses()
    data_args: DataConfig

    print("Loading raw label file input.")
    with open(data_args.raw_label_path, "r") as raw_label_file:
        raw_labels = raw_label_file.readlines()

    filtered_labels: List[str] = [raw_labels[0]]
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
        filtered_label_file.writelines(filtered_labels)

    label_id_to_label_text: Dict[int, str] = {}
    for label_text, label_id in label_text_to_label_id.items():
        label_id_to_label_text[label_id] = label_text

    with open(
        data_args.label_id_to_label_text_path, "w"
    ) as label_id_to_label_text_file:
        json.dump(label_id_to_label_text, label_id_to_label_text_file, indent=2)


if __name__ == "__main__":
    main()
