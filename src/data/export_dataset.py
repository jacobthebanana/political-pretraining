"""
Export a dataset of tweets (with uid info) to
a JSON dictionary mapping uid to input_ids.

If a uid appears in more than one tweets, the output will contain
only the most recent tweet.
"""
from typing import Dict, List, Any, Tuple, Optional
from datasets import load_from_disk
from datasets.dataset_dict import DatasetDict
from datasets.arrow_dataset import Dataset
import argparse
import json

from tqdm.auto import tqdm

from ..config import DatasetFeatures, NUM_TQDM_COLUMNS


def extract_key_and_value(
    dataset_entry: Dict[DatasetFeatures, Any]
) -> Tuple[str, List[int]]:
    """
    Retrieve the uid and input_ids fields.
    """
    assert "uid" in dataset_entry.keys()
    assert "input_ids" in dataset_entry.keys()

    uid: str = dataset_entry["uid"]
    input_ids: List[int] = dataset_entry["input_ids"]
    return uid, input_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dataset_path")
    parser.add_argument("output_json_path")
    parser.add_argument("--splits", nargs="+", required=False)

    args = parser.parse_args()
    input_dataset_path: str = args.input_dataset_path
    output_json_path: str = args.output_json_path
    splits: Optional[List[str]] = args.splits

    # Load dataset
    dataset_dict = load_from_disk(input_dataset_path)  # type: ignore
    dataset_dict: DatasetDict
    if splits is None:
        splits = list(dataset_dict.keys())

    # Initialize output dictionary Dict[str, List[int]]
    output: Dict[str, List[int]] = {}

    # Iterate through dataset, extracting uid and input_ids
    # Update output dictionary if necessary
    for split_name in splits:
        dataset = dataset_dict[split_name]
        for entry in tqdm( # type: ignore
            dataset, ncols=NUM_TQDM_COLUMNS, desc="Exporting {}".format(split_name)
        ):
            entry: Dict[DatasetFeatures, Any]
            key, value = extract_key_and_value(entry)
            output[key] = value

    # Save dictionary to JSON file.
    with open(output_json_path, "w") as output_json_file:
        json.dump(output, output_json_file, indent=2)


if __name__ == "__main__":
    main()
