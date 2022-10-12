"""
Report the number of tweets under each uid.
"""
import argparse
from collections import Counter
import json

from datasets import load_from_disk
from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict
from tqdm.auto import tqdm

from ..config import DatasetFeatures, NUM_TQDM_COLUMNS


def get_stats_in_dataset_split(dataset: Dataset) -> Counter:
    """
    Return a counter of all users in a dataset split.
    """
    counter = Counter(dataset["uid"])
    return counter


def get_stats_from_dataset_dict(dataset_dict: DatasetDict) -> Counter:
    """
    Get stats from all splits of the dataset dict, combined.
    """
    counter_output = Counter()
    for dataset in tqdm(dataset_dict.values(), ncols=NUM_TQDM_COLUMNS):
        counter_output += get_stats_in_dataset_split(dataset)

    return counter_output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dict")
    parser.add_argument("output_stats_json")
    args = parser.parse_args()

    dataset_dict_path: str = args.dataset_dict
    output_stats_json_path: str = args.output_stats_json

    dataset_dict: DatasetDict = load_from_disk(dataset_dict_path)  # type: ignore
    assert isinstance(dataset_dict, DatasetDict), "Must specify a dataset dict"

    stats_counter = get_stats_from_dataset_dict(dataset_dict)

    with open(output_stats_json_path, "w") as output_stats_json_file:
        stats_dict = dict(stats_counter)
        json.dump(stats_dict, output_stats_json_file, indent=2)


if __name__ == "__main__":
    main()
