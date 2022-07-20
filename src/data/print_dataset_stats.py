from collections import Counter
import json

import datasets
from datasets import load_from_disk
from transformers import HfArgumentParser

from ..config import DataConfig

Dataset = datasets.arrow_dataset.Dataset


def get_unique_uids(dataset: Dataset) -> Counter:
    uids = dataset["uid"]
    return Counter(uids)


def main():
    parser = HfArgumentParser((DataConfig,))
    (data_args,) = parser.parse_args_into_dataclasses()
    data_args: DataConfig

    dataset: Dataset = load_from_disk(data_args.processed_dataset_path)  # type: ignore
    uids_stats = get_unique_uids(dataset)

    print(json.dumps(uids_stats, indent=2))


if __name__ == "__main__":
    main()
