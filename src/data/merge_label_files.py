from transformers.hf_argparser import HfArgumentParser
import pandas as pd
from collections import Counter

from ..config import DataConfig


def main():
    """
    Merge true label file with uid file by "screen_name".
    """
    parser = HfArgumentParser((DataConfig,))
    (data_args,) = parser.parse_args_into_dataclasses()
    data_args: DataConfig

    labels = pd.read_json(data_args.raw_true_label_jsonl_path, lines=True)
    screen_names = pd.read_csv(
        data_args.scree_name_to_uid_tsv_path,
        delimiter="\t",
    )
    uid_labels = screen_names.merge(labels, on="screen_name")

    filtered_labels = uid_labels[["true_label", "screen_name", "user_id"]]
    filtered_labels = filtered_labels.dropna(inplace=False, subset=["true_label"])
    filtered_labels["true_label"].map(int)

    print(Counter(filtered_labels["true_label"]))

    filtered_labels.to_csv(data_args.processed_true_label_path, sep=",", index=False)
    print("Writing filtered label file to", data_args.processed_true_label_path)


if __name__ == "__main__":

    main()
