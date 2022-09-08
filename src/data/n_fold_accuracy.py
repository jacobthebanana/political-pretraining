"""
Evaluate per-user predictions on a specific split of an 
n-fold test set.
"""
from typing import Dict, List
import argparse
import json
from socket import gethostname
import datetime


import wandb
from tqdm.auto import tqdm

from . import load_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label_csv_paths", nargs="+")
    parser.add_argument("--fold_json_path")
    parser.add_argument("--prediction_json_path")
    parser.add_argument("--fold_key", type=str)
    parser.add_argument("--wandb_entity", type=str)
    parser.add_argument("--wandb_project", type=str)
    args = parser.parse_args()

    label_csv_paths: List[str] = args.label_csv_paths
    fold_json_path: str = args.fold_json_path
    prediction_json_path: str = args.prediction_json_path
    fold_key: str = args.fold_key

    wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        name=datetime.datetime.now().isoformat() + "-" + gethostname(),
    )
    wandb.config.update({"eval_args": args.__dict__, "fold_key": fold_key})

    # Load label file
    user_labels: Dict[str, int] = {}
    for label_csv_path in label_csv_paths:
        user_labels_subset: Dict[str, int] = load_labels(label_csv_path)
        for user_id, label in user_labels_subset.items():
            user_labels[user_id] = label

    # Load split file
    with open(fold_json_path, "r") as fold_json_file:
        folds: Dict[str, List[str]] = json.load(fold_json_file)

    # Load prediction file
    with open(prediction_json_path, "r") as prediction_json_file:
        predictions: Dict[str, int] = json.load(prediction_json_file)

    # Evaluate prediction on the given fold
    uids_in_fold: List[str] = folds[fold_key]
    num_labelled_users = 0
    num_labelled_users_correct = 0

    for uid in tqdm(uids_in_fold):
        label = user_labels.get(uid)
        prediction = predictions.get(uid)
        if (label is not None) and (prediction is not None):
            assert isinstance(label, (int, float))
            assert isinstance(prediction, (int, float))
            num_labelled_users += 1

            if label == prediction:
                num_labelled_users_correct += 1

    if num_labelled_users >= 1:
        correct_user_ratio = num_labelled_users_correct / num_labelled_users
    else:
        correct_user_ratio = -1

    stats = {
        "test_correct_user_ratio": correct_user_ratio,
        "_test_num_users": num_labelled_users,
    }

    # Upload metrics on the fold to wandb
    wandb.log(stats)


if __name__ == "__main__":
    main()
