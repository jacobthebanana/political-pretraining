from typing import NamedTuple, List, Dict
import json
from socket import gethostname
import datetime
import os
import pickle

from sklearn.linear_model import LogisticRegression
from datasets import load_from_disk
from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict
from transformers.hf_argparser import HfArgumentParser
from tqdm.auto import tqdm
import wandb

from ..config import DataConfig, ModelConfig, PipelineConfig, NUM_TQDM_COLUMNS


def get_model_name(data_args: DataConfig) -> str:
    """
    Return model name string with wandb run name included if available.
    """
    if wandb.run is not None:
        model_name = data_args.model_output_path + "-" + wandb.run.id
    else:
        model_name = data_args.model_output_path

    return model_name


class SklearnData(NamedTuple):
    X: List[List[float]]
    y: List[int]
    user_ids: List[str]


def get_sklearn_data(dataset: DatasetDict, split: str) -> SklearnData:
    """
    Load arrays in the dataset into sklearn-compatible
    nested lists.

    Args:
     dataset: HuggingFace DatasetDict
     split:
    """
    X = []
    y = []
    user_ids: List[str] = []

    dataset_split: DatasetDict = dataset[split]  # type: ignore
    for dataset_entry in tqdm(
        dataset_split,
        desc=f"Loading {split} examples",
        ncols=NUM_TQDM_COLUMNS,
    ):
        X.append(dataset_entry["input_ids"])
        y.append(dataset_entry["label"])

        user_id = dataset_entry["uid"]
        user_ids.append(user_id)

    return SklearnData(X, y, user_ids)


def get_accuracy(clf: LogisticRegression, data: SklearnData) -> float:
    """
    Return fraction of entries that are labelled correctly.

    Args:
     clf: sklearn classifier with predict method.
     data: SklearnData

    Returns:
     float: fraction of entries labelled correctly. Returns (-1) if
     data is empty.
    """
    y_pred = clf.predict(data.X)
    num_entries = 0
    num_entries_correct = 0

    for prediction, label in zip(y_pred, data.y):
        num_entries += 1
        if prediction == label:
            num_entries_correct += 1

    return num_entries_correct / num_entries


def main():
    parser = HfArgumentParser((ModelConfig, DataConfig, PipelineConfig))
    model_args, data_args, pipeline_args = parser.parse_args_into_dataclasses()
    model_args: ModelConfig
    data_args: DataConfig
    pipeline_args: PipelineConfig

    wandb.init(
        entity=pipeline_args.wandb_entity,
        project=pipeline_args.wandb_project,
        name=datetime.datetime.now().isoformat() + "-" + gethostname(),
    )
    wandb.config.update(
        {
            "model_args": model_args.__dict__,
            "data_args": data_args.__dict__,
            "pipeline_args": pipeline_args.__dict__,
            "baseline": "Preotiuc-Pietro",
            "fold_key": os.environ.get("fold_key"),
        }
    )

    processed_dataset = load_from_disk(data_args.processed_dataset_path)  # type: ignore
    processed_dataset: DatasetDict

    train_data = get_sklearn_data(processed_dataset, "train")

    print("Fitting predictor")
    clf = LogisticRegression(random_state=pipeline_args.train_prng_key).fit(
        train_data.X, train_data.y
    )

    stats: Dict[str, float] = {}
    per_user_predictions: Dict[str, int] = {}

    for split in ("validation", "test"):
        data = get_sklearn_data(processed_dataset, split)
        accuracy_score = clf.score(data.X, data.y)
        predictions = clf.predict(data.X)
        print(f"{split} accuracy", accuracy_score)

        stats[split + "_correct_users_ratio"] = accuracy_score
        stats[split + "_num_users"] = len(data.y)

        for user_id, prediction in zip(data.user_ids, predictions):
            per_user_predictions[user_id] = prediction

    model_output_folder = get_model_name(data_args)
    os.makedirs(model_output_folder, exist_ok=True)

    model_pkl_path = os.path.join(model_output_folder, "sklearn_clf_model.pkl")
    with open(model_pkl_path, "wb") as model_pkl_file:
        pickle.dump(clf, model_pkl_file)
        print("Model saved to", model_pkl_path)

    predictions_json_path = os.path.join(model_output_folder, "predictions.json")
    with open(predictions_json_path, "w") as model_json_file:
        json.dump(per_user_predictions, model_json_file, indent=2)
        print("Predictions saved to", predictions_json_path)

    wandb.log(stats)


if __name__ == "__main__":
    main()
