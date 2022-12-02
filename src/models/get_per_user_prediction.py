"""
Run the given model on the given dataset split.
Save the output per-user labels in a json file.
"""
from typing import Dict
import argparse
import json

from flax.jax_utils import replicate
from transformers import FlaxRobertaForSequenceClassification
from datasets import load_from_disk
from datasets.dataset_dict import DatasetDict
from datasets.arrow_dataset import Dataset
from optax import Params

from .train_model_cross_entropy import get_test_stats_and_predictions
from ..config import UserID


def get_user_predictions(
    dataset: Dataset,
    test_batch_size: int,
    model: FlaxRobertaForSequenceClassification,
    replicated_model_params: Params,
) -> Dict[UserID, int]:
    user_predictions: Dict[UserID, int] = {}
    split_output = get_test_stats_and_predictions(
        dataset,
        test_batch_size,
        model,
        replicated_model_params,
        {},
        None,
        metric_prefix="",
    )

    for user_id, user_prediction in split_output.predictions.items():
        user_predictions[user_id] = user_prediction

    return user_predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("hf_model")
    parser.add_argument("dataset")
    parser.add_argument("output_json")
    parser.add_argument("--split", dest="split", default="test")
    parser.add_argument(
        "--test_batch_size", dest="test_batch_size", default=512, type=int
    )

    args = parser.parse_args()
    hf_model_path: str = args.hf_model
    dataset_path: str = args.dataset
    split: str = args.split
    test_batch_size: int = args.test_batch_size
    output_json_path: str = args.output_json

    # Load model
    model = FlaxRobertaForSequenceClassification.from_pretrained(  # type: ignore
        hf_model_path
    )
    model: FlaxRobertaForSequenceClassification
    model_params = model.params
    replicated_model_params = replicate(model_params)

    # Load dataset
    dataset_dict: DatasetDict = load_from_disk(dataset_path)  # type: ignore
    dataset: Dataset = dataset_dict[split]

    # Set up dataloader and evaluate model
    user_predictions = get_user_predictions(
        dataset, test_batch_size, model, replicated_model_params
    )

    # Save predictions
    with open(output_json_path, "w") as prediction_json_file:
        json.dump(user_predictions, prediction_json_file, indent=2)


if __name__ == "__main__":
    main()
