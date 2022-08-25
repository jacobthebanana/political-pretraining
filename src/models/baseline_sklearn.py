from typing import NamedTuple, List

from sklearn.linear_model import LogisticRegression
from datasets import load_from_disk
from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict
from transformers.hf_argparser import HfArgumentParser
from tqdm.auto import tqdm

from ..config import DataConfig, ModelConfig, PipelineConfig, NUM_TQDM_COLUMNS


class SklearnData(NamedTuple):
    X: List[List[float]]
    y: List[int]


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

    dataset_split: DatasetDict = dataset[split]  # type: ignore
    for dataset_entry in tqdm(
        dataset_split,
        desc=f"Loading {split} examples",
        ncols=NUM_TQDM_COLUMNS,
    ):
        X.append(dataset_entry["input_ids"])
        y.append(dataset_entry["label"])

    return SklearnData(X, y)


def main():
    parser = HfArgumentParser((ModelConfig, DataConfig, PipelineConfig))
    model_args, data_args, pipeline_args = parser.parse_args_into_dataclasses()
    model_args: ModelConfig
    data_args: DataConfig
    pipeline_args: PipelineConfig

    processed_dataset = load_from_disk(data_args.processed_dataset_path)  # type: ignore
    processed_dataset: DatasetDict

    train_data = get_sklearn_data(processed_dataset, "train")

    print("Fitting predictor")
    clf = LogisticRegression(random_state=pipeline_args.train_prng_key).fit(
        train_data.X, train_data.y
    )

    validation_data = get_sklearn_data(processed_dataset, "validation")

    print("Validation accuracy:", clf.score(validation_data.X, validation_data.y))


if __name__ == "__main__":
    main()
