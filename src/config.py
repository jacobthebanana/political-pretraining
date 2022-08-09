from typing import Any, Container, Dict, Tuple, Optional
from typing_extensions import Literal
from enum import Enum
import multiprocessing

from dataclasses import dataclass, field


BatchTokenKeys = Literal["input_ids", "attention_mask"]
BatchTokenKeysWithLabels = Literal["input_ids", "attention_mask", "label"]
BatchInfoKeys = Literal["uid", "tid"]
MetricKeys = Literal[
    "training_loss",
    "training_loss_net",
    "training_accuracy",
    "eval_accuracy",
    "eval_loss",
]
UserID = str

LookupByUID = Dict[UserID, Tuple[int, ...]]
LabelByUID = Dict[UserID, int]


class PoolingStrategy(Enum):
    CLS_EMBEDDING_WITH_DENSE_LAYER = "cls_embedding_with_dense_layer"
    CLS_EMBEDDING_ONLY = "cls_embedding_only"
    WORD_EMBEDDING_MEAN = "word_embedding_mean"


class ConcatenationDelimiter(Enum):
    SPACE = "space"
    NEWLINE = "newline"


class DistanceFunction(Enum):
    L2 = "l2"
    COSINE_DISTANCE = "cosine_distance"


CONCATENATION_DELIMITER_MAP = {
    "space": " ",
    "newline": "\n",
}


@dataclass(frozen=True)
class ModelConfig:
    base_model_name: str = field(default="roberta-base")
    max_seq_length: int = field(default=128)
    pooling_strategy: PoolingStrategy = field(
        default=PoolingStrategy.CLS_EMBEDDING_WITH_DENSE_LAYER
    )
    triplet_threshold: Optional[float] = field(default=1e2)
    learning_rate: float = field(default=0.001)
    weight_decay: float = field(default=0.00001)
    distance_function: DistanceFunction = field(default=DistanceFunction.L2)
    # Whether mask out triplets where the author of anc and pos are from
    # the same category as the author of neg.
    enable_masking: bool = field(default=False)


@dataclass
class DataConfig:
    source_format: str = field(default="csv")
    source_path: str = field(default="data/raw/tweets.csv")
    raw_label_path: str = field(default="data/raw/user_labels.csv")
    filtered_label_path: str = field(default="data/interim/filtered_user_labels.csv")
    label_id_to_label_text_path: str = field(
        default="data/interim/label_id_to_label_text.json"
    )
    # Whether to exclude text from unlabelled users in the preprocessed dataset.
    require_labels: bool = field(default=False)
    processed_dataset_path: str = field(default="data/processed/tweets")
    processed_lookup_by_uid_json_path: str = field(
        default="data/processed/tweets/lookup_by_uid.json"
    )
    model_output_path: str = field(default="data/artifacts/saved_model")
    output_embeddings_json_path: str = field(default="data/artifacts/embeddings.json")
    num_procs: int = field(default=multiprocessing.cpu_count())
    # Keep only the last 1/shard_denominator of data.
    shard_denominator: int = field(default=1)
    # Whether to generate the lookup_by_uid json.
    enable_indexing: bool = field(default=True)
    # Whether to tokenize the dataset instead of loading the saved one.
    rerun_tokenization: bool = field(default=True)
    per_user_concatenation: bool = field(default=False)
    concatenation_delimiter: ConcatenationDelimiter = field(
        default=ConcatenationDelimiter.NEWLINE
    )
    test_ratio: float = field(default=0.2)


@dataclass
class PipelineConfig:
    # Batch size for forward passes, including ranking triplets.
    eval_per_device_batch_size: int = field(default=128)
    # Batch size for calculating gradients
    train_per_device_batch_size: int = field(default=16)
    train_prng_key: int = field(default=0)
    num_epochs: int = field(default=1)
    save_every_num_batches: int = field(default=1000)
    eval_every_num_batches: int = field(default=10)
    wandb_project: str = field(default="")
    wandb_entity: str = field(default="")