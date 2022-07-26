from typing import Any, Container, Dict, Tuple
from dataclasses import dataclass, field


BatchTokenKeys = str  # "input_ids", "attention_mask"
BatchInfoKeys = str  # "uid"
UserID = str

LookupByUID = Dict[UserID, Tuple[int, ...]]


@dataclass
class ModelConfig:
    base_model_name: str = field(default="roberta-base")
    max_seq_length: int = field(default=128)


@dataclass
class DataConfig:
    source_format: str = field(default="csv")
    source_path: str = field(default="data/raw/tweets.csv")
    processed_dataset_path: str = field(default="data/processed/tweets")
    processed_lookup_by_uid_json_path: str = field(
        default="data/processed/tweets/lookup_by_uid.json"
    )
    output_embeddings_json_path: str = field(default="data/artifacts/embeddings.json")
    num_procs: int = field(default=32)
    # Keep only the last 1/shard_denominator of data.
    shard_denominator: int = field(default=1)


@dataclass
class PipelineConfig:
    # Batch size for forward passes, including ranking triplets.
    eval_per_device_batch_size: int = field(default=128)
    # Batch size for calculating gradients
    train_per_device_batch_size: int = field(default=16)
