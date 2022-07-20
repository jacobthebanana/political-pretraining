from dataclasses import dataclass, field
from typing import Optional
import inspect


@dataclass
class ModelConfig:
    base_model_name: str = field(default="roberta-base")
    max_seq_length: int = field(default=128)

    @classmethod
    def from_dict(cls, args):
        cleaned_args = {}
        for k, v in args.items():
            if k in inspect.signature(cls).parameters:
                cleaned_args[k] = v

        return cls(**cleaned_args)


@dataclass
class DataConfig:
    csv_path: str = field(default="data/raw/tweets.csv")
    processed_dataset_path: str = field(default="data/raw/tweets.csv")
    num_procs: int = field(default=32)
