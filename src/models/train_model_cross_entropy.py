"""
Functions for training with classification objective with 
a cross-entropy loss.
"""
from typing import Iterator, Tuple, Dict, List, Any
from collections import defaultdict, Counter
import json
from socket import gethostname
import datetime

import jax
import jax.numpy as jnp
from flax.training.common_utils import shard
from flax.jax_utils import replicate, unreplicate
import optax
import numpy as np

import datasets
from datasets import load_from_disk
from datasets.dataset_dict import DatasetDict
from transformers import FlaxRobertaForSequenceClassification
from transformers.hf_argparser import HfArgumentParser
from transformers.modeling_flax_outputs import FlaxSequenceClassifierOutput
from tqdm.auto import tqdm
import wandb

from ..config import (
    ModelConfig,
    DataConfig,
    PipelineConfig,
    BatchTokenKeysWithLabels,
    MetricKeys,
    LabelByUID,
    DatasetFeatures,
    UserID,
)
from ..data import load_labels
from .model_utils import (
    LabelledBatch,
    ShardedLabelledBatch,
    TokenizerOutput,
    TrainStepOutput,
    EvalStepOutput,
)

Dataset = datasets.arrow_dataset.Dataset
Array = jnp.ndarray

ModelParams = optax.Params
ReplicatedModelParams = optax.Params


def get_classification_dataloader(
    dataset: Dataset,
    per_device_batch_size: int,
    shuffle: bool = True,
    prng_key: jax.random.KeyArray = jax.random.PRNGKey(0),
) -> Iterator[Tuple[ShardedLabelledBatch, Dict[DatasetFeatures, Any]]]:
    """
    Return dataloader for classification (training and inference.)
    If shuffle = True, the dataloader loop would retrieve from the
    dataset with shuffled lists of indices.

    Args:
     dataset: labelled dataset to retrieve examples from.
     per_device_batch_size: specifies batch_size of each shard.
     shuffle: enables shuffling.
     prng_key: for use when shuffle is True.

    Return:
     ShardedTokenBatch: note that each array in this batch is of shape
     (num_devices, per_device_batch_size, ...)
     Dict[str, Any]: Raw examples from the dataset.
    """
    actual_batch_size = jax.device_count() * per_device_batch_size
    num_examples = len(dataset)
    num_batches = num_examples // actual_batch_size
    num_divisible_examples = num_batches * actual_batch_size
    num_trailing_examples = num_examples - num_divisible_examples

    indices = np.arange(num_examples)  # CPU-only
    if shuffle:
        indices = jax.random.permutation(prng_key, indices, independent=True)

    for j in range(num_batches):
        indices_in_batch = indices[j * actual_batch_size : (j + 1) * actual_batch_size]
        examples: Dict[DatasetFeatures, Array] = dataset[indices_in_batch]
        tokens: TokenizerOutput = {
            "input_ids": jnp.array(examples["input_ids"]),
            "attention_mask": jnp.array(examples["attention_mask"]),
        }
        batch = LabelledBatch(
            tokens=tokens,
            labels=jnp.array(examples["label"], dtype=int),
            loss_mask=jnp.ones(actual_batch_size),
        )

        sharded_batch: ShardedLabelledBatch = shard(batch)
        yield sharded_batch, examples

    if num_trailing_examples >= 1:
        trailing_indices = indices[num_divisible_examples:]
        trailing_examples: Dict[DatasetFeatures, Array] = dataset[trailing_indices]
        padding_length = actual_batch_size - num_trailing_examples

        # (num_trailing_examples, max_length)
        input_ids = jnp.array(trailing_examples["input_ids"])
        input_ids_padded = jnp.pad(
            input_ids,
            ((0, padding_length), (0, 0)),  # Pad only the batch dimension.
            mode="constant",
            constant_values=0,
        )

        # (num_trailing_examples, )
        attention_mask = jnp.array(trailing_examples["attention_mask"])
        attention_mask_padded = jnp.pad(
            attention_mask,
            ((0, padding_length), (0, 0)),  # Pad only the batch dimension.
            mode="constant",
            constant_values=0,
        )

        # (num_trailing_examples, )
        labels = jnp.array(trailing_examples["label"], dtype=int)
        labels_padded = jnp.pad(
            labels, ((0, padding_length),), mode="constant", constant_values=-1
        )

        loss_mask = jnp.ones(actual_batch_size).at[-padding_length:].set(0)

        tokens_padded: TokenizerOutput = {
            "input_ids": input_ids_padded,
            "attention_mask": attention_mask_padded,
        }

        trailing_batch = LabelledBatch(
            tokens=tokens_padded, labels=labels_padded, loss_mask=loss_mask
        )
        sharded_trailing_batch: ShardedLabelledBatch = shard(trailing_batch)
        yield sharded_trailing_batch, trailing_examples


def _loss_accuracy_fn_single_shard(
    batch: LabelledBatch,
    model: FlaxRobertaForSequenceClassification,
    model_params: Dict,
) -> Tuple[Array, Tuple[Array, Array]]:
    """
    Return the loss and accuracy of the given model
    on the given non-sharded data batch.

    Args:
     batch: non-sharded batch.
     model: abstract Flax model.
     model_params: parameters for the Flax model.

    Returns:
     cross-entropy loss, (classification accuracy, and predictions).
    """
    output = model(**(batch.tokens), params=model_params)  # type: ignore
    output: FlaxSequenceClassifierOutput
    logits: Array = output.logits

    loss = (
        optax.softmax_cross_entropy_with_integer_labels(logits, batch.labels)
        * batch.loss_mask
    )
    batch_loss = jnp.mean(loss)

    predictions = jnp.argmax(logits, axis=-1)
    correct_entries = jnp.where(
        batch.loss_mask == 1, predictions == batch.labels, False
    )
    num_correct = jnp.sum(correct_entries)
    num_predictions = jnp.sum(batch.loss_mask)

    safe_num_predictions = jnp.where(num_predictions == 0, 1, num_predictions)
    accuracy = num_correct / safe_num_predictions

    return batch_loss, (accuracy, predictions)


def _grad_accuracy_fn_single_shard(
    batch: LabelledBatch,
    model: FlaxRobertaForSequenceClassification,
    model_params: ModelParams,
) -> Tuple[Tuple[Array, Tuple[Array, Array]], optax.Updates]:
    ...


_grad_accuracy_fn_single_shard = jax.value_and_grad(
    _loss_accuracy_fn_single_shard, argnums=2, has_aux=True
)


def _eval_step_single_shard(
    batch: LabelledBatch,
    model: FlaxRobertaForSequenceClassification,
    model_params: Dict,
    metric_prefix: str,
) -> EvalStepOutput:
    """
    Get eval stats on the given batch.
    Applied pmean to the stats.

    Args:
     batch: non-sharded data.
     model: abstract Flax model.
     model_params: non-replicated weights for the Flax model.
     metric_prefix: string to add in front of each eval metric.

    Returns:
     Dict[str, Array]: non-replicated model stats.
    """
    loss, (accuracy, predictions) = _loss_accuracy_fn_single_shard(
        batch, model, model_params
    )
    loss, accuracy = jax.lax.pmean((loss, accuracy), axis_name="data")

    metrics: Dict[str, Array] = {
        metric_prefix + "_loss": loss,
        metric_prefix + "_accuracy": accuracy,
    }

    return EvalStepOutput(metrics=metrics, predictions=predictions)


def _eval_step(
    batch: ShardedLabelledBatch,
    model: FlaxRobertaForSequenceClassification,
    model_params: ReplicatedModelParams,
    metric_prefix: str,
) -> EvalStepOutput:
    """
    Evaluation step with pmean on the stats.
    Note that the stats arrays in the output are replicated.
    """
    ...


_eval_step = jax.pmap(
    _eval_step_single_shard, axis_name="data", static_broadcasted_argnums=(1, 3)
)


def _train_step_single_shard(
    batch: LabelledBatch,
    model: FlaxRobertaForSequenceClassification,
    model_params: ModelParams,
    optimizer: optax.GradientTransformation,
    optimizer_state: optax.OptState,
) -> TrainStepOutput:
    """
    Train the given model on the given batch.
    Applied pmean to both gradients and stats.

    Args:
     batch: non-sharded data.
     model: abstract Flax model.
     model_params: non-replicated weights for the Flax model.
     optimizer: Optax optimizer.
     optimizer_state: Optax optimizer state.

    Returns:
     TrainStepOutput: non-replicated model parameters and stats.
    """
    (loss, (accuracy, _)), param_grad = _grad_accuracy_fn_single_shard(
        batch, model, model_params
    )

    (loss, accuracy), param_grad = jax.lax.pmean(
        ((loss, accuracy), param_grad), axis_name="data"
    )

    updates, new_optimizer_state = optimizer.update(
        param_grad, optimizer_state, params=model_params
    )
    new_model_params = optax.apply_updates(model_params, updates)

    metrics: Dict[MetricKeys, Array] = {
        "training_loss": loss,
        "training_accuracy": accuracy,
    }

    return TrainStepOutput(
        metrics=metrics,
        model_params=new_model_params,
        optimizer_state=new_optimizer_state,
    )


def _train_step(
    batch: LabelledBatch,
    model: FlaxRobertaForSequenceClassification,
    model_params: ReplicatedModelParams,
    optimizer: optax.GradientTransformation,
    optimizer_state: optax.OptState,
) -> TrainStepOutput:
    ...


_train_step = jax.pmap(
    _train_step_single_shard, axis_name="data", static_broadcasted_argnums=(1, 3)
)


def get_num_batches(dataset: Dataset, per_device_batch_size: int) -> int:
    """
    Calculates number of batches after sharding.
    Accounts for the trailing batch.

    Args:
     dataset: HuggingFace dataset with proper len() support.
     per_device_batch_size: number of examples in each shard.

    Returns:
     int: number of regular batches, plus one if there is a trailing batch.
    """
    actual_batch_size = per_device_batch_size * jax.device_count()
    num_regular_batches = len(dataset) // actual_batch_size
    num_trailing_examples = len(dataset) - num_regular_batches * actual_batch_size

    if num_trailing_examples > 0:
        return num_regular_batches + 1

    return num_regular_batches


def get_test_stats(
    test_dataset: Dataset,
    test_batch_size: int,
    model: FlaxRobertaForSequenceClassification,
    replicated_model_params: ReplicatedModelParams,
    user_labels: LabelByUID,
    metric_prefix: str = "eval",
) -> Dict[str, float]:
    """
    Returns test stats.
    """
    test_dataloader = get_classification_dataloader(
        test_dataset, per_device_batch_size=test_batch_size, shuffle=False
    )
    num_test_batches = get_num_batches(test_dataset, test_batch_size)

    stats: Dict[str, List[float]] = defaultdict(list)
    predictions_by_user: Dict[UserID, List[int]] = defaultdict(list)
    for batch, examples in tqdm(
        test_dataloader,
        total=num_test_batches,
        ncols=80,
        desc=f"Evaluating {metric_prefix}",
    ):
        eval_output = _eval_step(batch, model, replicated_model_params, metric_prefix)
        batch_stats = eval_output.metrics

        user_ids: List[str] = examples["uid"]
        num_examples = len(user_ids)
        batch_predictions = eval_output.predictions.flatten()[:num_examples]

        for key, value in batch_stats.items():
            unreplicated_value: float = unreplicate(value)
            stats[key].append(unreplicated_value)

        for user_id, prediction_array in zip(user_ids, batch_predictions):
            user_id: UserID
            prediction: int = prediction_array.item()
            predictions_by_user[user_id].append(prediction)

    num_users = 0
    num_correct_users = 0
    for user_id, predictions in predictions_by_user.items():
        true_label = user_labels.get(user_id)
        if (predictions is not None) and (true_label is not None):
            num_users += 1
            majority_prediction, _ = Counter(predictions).most_common(1)[0]
            if majority_prediction == true_label:
                num_correct_users += 1

    stats_output: Dict[str, float] = {}

    if num_users > 0:
        stats_output[metric_prefix + "_user_accuracy"] = num_correct_users / num_users

    for key, values in stats.items():
        stats_output[key] = sum(values) / len(values)

    return stats_output


def get_num_classes(data_args: DataConfig) -> int:
    """
    Retrieve the number of label classes from
    `label_id_to_label_text.json`.

    Args:
     data_args: specifies `label_id_to_label_text_path`.

    Returns:
     int.
    """
    with open(data_args.label_id_to_label_text_path, "r") as label_lookup_file:
        label_lookup = json.load(label_lookup_file)

    return len(label_lookup.keys())


def get_model_name(data_args: DataConfig) -> str:
    """
    Return model name string with wandb run name included if available.
    """
    if wandb.run is not None:
        model_name = data_args.model_output_path + "-" + wandb.run.id
    else:
        model_name = data_args.model_output_path

    return model_name


def main():
    parser = HfArgumentParser((ModelConfig, DataConfig, PipelineConfig))
    model_args, data_args, pipeline_args = parser.parse_args_into_dataclasses()
    model_args: ModelConfig
    data_args: DataConfig
    pipeline_args: PipelineConfig

    split_dataset = load_from_disk(data_args.processed_dataset_path)  # type: ignore
    assert isinstance(split_dataset, DatasetDict)
    split_dataset: DatasetDict

    num_labels = get_num_classes(data_args)
    model = FlaxRobertaForSequenceClassification.from_pretrained(
        model_args.base_model_name, num_labels=num_labels, from_pt=True
    )  # type: ignore
    model: FlaxRobertaForSequenceClassification
    model_params = model.params

    train_dataset = split_dataset["train"]
    num_train_batches_per_epoch = get_num_batches(
        train_dataset, pipeline_args.train_per_device_batch_size
    )
    lr_schedule = optax.linear_schedule(
        init_value=model_args.learning_rate,
        end_value=0,
        transition_steps=num_train_batches_per_epoch * pipeline_args.num_epochs,
    )
    optimizer = optax.adamw(lr_schedule, weight_decay=model_args.weight_decay)
    # Initialize optimizer with original (non-replicated) model parameters
    optimizer_state = optimizer.init(model_params)

    replicated_model_params = replicate(model_params)
    replicated_optimizer_state = replicate(optimizer_state)

    eval_labels = load_labels(data_args.validation_filtered_label_path)

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
        }
    )

    for _ in tqdm(range(pipeline_args.num_epochs), ncols=80):
        train_dataloader = get_classification_dataloader(
            train_dataset,
            per_device_batch_size=pipeline_args.train_per_device_batch_size,
            shuffle=True,
            prng_key=jax.random.PRNGKey(pipeline_args.train_prng_key),
        )

        for batch_index, (batch, _) in enumerate(
            tqdm(train_dataloader, ncols=80, total=num_train_batches_per_epoch)
        ):
            if batch_index % pipeline_args.eval_every_num_batches == 0:
                eval_stats = {}
                for eval_split_key in ("validation", "test"):
                    eval_dataset = split_dataset[eval_split_key]
                    eval_stats = dict(
                        **eval_stats,
                        **get_test_stats(
                            eval_dataset,
                            pipeline_args.eval_per_device_batch_size,
                            model,
                            replicated_model_params,
                            eval_labels,
                            metric_prefix=eval_split_key,
                        ),
                    )

            else:
                eval_stats: Dict[MetricKeys, float] = {}

            train_step_output = _train_step(
                batch,
                model,
                replicated_model_params,
                optimizer,
                replicated_optimizer_state,
            )
            replicated_model_params = train_step_output.model_params
            replicated_optimizer_state = train_step_output.optimizer_state

            train_stats = unreplicate(train_step_output.metrics)
            stats = {**train_stats, **eval_stats}
            wandb.log(stats)

            if batch_index % pipeline_args.save_every_num_batches == 0:
                model_name = get_model_name(data_args)
                model_params = unreplicate(replicated_model_params)
                model.save_pretrained(model_name, params=jax.device_get(model_params))


if __name__ == "__main__":
    main()
