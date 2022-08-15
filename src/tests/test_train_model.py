import unittest
from typing import Dict, List
import json
from collections import Counter
from os import environ
from dataclasses import replace

import jax
import chex
import jax.numpy as jnp
from flax.jax_utils import replicate, unreplicate
from flax.training.common_utils import shard
import optax
import datasets
from datasets import load_from_disk
from transformers import (
    FlaxAutoModel,
    AutoConfig,
    RobertaConfig,
    FlaxRobertaModel,
    FlaxRobertaForSequenceClassification,
)
from tqdm.auto import tqdm

from ..models.train_model import (
    get_train_dataloader,
    get_anc_neg_distance,
    get_margins,
    get_top_triplet_pairs,
    get_triplet_loss,
    get_eligibility_mask,
    _get_neg_anc_indices,
    _train_step,
    _embed_mining_batch,
)
from ..models.train_model_cross_entropy import (
    get_classification_dataloader,
    get_num_classes,
    _train_step as _train_step_cross_entropy,
    get_test_stats as get_test_stats_cross_entropy,
    get_predictions_from_batch_output,
    get_most_popular_label,
    update_user_predictions,
    PerUserPredictions,
    get_fraction_correct_users,
)
from ..models.model_utils import (
    TokenBatch,
    get_token_batch,
    gather_shards,
    ModelParams,
    ReplicatedModelParams,
    BatchEmbeddings,
    ShardedTokenBatch,
    ShardedBatchEmbeddings,
    EvalStepOutput,
)
from ..data import load_labels
from ..config import (
    DataConfig,
    ModelConfig,
    PipelineConfig,
    LookupByUID,
    BatchTokenKeysWithLabels,
    DistanceFunction,
)

Dataset = datasets.arrow_dataset.Dataset


data_args = DataConfig(
    processed_dataset_path="data/testing/processed/tweets",
    processed_lookup_by_uid_json_path="data/testing/processed/tweets/lookup_by_uid.json",
)
num_devices = jax.device_count()

pipeline_args = PipelineConfig(
    train_per_device_batch_size=2, eval_per_device_batch_size=7
)

base_model_name = environ.get("unittest_base_model_name")
if base_model_name:
    model_args = ModelConfig(base_model_name=base_model_name)
else:
    model_args = ModelConfig()


class GetTrainDataLoaderFromProcessedDataset(unittest.TestCase):
    def setUp(self):
        self.preprocessed_dataset: Dataset = load_from_disk(data_args.processed_dataset_path)  # type: ignore

        with open(
            data_args.processed_lookup_by_uid_json_path, "r"
        ) as lookup_by_uid_json_file:
            self.lookup_by_uid: LookupByUID = json.load(lookup_by_uid_json_file)

        for feature in ["uid", "input_ids", "attention_mask"]:
            self.assertIn(feature, self.preprocessed_dataset.features)  # type: ignore

    def test_batch_shape(self):
        dataloader = get_train_dataloader(
            dataset=self.preprocessed_dataset,
            lookup_by_uid=self.lookup_by_uid,
            prng_key=jax.random.PRNGKey(0),
            pipeline_args=pipeline_args,
        )

        for batch in dataloader:
            self.assertSetEqual(
                set(batch.anchor_batch.info.keys()), set(["uid", "tid"])
            )
            self.assertSetEqual(
                set(batch.anchor_batch.tokens.keys()),
                set(["attention_mask", "input_ids", "label"]),
            )
            for key in ["attention_mask", "input_ids"]:  # type: ignore
                key: BatchTokenKeysWithLabels
                value = batch.anchor_batch.tokens[key]
                chex.assert_axis_dimension(value, 0, jax.device_count())
                chex.assert_axis_dimension(
                    value, 1, pipeline_args.train_per_device_batch_size
                )
                chex.assert_axis_dimension(value, 2, model_args.max_seq_length)

            for key in ["attention_mask", "input_ids"]:  # type: ignore
                key: BatchTokenKeysWithLabels
                value = batch.positive_batch.tokens[key]
                chex.assert_axis_dimension(value, 0, jax.device_count())
                chex.assert_axis_dimension(
                    value, 1, pipeline_args.train_per_device_batch_size
                )
                chex.assert_axis_dimension(value, 2, model_args.max_seq_length)

            for key in ["attention_mask", "input_ids"]:  # type: ignore
                key: BatchTokenKeysWithLabels
                value = batch.negative_candidate_batch.tokens[key]
                chex.assert_axis_dimension(value, 0, jax.device_count())
                chex.assert_axis_dimension(
                    value, 1, pipeline_args.eval_per_device_batch_size
                )
                chex.assert_axis_dimension(value, 2, model_args.max_seq_length)

    def test_sampling_repetition(self):
        dataloader = get_train_dataloader(
            dataset=self.preprocessed_dataset,
            lookup_by_uid=self.lookup_by_uid,
            prng_key=jax.random.PRNGKey(0),
            pipeline_args=pipeline_args,
        )

        anc_tid_counter = Counter()
        pos_tid_counter = Counter()
        neg_tid_counter = Counter()

        for batch in dataloader:
            for anc_tid in batch.anchor_batch.info["tid"]:
                anc_tid_counter[anc_tid] += 1

            for pos_tid in batch.positive_batch.info["tid"]:
                pos_tid_counter[pos_tid] += 1

            for neg_tid in batch.negative_candidate_batch.info["tid"]:
                neg_tid_counter[neg_tid] += 1

        # Repetitions are not allowed except for tweet ids for neg.
        self.assertEqual(max(anc_tid_counter.values()), 1)
        self.assertEqual(min(anc_tid_counter.values()), 1)
        self.assertLessEqual(
            max(neg_tid_counter.values()) - min(neg_tid_counter.values()), 1
        )


class EmbedBatchForTripletMining(unittest.TestCase):
    def setUp(self):
        self.model: FlaxRobertaModel = FlaxAutoModel.from_pretrained(
            model_args.base_model_name, from_pt=True
        )
        self.preprocessed_dataset: Dataset = load_from_disk(data_args.processed_dataset_path)  # type: ignore
        self.num_batches = (
            len(self.preprocessed_dataset)
            // (pipeline_args.train_per_device_batch_size * num_devices)
            + 1
        )

        with open(
            data_args.processed_lookup_by_uid_json_path, "r"
        ) as lookup_by_uid_json_file:
            self.lookup_by_uid: LookupByUID = json.load(lookup_by_uid_json_file)

        self.hf_model_config: RobertaConfig = AutoConfig.from_pretrained(
            model_args.base_model_name, from_pt=True
        )

    def test_embed_shape(self):
        dataloader = get_train_dataloader(
            self.preprocessed_dataset,
            self.lookup_by_uid,
            jax.random.PRNGKey(0),
            pipeline_args,
        )

        # Note that the dataloader shards data by default.
        for batch in tqdm(
            dataloader, total=self.num_batches, desc="Unit testing", ncols=80
        ):
            mining_token_batch = get_token_batch(batch)
            model_params: ModelParams = self.model.params  # type: ignore
            replicated_model_params: ReplicatedModelParams = replicate(model_params)
            mining_embeddings = _embed_mining_batch(
                mining_token_batch, self.model, model_args, replicated_model_params
            )

            for embeddings in (
                mining_embeddings.anchor_embeddings,
                mining_embeddings.positive_embeddings,
            ):
                # device, per_device_batch, embedding_dim.
                self.assertEqual(len(embeddings.shape), 3)
                self.assertEqual(
                    embeddings.shape[1], pipeline_args.train_per_device_batch_size
                )
                self.assertEqual(embeddings.shape[-1], self.hf_model_config.hidden_size)
                self.assertEqual(embeddings.shape[0], num_devices)


class MiningDistanceCalculations(unittest.TestCase):
    def setUp(self):
        self.train_batch_size = 5  # anc batch size
        self.eval_batch_size = 7  # neg batch size
        self.embedding_dim = 11
        self.placeholder_value = 5

        self.nonzero_distance = (
            self.embedding_dim * self.placeholder_value * self.placeholder_value
        )

        # The output should be zero everywhere except along row 1 or column 0,
        # but zero at (1, 0)
        anc_embeddings = jnp.zeros((self.train_batch_size, self.embedding_dim))
        pos_embeddings = (
            jnp.zeros((self.train_batch_size, self.embedding_dim))
            .at[1, :]
            .set(self.placeholder_value * 2)
        )
        neg_embeddings = (
            jnp.zeros((self.eval_batch_size, self.embedding_dim))
            .at[6, :]
            .set(self.placeholder_value)
        )

        self.batch_embeddings = BatchEmbeddings(
            anchor_embeddings=anc_embeddings,
            positive_embeddings=pos_embeddings,
            negative_embeddings=neg_embeddings,
        )

    def test_anc_neg_distance_output(self):
        distances = get_anc_neg_distance(self.batch_embeddings)
        chex.assert_shape(distances, (self.eval_batch_size, self.train_batch_size))

        for j in range(self.eval_batch_size):  # neg index
            for k in range(self.train_batch_size):  # anc index
                distance = distances[j, k]

                if j == 6:
                    self.assertEqual(distance, self.nonzero_distance)
                else:
                    self.assertEqual(distance, 0)

    def test_anc_pos_neg_margin_output(self):
        # a: anchor, n: negative, p: positive.
        # margin = d(a - n) - d(a - p)
        margins = get_margins(self.batch_embeddings)

        for j in range(self.eval_batch_size):  # neg index
            for k in range(self.train_batch_size):  # anc index
                margin = margins[j, k]

                # d(a - p) > d(a - n) whenever a_1 is chosen.
                # d(a - p) > d(a - n) at n_6 and a_1.
                if k == 1 and (j in range(6)):
                    self.assertEqual(
                        margin,
                        -self.nonzero_distance * 2 * 2,
                        (j, k),
                    )
                elif k == 1 and (j == 6):
                    self.assertEqual(margin, -self.nonzero_distance * 3, (j, k))
                elif j == 6:
                    self.assertEqual(margin, self.nonzero_distance, (j, k))
                else:
                    self.assertEqual(margin, 0, (j, k))


class GatherShards(unittest.TestCase):
    def setUp(self):
        embedding_dim = 11
        train_batch_size = pipeline_args.train_per_device_batch_size
        eval_batch_size = pipeline_args.eval_per_device_batch_size
        num_devices = jax.device_count()
        self.batch_embeddings: BatchEmbeddings = BatchEmbeddings(
            anchor_embeddings=jnp.arange(
                0, train_batch_size * num_devices * embedding_dim
            ).reshape((train_batch_size * num_devices, embedding_dim)),
            positive_embeddings=jnp.arange(
                0, train_batch_size * num_devices * embedding_dim
            ).reshape((train_batch_size * num_devices, embedding_dim)),
            negative_embeddings=jnp.arange(
                0, eval_batch_size * num_devices * embedding_dim
            ).reshape((eval_batch_size * num_devices, embedding_dim)),
        )

        self.sharded_batch_embeddings: ShardedBatchEmbeddings = shard(
            self.batch_embeddings
        )
        self.gathered_batch_embeddings: BatchEmbeddings = gather_shards(
            self.sharded_batch_embeddings
        )

    def test_gathered_batch_embedding_shape(self):
        chex.assert_trees_all_equal(
            self.gathered_batch_embeddings, self.batch_embeddings
        )


class RankMiningTriplets(unittest.TestCase):
    def setUp(self):
        self.model: FlaxRobertaModel = FlaxAutoModel.from_pretrained(
            model_args.base_model_name, from_pt=True
        )
        self.preprocessed_dataset: Dataset = load_from_disk(data_args.processed_dataset_path)  # type: ignore
        self.num_batches = (
            len(self.preprocessed_dataset)
            // (pipeline_args.train_per_device_batch_size * num_devices)
            + 1
        )

        with open(
            data_args.processed_lookup_by_uid_json_path, "r"
        ) as lookup_by_uid_json_file:
            self.lookup_by_uid: LookupByUID = json.load(lookup_by_uid_json_file)

        self.hf_model_config: RobertaConfig = AutoConfig.from_pretrained(
            model_args.base_model_name, from_pt=True
        )

    def test_eligibility_mask_shape_and_value(self):
        dataloader = get_train_dataloader(
            self.preprocessed_dataset,
            self.lookup_by_uid,
            jax.random.PRNGKey(0),
            pipeline_args,
        )

        model_args_with_eligibility_mask = replace(model_args, enable_masking=True)

        mining_batch = next(dataloader)
        batch_tokens: ShardedTokenBatch = get_token_batch(mining_batch)
        gathered_batch_tokens: TokenBatch = gather_shards(batch_tokens)

        eligibility_mask = get_eligibility_mask(
            gathered_batch_tokens, model_args_with_eligibility_mask, data_args
        )

        n_anc = pipeline_args.train_per_device_batch_size * num_devices
        n_neg = pipeline_args.eval_per_device_batch_size * num_devices

        chex.assert_axis_dimension(eligibility_mask, 1, n_anc)
        chex.assert_axis_dimension(eligibility_mask, 0, n_neg)

    def test_ranking_shape(self):
        dataloader = get_train_dataloader(
            self.preprocessed_dataset,
            self.lookup_by_uid,
            jax.random.PRNGKey(0),
            pipeline_args,
        )

        replicated_model_params = replicate(self.model.params)

        # Note that the dataloader shards data by default.
        for mining_batch in tqdm(
            dataloader, total=self.num_batches, desc="Unit testing", ncols=80
        ):
            filtered_token_batch = get_top_triplet_pairs(
                mining_batch, self.model, model_args, data_args, replicated_model_params
            )
            chex.assert_equal_shape(
                (
                    filtered_token_batch.anchor_tokens["input_ids"],
                    filtered_token_batch.positive_tokens["input_ids"],
                    filtered_token_batch.negative_tokens["input_ids"],
                )
            )
            chex.assert_axis_dimension(
                filtered_token_batch.anchor_tokens["input_ids"], 0, num_devices
            )
            chex.assert_axis_dimension(
                filtered_token_batch.anchor_tokens["input_ids"],
                1,
                pipeline_args.train_per_device_batch_size,
            )
            chex.assert_axis_dimension(
                filtered_token_batch.anchor_tokens["input_ids"],
                2,
                model_args.max_seq_length,
            )

    def test_indices_generation(self):
        num_anc = 5
        num_neg = 7
        neg_indices_repeated, anc_indices_repeated = _get_neg_anc_indices(
            num_neg, num_anc
        )

        for j in range(num_neg):
            for k in range(num_anc):
                self.assertEqual(neg_indices_repeated[j, k], j)
                self.assertEqual(anc_indices_repeated[j, k], k)


class GetTripletLoss(unittest.TestCase):
    """
    Validates the triplet loss implementation.
    """

    def setUp(self):
        self.embedding_dim = 7

    def test_triplet_loss_distance(self):
        anc_example = (
            jnp.ones((pipeline_args.train_per_device_batch_size, self.embedding_dim))
            .at[(0, 1), 0]
            .set(0)
        )
        pos_example = jnp.ones(
            (pipeline_args.train_per_device_batch_size, self.embedding_dim)
        )
        neg_example = (
            jnp.ones((pipeline_args.train_per_device_batch_size, self.embedding_dim))
            .at[(0), 0]
            .set(0)
        )

        example_batch = BatchEmbeddings(
            anchor_embeddings=anc_example,
            positive_embeddings=pos_example,
            negative_embeddings=neg_example,
        )

        loss_value = get_triplet_loss(example_batch, model_args)
        d_anc_pos = jnp.mean(
            jnp.sum(jnp.square(anc_example - pos_example), axis=1), axis=0
        )
        d_anc_neg = jnp.mean(
            jnp.sum(jnp.square(anc_example - neg_example), axis=1), axis=0
        )

        self.assertEqual(
            loss_value, d_anc_pos - d_anc_neg + model_args.triplet_threshold
        )


class StepTraining(unittest.TestCase):
    """
    Run the training step to ensure a reduction in loss value.
    """

    def setUp(self):
        self.model: FlaxRobertaModel = FlaxAutoModel.from_pretrained(
            model_args.base_model_name, from_pt=True
        )
        self.preprocessed_dataset: Dataset = load_from_disk(data_args.processed_dataset_path)  # type: ignore
        self.num_batches = (
            len(self.preprocessed_dataset)
            // (pipeline_args.train_per_device_batch_size * num_devices)
            + 1
        )

        with open(
            data_args.processed_lookup_by_uid_json_path, "r"
        ) as lookup_by_uid_json_file:
            self.lookup_by_uid: LookupByUID = json.load(lookup_by_uid_json_file)

        self.hf_model_config: RobertaConfig = AutoConfig.from_pretrained(
            model_args.base_model_name, from_pt=True
        )

    def test_loss_reduction(self):
        for distance_function in (
            DistanceFunction.L2,
            DistanceFunction.COSINE_DISTANCE,
        ):
            self.model_args = replace(model_args, distance_function=distance_function)
            replicated_model_params = replicate(self.model.params)
            optimizer = optax.adamw(0.001)
            optimizer_state = optimizer.init(self.model.params)
            replicated_optimizer_state = replicate(optimizer_state)

            train_losses = []
            for n_epoch in range(5):  # test epoch
                # Note that the dataloader shards data by default.
                dataloader = get_train_dataloader(
                    self.preprocessed_dataset,
                    self.lookup_by_uid,
                    jax.random.PRNGKey(0),
                    pipeline_args,
                )

                epoch_train_losses = []
                for mining_batch in tqdm(
                    dataloader, total=self.num_batches, desc="Unit testing", ncols=80
                ):
                    filtered_token_batch = get_top_triplet_pairs(
                        mining_batch,
                        self.model,
                        self.model_args,
                        data_args,
                        replicated_model_params,
                    )

                    train_step_output = _train_step(
                        filtered_token_batch,
                        self.model,
                        self.model_args,
                        replicated_model_params,
                        optimizer,
                        replicated_optimizer_state,
                    )

                    replicated_model_params = train_step_output.model_params
                    replicated_optimizer_state = train_step_output.optimizer_state

                    training_loss = train_step_output.metrics["training_loss"][0]
                    epoch_train_losses.append(training_loss)

                train_losses.append(sum(epoch_train_losses) / len(epoch_train_losses))

            print("Train losses (mean per epoch):", train_losses)


class GetClassificationDataloader(unittest.TestCase):
    def setUp(self):
        self.preprocessed_dataset: Dataset = load_from_disk(  # type: ignore
            data_args.processed_dataset_path
        )["train"]

    def test_batch_shape(self):
        batch_size = pipeline_args.eval_per_device_batch_size
        dataloader = get_classification_dataloader(
            self.preprocessed_dataset,
            per_device_batch_size=batch_size,
            shuffle=True,
            prng_key=jax.random.PRNGKey(0),
        )

        num_real_entries = 0

        for batch, _ in dataloader:
            chex.assert_equal_shape(
                (batch.tokens["attention_mask"], batch.tokens["input_ids"])
            )
            chex.assert_equal_shape((batch.labels, batch.loss_mask))

            chex.assert_axis_dimension(batch.labels, 1, batch_size)
            chex.assert_axis_dimension(batch.labels, 0, num_devices)

            chex.assert_axis_dimension(batch.tokens["attention_mask"], 0, num_devices)
            chex.assert_axis_dimension(
                batch.tokens["attention_mask"], 2, model_args.max_seq_length
            )
            chex.assert_axis_dimension(batch.tokens["attention_mask"], 1, batch_size)
            num_real_entries += jnp.sum(batch.loss_mask)

        self.assertEqual(num_real_entries, len(self.preprocessed_dataset))


class StepCrossEntropyLossTrainLoop(unittest.TestCase):
    def setUp(self):
        self.preprocessed_dataset = load_from_disk(data_args.processed_dataset_path)  # type: ignore
        self.dataloader = get_classification_dataloader(
            self.preprocessed_dataset["train"],  # type: ignore
            per_device_batch_size=pipeline_args.train_per_device_batch_size,
            shuffle=True,
            prng_key=jax.random.PRNGKey(0),
        )

    def test_step_cross_entropy_train_function(self):
        num_labels = get_num_classes(data_args)

        model = FlaxRobertaForSequenceClassification.from_pretrained(
            model_args.base_model_name, num_labels=num_labels
        )  # type: ignore
        model: FlaxRobertaForSequenceClassification
        model_params = model.params

        optimizer = optax.adamw(0.0001, weight_decay=model_args.weight_decay)
        # Initialize optimizer with original (non-replicated) model parameters
        optimizer_state = optimizer.init(model_params)

        replicated_model_params = replicate(model_params)
        replicated_optimizer_state = replicate(optimizer_state)

        batch, _ = next(self.dataloader)
        training_losses = []

        for _ in tqdm(range(12), desc="Unit testing", ncols=80):
            train_step_output = _train_step_cross_entropy(
                batch,
                model,
                replicated_model_params,
                optimizer,
                replicated_optimizer_state,
            )
            replicated_model_params = train_step_output.model_params
            replicated_optimizer_state = train_step_output.optimizer_state

            metrics = unreplicate(train_step_output.metrics)
            training_losses.append(metrics["training_loss"])

        self.assertLess(training_losses[-1], training_losses[0])

    def test_get_model_eval_metrics(self):
        num_labels = get_num_classes(data_args)
        user_labels = load_labels(data_args.filtered_label_path)

        model = FlaxRobertaForSequenceClassification.from_pretrained(
            model_args.base_model_name, num_labels=num_labels
        )  # type: ignore
        model: FlaxRobertaForSequenceClassification
        model_params = model.params

        replicated_model_params = replicate(model_params)
        dataset: Dataset = self.preprocessed_dataset["test"]  # type: ignore
        test_stats = get_test_stats_cross_entropy(
            dataset,
            pipeline_args.eval_per_device_batch_size,
            model,
            replicated_model_params,
            user_labels,
            metric_prefix="validation",
        )


class GetModelTestMetrics(unittest.TestCase):
    def setUp(self):
        actual_eval_batch_size = pipeline_args.eval_per_device_batch_size * num_devices
        self.actual_batch_size = min(11, actual_eval_batch_size)
        self.example_eval_output = EvalStepOutput(
            metrics={},
            predictions=jnp.arange(actual_eval_batch_size).reshape(
                (num_devices, pipeline_args.eval_per_device_batch_size)
            ),
            loss_mask=jnp.ones(actual_eval_batch_size)
            .at[self.actual_batch_size :]
            .set(0)
            .reshape((num_devices, pipeline_args.eval_per_device_batch_size)),
        )

    def test_get_predictions_from_batch_output(self):
        predictions = get_predictions_from_batch_output(self.example_eval_output)
        print(predictions)
        self.assertEqual(len(predictions), self.actual_batch_size)
        self.assertEqual(predictions[0], 0)
        self.assertEqual(predictions[-1], self.actual_batch_size - 1)

    def test_get_most_popular_label(self):
        most_popular_label = get_most_popular_label([9, 9, 6])
        self.assertEqual(most_popular_label, 9)

    def test_update_per_user_predictions(self):
        previous_per_user_predictions: PerUserPredictions = {
            "7": [0, 1],
            "11": [0, 0, 1],
            "12": [1, 0, 1],
        }

        predictions: List[int] = [-1, 0, 0, 1, -1, 1]
        user_ids: List[str] = ["11", "5", "11", "9", "5", "5"]
        updated_per_user_predictions = update_user_predictions(
            previous_per_user_predictions, predictions, user_ids
        )

        self.assertListEqual(updated_per_user_predictions["5"], [0, -1, 1])
        self.assertListEqual(updated_per_user_predictions["7"], [0, 1])
        self.assertListEqual(updated_per_user_predictions["9"], [1])
        self.assertListEqual(updated_per_user_predictions["11"], [-1, 0, 0, 0, 1])
        self.assertListEqual(updated_per_user_predictions["12"], [1, 0, 1])

    def test_get_fraction_correct_users(self):
        user_labels = {
            "7": 0,
            "11": 1,
        }

        user_float_labels = {
            "7": 0.0,
            "11": 1.0,
        }

        per_user_predictions: PerUserPredictions = {
            "7": [0, 1],
            "11": [0, 0, 1],
            "12": [1, 0, 1],
        }

        fraction_correct = get_fraction_correct_users(per_user_predictions, user_labels)
        self.assertEqual(fraction_correct, 1 / 2)

        fraction_correct_float_labels = get_fraction_correct_users(
            per_user_predictions, user_float_labels  # type: ignore
        )
        self.assertEqual(fraction_correct_float_labels, 1 / 2)
