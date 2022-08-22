import jax
from typing import Callable, Dict
from optax import Params
from flax import linen as nn
from transformers.modeling_flax_outputs import FlaxSequenceClassifierOutput
from .model_utils import TokenizerOutput

Array = jax.numpy.ndarray


class LinearRegression(nn.Module):
    """
    Implementation of linear regression in jax and flax.
    """

    num_classes: int

    @nn.compact
    def __call__(self, x) -> FlaxSequenceClassifierOutput:
        # duck-type HuggingFace transformers, hence the attention_masks kwarg here
        logits = nn.Dense(self.num_classes)(x)

        return logits

    def get_duck_typed_model(
        self,
    ) -> Callable[[Array, Array, Params], FlaxSequenceClassifierOutput]:
        def model(
            input_ids: Array, attention_mask: Array, params: Params
        ) -> FlaxSequenceClassifierOutput:
            logits = self.apply(params, input_ids)  # type: ignore

            return FlaxSequenceClassifierOutput(logits=logits)

        return model
