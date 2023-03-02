import math
import numpy as np
import jax.numpy as jnp

from typing import Any

from jax import lax
from flax import linen as nn


class ALiBiPositionBiases(nn.Module):
  """Adds ALiBi positional embeddings to the attention logits.
  Attributes:
    Attribute
  """
  num_heads: int
  decoder: bool
  dtype: Any

  @nn.compact
  def __call__(self, qlen, klen):
    """Produce ALiBi
    
    Args:
      qlen:
      klen:
    Returns:
      output:
    """

    def _get_slopes(n):
        def _get_slopes_power_of_2(n):
            start = (2 ** (-2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio ** i for i in range(n)]

        if math.log2(n).is_integer():
            return _get_slopes_power_of_2(n)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n))
            return _get_slopes_power_of_2(closest_power_of_2) + \
              _get_slopes(2 * closest_power_of_2)[0::2][:n - closest_power_of_2]


    slopes = jnp.asarray(_get_slopes(self.num_heads), dtype=self.dtype)

    if self.decoder:
        constant_bias = jnp.expand_dims(slopes, axis=(1,2)) * jnp.expand_dims(jnp.arange(qlen), axis=(0,1)).repeat(self.num_heads, 0)
        mask = jnp.expand_dims(jnp.triu(jnp.ones((qlen,qlen))/0 * -1, 1), 0)
        values = mask + constant_bias
    else:
        context_position = np.arange(qlen, dtype=self.dtype)[:, None]
        memory_position = np.arange(klen, dtype=self.dtype)[None, :]
        constant_bias = jnp.abs(memory_position - context_position)  # shape (qlen, klen)
        # Constant Bias
        #  0, ...
        # -1, 0, ...
        # -2, -1, 0, ...
        # -3, -2, -1, 0, ...
        # -4, -3, -2, -1, 0, ...
        # ...

        # head-specific scalar
        slopes = slopes*-1
        # --> shape (num_heads, qlen, klen)
        values = lax.dot_general(
            slopes,
            constant_bias,
            (
            ((),()),
            ((),())
            )
        )  # no batched dims

    # Add a singleton batch dimension.
    # --> shape (1, num_heads, qlen, klen)
    return values[jnp.newaxis, ...]