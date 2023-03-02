# Copyright 2022 The T5X Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for attention classes."""

import dataclasses
from typing import Optional
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from flax import linen as nn
from flax.core import freeze
from flax.linen import partitioning as nn_partitioning
import jax
from jax import random
from jax.nn import initializers
import jax.numpy as jnp
import numpy as np

# from models.scalable_t5.alibi_position_biases import ALiBiPositionBiases
from alibi_position_biases import ALiBiPositionBiases

# Parse absl flags test_srcdir and test_tmpdir.
jax.config.parse_flags_with_absl()

Array = jnp.ndarray
AxisMetadata = nn_partitioning.AxisMetadata  # pylint: disable=invalid-name


class ALiBiPositionBiasesTest(absltest.TestCase):

  def setUp(self):
    self.num_heads = 3
    self.query_len = 5
    self.key_len = 7
    self.relative_attention = ALiBiPositionBiases(
        num_heads=self.num_heads,
        dtype=jnp.float32,
    )
    super(ALiBiPositionBiasesTest, self).setUp()

  def test_relative_attention_bidirectional_params(self):
    """Tests that bidirectional relative position biases have expected params."""
    params = self.relative_attention.init(
        random.PRNGKey(0), self.query_len, self.key_len)
    param_shapes = jax.tree_map(lambda x: x.shape, params)
    self.assertEqual(
        param_shapes, {
            'params': {
                'rel_embedding': (3, 12),
            },
            'params_axes': {
                'rel_embedding_axes':
                    AxisMetadata(names=('heads', 'relpos_buckets')),
            }
        })

  def test_alibi_position_bias_values(self):
    """Tests that bidirectional relative position biases match expected values.

    See top docstring note on matching T5X behavior for these regression tests.
    """
    outputs, unused_params = self.relative_attention.init_with_output(
        random.PRNGKey(0), self.query_len, self.key_len)

    self.assertEqual(outputs.shape,
                     (1, self.num_heads, self.query_len, self.key_len))

# [[ 0.          0.          0.          0.          0.
#      0.          0.        ]
#    [-0.0625      0.          0.          0.          0.
#      0.          0.        ]
#    [-0.125      -0.0625      0.          0.          0.
#      0.          0.        ]
#    [-0.1875     -0.125      -0.0625      0.          0.
#      0.          0.        ]
#    [-0.25       -0.1875     -0.125      -0.0625      0.
#      0.          0.        ]]

    # expected = jnp.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
    # np.testing.assert_array_equal(outputs, expected)
    # self.assertEqual(outputs[0, 0, 4, 0], -0.25)
    # self.assertEqual(outputs[0, 0, 4, 1], -0.1875)
    # self.assertEqual(outputs[0, 0, 4, 2], -0.125)
    # self.assertEqual(outputs[0, 0, 4, 3], -0.0625)
    # self.assertEqual(outputs[0, 0, 4, 4], 0.)


#   def test_relative_attention_unidirectional_params(self):
#     """Tests that unidirectional relative position biases have expected params."""
#     params = self.relative_attention.init(
#         random.PRNGKey(0), self.query_len, self.key_len, bidirectional=False)
#     param_shapes = jax.tree_map(lambda x: x.shape, params)
#     self.assertEqual(
#         param_shapes, {
#             'params': {
#                 'rel_embedding': (3, 12),
#             },
#             'params_axes': {
#                 'rel_embedding_axes':
#                     AxisMetadata(names=('heads', 'relpos_buckets')),
#             }
#         })

#   def test_regression_relative_attention_unidirectional_values(self):
#     """Tests that unidirectional relative position biases match expected values.

#     See top docstring note on matching T5X behavior for these regression tests.
#     """
#     outputs, unused_params = self.relative_attention.init_with_output(
#         random.PRNGKey(0), self.query_len, self.key_len, bidirectional=False)
#     self.assertEqual(outputs.shape,
#                      (1, self.num_heads, self.query_len, self.key_len))
#     self.assertAlmostEqual(outputs[0, 0, 0, 0], 0.55764728, places=5)
#     self.assertAlmostEqual(outputs[0, 1, 2, 1], -0.10935841, places=5)
#     self.assertAlmostEqual(outputs[0, 1, 4, 6], -0.13101986, places=5)
#     self.assertAlmostEqual(outputs[0, 2, 4, 6], 0.39296466, places=5)


if __name__ == '__main__':
  absltest.main()
