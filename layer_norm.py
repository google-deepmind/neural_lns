# Copyright 2021 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Model layer normalisation and dropout utilities."""

import sonnet as snt
import tensorflow.compat.v2 as tf


class ResidualDropoutWrapper(snt.Module):
  """Wrapper that applies residual connections, dropout and layer norm."""

  def __init__(self, layer, dropout_rate, apply_layer_norm=True, name=None):
    """Creates the Wrapper Class.

    Args:
      layer: module to wrap.
      dropout_rate: dropout rate. A rate of 0. will turn off dropout.
      apply_layer_norm: (default True) whether to apply layer norm after
        residual.
      name: name of the module.
    """

    super(ResidualDropoutWrapper, self).__init__(name=name)
    self._layer = layer
    self._dropout_rate = dropout_rate
    self._apply_layer_norm = apply_layer_norm

    if self._apply_layer_norm:
      self._layer_norm = snt.LayerNorm(
          axis=-1, create_scale=True, create_offset=True)

  def __call__(self, inputs, *args, **kwargs):
    """Returns the result of the residual dropout computation.

    Args:
      inputs: inputs to the main module.
      *args: Additional arguments to inner layer.
      **kwargs: Additional named arguments to inner layer.
    """

    # Apply main module.
    outputs = self._layer(inputs, *args, **kwargs)

    # Dropout before residual.
    if kwargs.get('is_training', False):
      outputs = tf.nn.dropout(outputs, rate=self._dropout_rate)

    if 'query_inputs' in kwargs:
      outputs += kwargs['query_inputs']
    else:
      outputs += inputs

    if self._apply_layer_norm:
      outputs = self._layer_norm(outputs)

    return outputs
