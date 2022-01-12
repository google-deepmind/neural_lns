# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
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
"""Sampling strategies for Neural LNS."""

import abc
from typing import Any, List, NamedTuple, Optional
from graph_nets import graphs
import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp


class Assignment(NamedTuple):
  names: List[str]
  lower_bounds: List[int]
  upper_bounds: List[int]


def sample_probas(model: Any, gt: graphs.GraphsTuple,
                  node_indices: np.ndarray) -> np.ndarray:
  """Obtain variable probabilities from a conditionally independent model.

  Args:
    model: SavedModel to sample from.
    gt: GraphsTuple of input MIP.
    node_indices: Indices of variables to predict.

  Returns:
    np.ndarray of probabilities of the sample.
  """
  _, probas = model.greedy_sample(gt, node_indices)
  return probas.numpy()


class BaseSampler(metaclass=abc.ABCMeta):
  """Abstract class for samplers."""

  def __init__(self, model_path: Optional[str] = None):
    """Initialization.

    Args:
      model_path: Model path to load for prediction/sampling.
    """
    self.model_path = model_path
    self.model = None

  @abc.abstractmethod
  def sample(
      self,
      graphs_tuple: graphs.GraphsTuple,
      var_names: np.ndarray,
      lbs: np.ndarray,
      ubs: np.ndarray,
      node_indices: np.ndarray,
      **kwargs) -> Assignment:
    """Returns a sample assignment for given inputs.

    Args:
      graphs_tuple: Input MIP with features.
      var_names: Names of MIP variables.
      lbs: Lower bounds of variables.
      ubs: Upper bounds of variables.
      node_indices: Node indices of the binary nodes.
      **kwargs: Sampler specific arguments.

    Returns:
      A single Assignment.
    """
    return Assignment([], [], [])


class RandomSampler(BaseSampler):
  """Sampler that returns assignments after randomly unassigning variables.

  This sampler returns an assignment obtained from leaving a random set of k
  variables unassigned, and fixing the rest.
  The value of k is the minimum of the number of provided node indices and
  a num_unassigned_vars parameter. In other words, the variables that were
  selected for flipping are the ones left unassigned.
  """

  def sample(self,
             graphs_tuple: graphs.GraphsTuple,
             var_names: np.ndarray,
             var_values: np.ndarray,
             node_indices: np.ndarray,
             num_unassigned_vars: int) -> Assignment:
    """Sampling.

    Args:
      graphs_tuple: GraphsTuple to produce samples for.
      var_names: Variable names array.
      var_values: Variable values.
      node_indices: Node indices array for which to produce predictions.
      num_unassigned_vars: The number of variables to keep free in the submip.

    Returns:
      Sampler's Assignment.
    """
    flattened_indices = np.squeeze(node_indices)
    num_top_vars = np.min([num_unassigned_vars, np.size(flattened_indices)])

    # The following gives us the indices of the variables to unassign.
    # We randomly select binary indices assuming a uniform distribution.
    top_indices = set(np.random.choice(
        flattened_indices, size=(num_top_vars,), replace=False))

    accept_mask = []
    for idx in range(len(var_names)):
      # Fix all binary vars except the ones selected to be unassigned above.
      # Leave the non-binary vars unfixed, too.
      fix_var = False if idx in top_indices or idx not in node_indices else True
      accept_mask.append(fix_var)

    var_names_to_assign = []
    var_values_to_assign = []

    for accept, val, name in zip(accept_mask, var_values, var_names):
      if accept:
        var_name = name.decode() if isinstance(name, bytes) else name
        var_names_to_assign.append(var_name)
        var_values_to_assign.append(val)

    return Assignment(
        var_names_to_assign, var_values_to_assign, var_values_to_assign)


class RepeatedCompetitionSampler(BaseSampler):
  """Sampler that repeatedly samples from the topK not yet unassigned variables.
  """

  def __init__(self, model_path: str):
    super().__init__(model_path)
    self.model = tf.saved_model.load(self.model_path)

  def sample(self,
             graphs_tuple: graphs.GraphsTuple,
             var_names: np.ndarray,
             var_values: np.ndarray,
             node_indices: np.ndarray,
             num_unassigned_vars: int,
             probability_power: Optional[float] = None,
             eps: float = 0.) -> Assignment:
    """Sampling.

    Args:
      graphs_tuple: GraphsTuple to produce samples for.
      var_names: Variable names array.
      var_values: Variable values.
      node_indices: Node indices array for which to produce predictions.
      num_unassigned_vars: The number of variables to keep free in the submip.
      probability_power: powers the probabilities to smoothen the distribution,
        works similarly to temperature.
      eps: a number to add to all probabilities.

    Returns:
      Sampler's assignment.
    """
    proba = sample_probas(self.model, graphs_tuple, node_indices)
    proba = np.squeeze(proba) + eps

    num_top_vars = np.min([num_unassigned_vars, len(proba)])

    unfixed_variables = set()
    for _ in range(num_top_vars):
      # NB `proba` has the probabilities for the variables corresponding to
      # `node_indices` only. So the result of `argsort` gives us the indices of
      # the right indices in `node_indices`.

      round_proba = proba.copy()
      if probability_power is not None:
        np.power(round_proba, probability_power, out=round_proba)
      np.divide(round_proba, round_proba.sum(), out=round_proba)

      var_idx = tfp.distributions.Categorical(probs=round_proba).sample()

      unfixed_variables.add(var_idx.numpy())
      proba[var_idx] = 0.

    accept_mask = []
    for idx in range(len(var_names)):
      # Fix all binary vars except the ones with highest flip prediction.
      # Leave the non-binary vars unfixed, too.
      fix_var = idx not in unfixed_variables and idx in node_indices
      accept_mask.append(fix_var)

    var_names_to_assign = []
    var_values_to_assign = []

    for accept, val, name in zip(accept_mask, var_values, var_names):
      if accept:
        var_name = name.decode() if isinstance(name, bytes) else name
        var_names_to_assign.append(var_name)
        var_values_to_assign.append(val)

    return Assignment(
        var_names_to_assign, var_values_to_assign, var_values_to_assign)


SAMPLER_DICT = {
    'random': RandomSampler,
    'competition': RepeatedCompetitionSampler,
}
