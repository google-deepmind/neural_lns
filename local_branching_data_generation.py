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
"""Library with functions required to generate LNS imitation data for one MIP."""

import collections as py_collections
import os
import pickle
from typing import Any, Dict, Optional, Sequence, Text

from absl import logging
import ml_collections
import numpy as np

from neural_lns import data_utils
from neural_lns import local_branching_expert
from neural_lns import mip_utils


# LP feature extraction needs to fully process the root node, so allow enough
# time for that.
MIN_SOLVE_TIME = 1800

# SCIP solving parameters
SCIP_SOLVING_PARAMS = ml_collections.ConfigDict({
    'seed': 42,
    'time_limit_seconds': 1800,
    'relative_gap': 0
})


def get_incumbent(
    instance_name: Text,
    dataset: Text,
    solution_index: int) -> Optional[mip_utils.MPSolutionResponse]:
  """Tries to retrieve a solution for the MIP from corresponding pickle file."""
  instance_path = os.path.join(dataset, instance_name)
  solutions = pickle.load(open(instance_path, 'rb'))

  if len(solutions) <= solution_index:
    raise ValueError(
        f'Fewer than {solution_index+1} solutions found for {instance_name}')
  else:
    solution = solutions[solution_index]

  return solution


def get_flipped_vars(mip: mip_utils.MPModel,
                     incumbent: mip_utils.MPSolutionResponse,
                     improved: mip_utils.MPSolutionResponse,
                     var_names: np.ndarray) -> np.ndarray:
  """Returns an array indicating which binary variables were flipped."""
  is_flipped = {}

  # Note that non-binary variables are always assigned a 0.
  for idx, variable in enumerate(mip.variable):
    if (mip_utils.is_var_binary(variable) and round(
        incumbent.variable_value[idx]) != round(improved.variable_value[idx])):
      is_flipped[variable.name] = 1.0
    else:
      is_flipped[variable.name] = 0.0

  # Make sure the array has the variables in the order in which they appear in
  # the features.
  is_flipped_reordered = np.zeros(len(var_names), dtype=np.bool)
  for idx, var_name in enumerate(var_names):
    if 'Constant' in var_name.decode():
      is_flipped_reordered[idx] = 0.0
    else:
      is_flipped_reordered[idx] = is_flipped[var_name.decode()]

  return is_flipped_reordered


def enhance_root_features(
    root_features: Dict[str, Any],
    incumbents: Sequence[Any],
    lp_sol: Optional[Any] = None
    ) -> Dict[str, Any]:
  """Adds incumbent var values and integer mask to the feature array.

  This accepts a list of up to NUM_PAST_INCUMBENTS past incumbents,
  sorted from most recent to least. Each incumbent will introduce two columns
  to the features: The first column represents the incumbent variable values,
  and the second one is a all-ones column indicating that the incumbent is
  present in the features.

  A final column is added to the end that masks out continuous variables.

  Args:
    root_features: Root features without incumbent information.
    incumbents: List of past incumbents, ordered by most recent first.
    lp_sol: solution to the LP relaxation of the LNS MIP solved by the expert.

  Returns:
    Updated features dict.
  """
  if len(incumbents) > data_utils.NUM_PAST_INCUMBENTS:
    raise ValueError(
        f'The number of past incumbents is not sufficient: {len(incumbents)}')

  # Fill columns corresponding to incumbents
  for idx, incumbent in enumerate(incumbents):
    column = data_utils.NUM_ROOT_VARIABLE_FEATURES + 2 * idx
    incumbent_values = np.array(
        [incumbent[var_name.decode()]
         for var_name in root_features['variable_names']],
        dtype=root_features['variable_features'].dtype)

    # Override features column corresponding to incumbent values.
    root_features['variable_features'][:, column] = incumbent_values

    # Override features column corresponding to incumbent presence indicator.
    root_features['variable_features'][:, column + 1] = np.ones(
        len(incumbent_values))

  if lp_sol is not None:
    lp_sol_values = np.array([
        lp_sol[var_name.decode()]
        for var_name in root_features['variable_names']
    ],
                             dtype=root_features['variable_features'].dtype)

    lp_sol_column_index = data_utils.NUM_ROOT_VARIABLE_FEATURES + 2 * len(
        incumbents)
    root_features['variable_features'][:, lp_sol_column_index] = lp_sol_values

  # The last column masks out the continuous variables.
  integer_values_mask = np.ones(len(root_features['variable_names']),
                                dtype=root_features['variable_features'].dtype)
  for idx, _ in enumerate(integer_values_mask):
    if idx not in root_features['all_integer_variable_indices']:
      integer_values_mask[idx] = 0.0

  root_features['variable_features'][:, -1] = integer_values_mask

  return root_features


def generate_data_for_instance(
    instance_name: Text,
    dataset: Text,
    neighbourhood_size: int = 20,
    percentage: bool = False,
    sequence_length: int = 10,
    add_incumbent_to_scip: bool = True,
    solution_index: int = 0,
    scip_params: ml_collections.ConfigDict = SCIP_SOLVING_PARAMS,
    num_var_features: int = data_utils.NUM_VARIABLE_FEATURES) -> int:
  """Generates data from which we learn to imitate the expert.

  This loads a MIP instance from a pickle file and generates the expert data.

  Args:
    instance_name: The name of the MIP instance.
    dataset: Dataset name that the instance belongs to.
    neighbourhood_size: Maximum Hamming dist to search.
    percentage: Whether neighbourhood_size should be treated as a percentage
                of total number of variables.
    sequence_length: How many consecutive improvements to do.
    add_incumbent_to_scip: Whether to feed SCIP the incumbent solution.
    solution_index: Which of the solutions to use as the first incumbent.
    scip_params: Dictionary of SCIP parameters to use.
    num_var_features: Number of features, NUM_VARIABLE_FEATURES or
                      NUM_VARIABLE_FEATURES_LP.

  Returns:
    status: 1 if expert data generation was successful, 0 otherwise.
  """
  mip = pickle.load(open(os.path.join(dataset, instance_name), 'rb'))
  if percentage:
    num_integer = 0
    for var in mip.variable:
      if var.is_integer:
        num_integer += 1
    neighbourhood_size = int(num_integer * neighbourhood_size / 100)

  try:
    incumbent = get_incumbent(instance_name, dataset, solution_index)
  except ValueError:
    logging.warning('No solution found for %s', instance_name)
    return 0

  root_features = data_utils.get_features(mip, scip_params)
  if root_features is None or root_features['variable_features'] is None:
    logging.warning('No root features found for %s', instance_name)
    return 0

  # Append dummy columns to the variable features, which is where we will put
  # the past incumbent solutions and the mask for assigned values at each step.
  num_extra_var_features = num_var_features - data_utils.NUM_ROOT_VARIABLE_FEATURES
  dummy_columns = np.zeros((root_features['variable_features'].shape[0],
                            num_extra_var_features),
                           dtype=root_features['variable_features'].dtype)

  if root_features is not None:
    root_features['variable_features'] = np.concatenate(
        [root_features['variable_features'], dummy_columns], axis=1)

    assert root_features['variable_features'].shape[
        1] == data_utils.NUM_VARIABLE_FEATURES

  status = 1
  past_incumbents = py_collections.deque([incumbent])

  for step in range(sequence_length):
    incumbent = past_incumbents[0]
    improved_sol = local_branching_expert.improve_solution(
        mip, incumbent, neighbourhood_size, scip_params,
        add_incumbent_to_scip=add_incumbent_to_scip)

    lp_sol = local_branching_expert.get_lns_lp_solution(
        mip, incumbent, neighbourhood_size, scip_params)

    if improved_sol is None:
      # In case of solver failure, print a warning and break.
      logging.warning('Solver failed for MIP %s at step %d ',
                      instance_name, step)
      status = 0
      break

    # Add features corresponding to the incumbent solution and integer mask.
    # NB This will overwrite the last column of the variable features.
    features = enhance_root_features(root_features, past_incumbents, lp_sol)

    # Figure out which variables were flipped between incumbent and improved.
    features['best_solution_labels'] = get_flipped_vars(
        mip, incumbent, improved_sol, features['variable_names'])

    # Add new incumbent to incumbent list, and prune to size if necessary
    past_incumbents.appendleft(improved_sol)
    if len(past_incumbents) > data_utils.NUM_PAST_INCUMBENTS:
      past_incumbents.pop()

  return status
