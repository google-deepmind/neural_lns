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
"""Expert for Neural Neighbourhood Selection based on local branching."""

import copy
from typing import Any, Optional

from absl import logging
import ml_collections
import numpy as np

from neural_lns import mip_utils
from neural_lns import solving_utils


def get_binary_local_branching_mip(mip: mip_utils.MPModel,
                                   incumbent: mip_utils.MPSolutionResponse,
                                   neighbourhood_size: int) -> Any:
  """Add binary local branching to the MIP model and returns the new MIP.

  Args:
    mip: Input MIP
    incumbent: The incumbent solution.
    neighbourhood_size: Maximum Hamming distance (across integer vars) from
      incumbent.

  Returns:
    MIP with local branching constraints.
  """
  lns_mip = copy.deepcopy(mip)

  names = []
  values = []
  for variable, value in zip(mip.variable, incumbent.variable_value):
    if mip_utils.is_var_binary(variable):
      names.append(variable.name)
      # Rounding to make sure the conversion to int is correct.
      values.append(np.round(value))
  weights = np.ones(len(names))
  mip_utils.add_binary_invalid_cut(lns_mip, names, values, weights,
                                   neighbourhood_size)
  return lns_mip


def get_general_local_branching_mip(
    mip: mip_utils.MPModel, incumbent: mip_utils.MPSolutionResponse,
    neighbourhood_size: int) -> mip_utils.MPModel:
  """Add general local branching to the MIP model and returns the new MIP.

  Details see slide 23 of
  http://www.or.deis.unibo.it/research_pages/ORinstances/mic2003-lb.pdf

  Args:
    mip: Input MIP.
    incumbent: The incumbent solution.
    neighbourhood_size: Maximum Hamming distance (across integer vars) from
      incumbent.

  Returns:
    MIP with local branching constraints.
  """

  lns_mip = copy.deepcopy(mip)
  orig_names = set([v.name for v in mip.variable])

  name_to_aux_plus = {}
  name_to_aux_minus = {}

  # First, we add all new auxiliary variables to the new MIP.
  # General integers add aux. variables v_plus and v_minus.
  for v in mip.variable:
    if v.is_integer and not mip_utils.is_var_binary(v):
      # Find names for auxiliary vars that were not used in original names.
      aux_plus_name = v.name + '_plus'
      while aux_plus_name in orig_names:
        aux_plus_name += '_'
      aux_minus_name = v.name + '_minus'
      while aux_minus_name in orig_names:
        aux_minus_name += '_'

      lns_mip.variable.append(
          mip_utils.MPVariable(name=aux_plus_name, lower_bound=0))
      name_to_aux_plus[v.name] = aux_plus_name

      lns_mip.variable.append(
          mip_utils.MPVariable(name=aux_minus_name, lower_bound=0))
      name_to_aux_minus[v.name] = aux_minus_name

  # Build index lookup table for all variables.
  name_to_idx = {v.name: i for i, v in enumerate(lns_mip.variable)}

  # Calculate weights and coefficients, and create local branching constraints.
  var_index = []
  coeffs = []
  constraint_ub = neighbourhood_size
  for v, val in zip(mip.variable, incumbent.variable_value):
    if v.is_integer:
      w = 1.0 / (v.upper_bound - v.lower_bound)
      if np.isclose(val, v.lower_bound):
        var_index.append(name_to_idx[v.name])
        coeffs.append(w)
        constraint_ub += (w * v.lower_bound)
      elif np.isclose(val, v.upper_bound):
        var_index.append(name_to_idx[v.name])
        coeffs.append(-w)
        constraint_ub -= (w * v.upper_bound)
      else:
        var_index.append(name_to_idx[name_to_aux_plus[v.name]])
        coeffs.append(w)
        var_index.append(name_to_idx[name_to_aux_minus[v.name]])
        coeffs.append(w)

      # Add auxiliary constraints for general integers.
      if not mip_utils.is_var_binary(v):
        aux_constraint = mip_utils.MPConstraint(
            upper_bound=val, lower_bound=val, name='aux_constraint_' + v.name)
        aux_constraint.var_index.extend([
            name_to_idx[v.name], name_to_idx[name_to_aux_plus[v.name]],
            name_to_idx[name_to_aux_minus[v.name]]
        ])
        aux_constraint.coefficient.extend([1., 1., -1.])
        lns_mip.constraint.append(aux_constraint)

  # Add local branching constraint
  constraint = mip_utils.MPConstraint(
      upper_bound=constraint_ub, name='local_branching')
  constraint.var_index.extend(var_index)
  constraint.coefficient.extend(coeffs)
  lns_mip.constraint.append(constraint)

  return lns_mip


def get_lns_lp_solution(
    mip: mip_utils.MPModel,
    incumbent: mip_utils.MPSolutionResponse,
    neighbourhood_size: int,
    scip_params: ml_collections.ConfigDict,
    binary_only: bool = True) -> Optional[mip_utils.MPSolutionResponse]:
  """Builds local branching MIP and solves its LP relaxation.

  Args:
    mip: Input MIP.
    incumbent: The incumbent solution.
    neighbourhood_size: Maximum Hamming distance (across integer vars) from
      incumbent.
    scip_params: SCIP parameters used in the solve.
    binary_only: Whether to use binary or general local branching.

  Returns:
    The found solution (depending on time limit and SCIP
    params, might not be as good as incumbent), or None if no solution found.
  """
  if binary_only:
    lns_mip = get_binary_local_branching_mip(mip, incumbent, neighbourhood_size)
  else:
    lns_mip = get_general_local_branching_mip(mip, incumbent,
                                              neighbourhood_size)

  # Solve LP corresponding to lns_mip
  lp_solver = solving_utils.Solver()

  lp = copy.deepcopy(lns_mip)
  for var in lp.variable:
    var.is_integer = False

  lp_solver.load_model(lp)
  lp_solver.solve(scip_params)

  return lp_solver.get_best_solution()


def improve_solution(
    mip: mip_utils.MPModel,
    incumbent: mip_utils.MPSolutionResponse,
    neighbourhood_size: int,
    scip_params: ml_collections.ConfigDict,
    binary_only: bool = True,
    add_incumbent_to_scip: bool = True
) -> Optional[mip_utils.MPSolutionResponse]:
  """Defines an improvement step and solves it.

  Args:
    mip: Input MIP.
    incumbent: The incumbent solution.
    neighbourhood_size: Maximum Hamming distance (across integer vars) from
      incumbent.
    scip_params: SCIP parameters used in the solve.
    binary_only: Whether to use binary or general local branching.
    add_incumbent_to_scip: Whether to add the incumbent solution to SCIP.

  Returns:
    The found solution (depending on time limit and SCIP
    params, might not be as good as incumbent), or None if no solution found.

    Optionally also returns the SCIP stats from the solve call.
  """
  if binary_only:
    lns_mip = get_binary_local_branching_mip(mip, incumbent, neighbourhood_size)
  else:
    lns_mip = get_general_local_branching_mip(mip, incumbent,
                                              neighbourhood_size)
  mip_solver = solving_utils.Solver()
  mip_solver.load_model(lns_mip)

  if add_incumbent_to_scip:
    added = mip_solver.add_solution(incumbent)
    if added:
      logging.info('Added known solution with objective value: %f',
                   incumbent.objective_value)
    else:
      logging.warn('Failed to add known solution to SCIP')

  mip_solver.solve(scip_params)

  return mip_solver.get_best_solution()
