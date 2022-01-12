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
"""MIP utility functions."""

import copy
import dataclasses
import enum
import math
from typing import Any, List

from absl import logging
import numpy as np

from neural_lns import sampling


class MPSolverResponseStatus(enum.Enum):
  """Enum of solver statuses."""
  OPTIMAL = 0
  FEASIBLE = 1
  NOT_SOLVED = 2
  INFEASIBLE = 3
  UNBOUNDED = 4
  INFEASIBLE_OR_UNBOUNDED = 5
  STOPPED = 6
  UNKNOWN = 7
  FAILED = 8
  BESTSOLLIMIT = 9


@dataclasses.dataclass
class MPVariable:
  """MPVariable contains all the information related to a single variable."""
  # Lower and upper bounds; lower_bound must be <= upper_bound.
  lower_bound: float = -math.inf
  upper_bound: float = math.inf
  # The coefficient of the variable in the objective. Must be finite.
  objective_coefficient: float = 0.0
  # True if the variable is constrained to be integer.
  is_integer: bool = True
  # The name of the variable.
  name: str = ""


@dataclasses.dataclass
class MPConstraint:
  """MPConstraint contains all the information related to a single constraint."""
  # var_index[i] is the variable index (w.r.t. to "variable" field of
  # MPModel) of the i-th linear term involved in this constraint, and
  # coefficient[i] is its coefficient. Only the terms with non-zero
  # coefficients need to appear. var_index may not contain duplicates.
  var_index: List[int] = dataclasses.field(default_factory=list)
  coefficient: List[float] = dataclasses.field(default_factory=list)
  # lower_bound must be <= upper_bound.
  lower_bound: float = -math.inf
  upper_bound: float = math.inf
  # The name of the constraint.
  name: str = ""


@dataclasses.dataclass
class MPModel:
  """MPModel fully encodes a Mixed-Integer Linear Programming model."""
  # All the variables appearing in the model.
  variable: List[MPVariable] = dataclasses.field(default_factory=list)
  # All the constraints appearing in the model.
  constraint: List[MPConstraint] = dataclasses.field(default_factory=list)
  # True if the problem is a maximization problem. Minimize by default.
  maximize: bool = False
  # Offset for the objective function. Must be finite.
  objective_offset: float = 0.0
  # Name of the model.
  name: str = ""


@dataclasses.dataclass
class MPSolutionResponse:
  """Class for solution response from the solver."""
  # Objective value corresponding to the "variable_value" below, taking into
  # account the source "objective_offset" and "objective_coefficient".
  objective_value: float
  # Variable values in the same order as the MPModel.variable field.
  # This is a dense representation. These are set iff 'status' is OPTIMAL or
  # FEASIBLE.
  variable_value: List[float]
  # Human-readable status string.
  status_str: str
  # Result of the optimization.
  status: MPSolverResponseStatus = MPSolverResponseStatus.UNKNOWN


def tighten_variable_bounds(mip: Any,
                            names: List[str],
                            lbs: List[float],
                            ubs: List[float]):
  """Tightens variables of the given MIP in-place.

  Args:
    mip: Input MIP.
    names: List of variable names to tighten.
    lbs: List of lower bounds, in same order as names.
    ubs: List of lower bounds, in same order as names.
  """
  if len(names) != len(lbs) or len(lbs) != len(ubs):
    raise ValueError(
        "Names, lower and upper bounds should have the same length")

  name_to_bounds = {}
  for name, lb, ub in zip(names, lbs, ubs):
    name = name.decode() if isinstance(name, bytes) else name
    name_to_bounds[name] = (lb, ub)

  c = 0
  for v in mip.variable:
    name = v.name.decode() if isinstance(v.name, bytes) else v.name
    if name in name_to_bounds:
      lb, ub = name_to_bounds[name]
      v.lower_bound = max(lb, v.lower_bound)
      v.upper_bound = min(ub, v.upper_bound)
      c += 1

  logging.info("Tightened %s vars", c)


def is_var_binary(variable: Any) -> bool:
  """Checks whether a given variable is binary."""
  lb_is_zero = np.isclose(variable.lower_bound, 0)
  ub_is_one = np.isclose(variable.upper_bound, 1)
  return variable.is_integer and lb_is_zero and ub_is_one


def add_binary_invalid_cut(mip: Any,
                           names: List[str],
                           values: List[int],
                           weights: List[float],
                           depth: float):
  """Adds a weighted binary invalid cut to the given MIP in-place.

  Given a binary assignment for all or some of the binary variables, adds
  a constraint in the form:

  sum_{i in zeros} w_i * x_i + sum_{j in ones} w_j * (1-x_j) <= d

  The first summation is over variables predicted to be zeros, the second
  summation is over variables predicted to be ones. d is the maximum distance
  allowed for a solution to be away from predicted assignment.

  Args:
    mip: Input MIP.
    names: Binary variable names.
    values: Predicted values of binary variables.
    weights: Weights associated with cost inccured by reversing prediction.
    depth: The amount of cost allowed to be incurred by flipping
      assignments.
  """
  assert len(names) == len(values) == len(weights)

  name_to_idx = {}
  for i, v in enumerate(mip.variable):
    name = v.name.decode() if isinstance(v.name, bytes) else v.name
    name_to_idx[name] = i

  ub = depth
  var_index = []
  coeffs = []

  for name, val, w in zip(names, values, weights):
    name = name.decode() if isinstance(name, bytes) else name
    assert is_var_binary(mip.variable[name_to_idx[name]])
    var_index.append(name_to_idx[name])

    if val == 1:
      ub -= w
      coeffs.append(-w)
    else:
      coeffs.append(w)

  constraint = mip.constraint.add()
  constraint.var_index.extend(var_index)
  constraint.coefficient.extend(coeffs)
  constraint.upper_bound = ub
  constraint.name = "weighted_invalid_cut"


def make_sub_mip(mip: Any, assignment: sampling.Assignment):
  """Creates a sub-MIP by tightening variables and applying cut."""
  sub_mip = copy.deepcopy(mip)
  tighten_variable_bounds(sub_mip, assignment.names,
                          assignment.lower_bounds, assignment.upper_bounds)
  return sub_mip
