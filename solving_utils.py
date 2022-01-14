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
"""Common utilities for Solver."""

import abc
import enum
from typing import Any, Dict, Optional

import ml_collections

from neural_lns import mip_utils


class SolverState(enum.Enum):
  INIT = 0
  MODEL_LOADED = 1
  FINISHED = 2


class Solver(abc.ABC):
  """Wrapper around a given classical MIP solver.

  This class contains the API needed to communicate with a MIP solver, e.g.
  SCIP.
  """

  def load_model(self, mip: Any) -> SolverState:
    """Loads a MIP model into the solver."""
    raise NotImplementedError('load_model method should be implemented')

  def solve(
      self, solving_params: ml_collections.ConfigDict
  ) -> mip_utils.MPSolverResponseStatus:
    """Solves the loaded MIP model."""
    raise NotImplementedError('solve method should be implemented')

  def get_best_solution(self) -> Optional[Any]:
    """Returns the best solution found from the last solve call."""
    raise NotImplementedError('get_best_solution method should be implemented')

  def add_solution(self, solution: Any) -> bool:
    """Adds a known solution to the solver."""
    raise NotImplementedError('add_solution method should be implemented')

  def extract_lp_features_at_root(
      self, solving_params: ml_collections.ConfigDict) -> Dict[str, Any]:
    """Returns a dictionary of root node features."""
    raise NotImplementedError(
        'extract_lp_features_at_root method should be implemented')
