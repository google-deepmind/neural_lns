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
"""SolutionData classes used to log solution process."""

import abc
import math
from typing import Any, Callable, Dict, List, Optional

from absl import logging


class BaseSolutionData(abc.ABC):
  """Base class for SolutionData.

  This class encapsulates information that were logged during the solving
  process. This includes primal bound improvements, the current best feasible
  solution, and the elapsed time at each improvement.
  """

  @abc.abstractproperty
  def objective_type(self) -> Callable[[Any, Any], Any]:
    raise NotImplementedError('objective_type property has to be implemented')

  @abc.abstractproperty
  def primal_bounds(self) -> List[float]:
    raise NotImplementedError('primal_bounds property has to be implemented')

  @abc.abstractproperty
  def calibrated_time(self) -> List[Optional[float]]:
    raise NotImplementedError('calibrated_time property has to be implemented')

  @abc.abstractproperty
  def time_in_seconds(self) -> List[float]:
    raise NotImplementedError('time_in_seconds property has to be implemented')

  @abc.abstractproperty
  def original_solutions(self) -> List[Any]:
    raise NotImplementedError(
        'original_solutions property has to be implemented')

  @abc.abstractproperty
  def best_original_solution(
      self) -> Optional[Any]:
    raise NotImplementedError(
        'best_original_solution property has to be implemented')

  @abc.abstractproperty
  def elapsed_real_time(self) -> float:
    raise NotImplementedError(
        'elapsed_real_time property has to be implemented')

  @abc.abstractproperty
  def elapsed_calibrated_time(self) -> Optional[float]:
    raise NotImplementedError(
        'elapsed_calibrated_time property has to be implemented')

  @abc.abstractmethod
  def _write(self, log_entry: Dict[str, Any], force_save_sol: bool):
    raise NotImplementedError('write method has to be implemented')

  def write(self, log_entry: Dict[str, Any], force_save_sol: bool = False):
    if ((log_entry['best_primal_point'] is not None or
         log_entry['primal_bound'] is not None) and
        abs(log_entry['primal_bound']) < 1e19):
      self._write(log_entry, force_save_sol)


class SolutionData(BaseSolutionData):
  """This is a basic implementation of BaseSolutionData."""

  def __init__(self,
               objective_type: Callable[[Any, Any], Any],
               write_intermediate_sols: bool = False):
    """The key solution process logging class.

    Args:
      objective_type: decides if the objective should be decreasing or
        increasing.
      write_intermediate_sols: controls if we're recording intermediate
        solutions of the solve (this is necessary for joint evals with
        DeepBrancher)
    """
    self._objective_type = objective_type
    self._write_intermediate_sols = write_intermediate_sols
    self._primal_bounds = []
    self._calibrated_time = []
    self._time_in_seconds = []
    self._original_solutions = []

  def _ensure_valid_primal_bound_update(self):
    """Ensures that primal bounds are monotonic, repairs them and logs a warning if not."""
    # Given the logging logic, solutions should be monotonically improving.
    if len(self._primal_bounds) > 1:
      better_bound = self._objective_type(
          self._primal_bounds[-1], self._primal_bounds[-2])
      if not math.isclose(self._primal_bounds[-1], better_bound,
                          rel_tol=1e-5, abs_tol=1e-5):
        logging.warn('Primal bounds were not be monotonic: %d and %d',
                     self._primal_bounds[-1], self._primal_bounds[-2])
        self._primal_bounds[-1] = better_bound

  @property
  def objective_type(self) -> Callable[[Any, Any], Any]:
    return self._objective_type

  @property
  def primal_bounds(self) -> List[float]:
    return self._primal_bounds

  @property
  def calibrated_time(self) -> List[Optional[float]]:
    return self._calibrated_time

  @property
  def time_in_seconds(self) -> List[float]:
    return self._time_in_seconds

  @property
  def original_solutions(self) -> List[Any]:
    return [sol for sol in self._original_solutions if sol is not None]

  @property
  def best_original_solution(
      self) -> Optional[Any]:
    best_orig_sol = None
    if self._original_solutions and self._original_solutions[-1] is not None:
      best_orig_sol = self._original_solutions[-1]
    return best_orig_sol

  @property
  def elapsed_real_time(self) -> float:
    elapsed_time = 0.0
    if self._time_in_seconds:
      elapsed_time = self._time_in_seconds[-1]
    return elapsed_time

  @property
  def elapsed_calibrated_time(self) -> Optional[float]:
    elapsed_time = 0.0
    if self._calibrated_time:
      elapsed_time = self._calibrated_time[-1]
    return elapsed_time

  def _write(self, log_entry: Dict[str, Any], force_save_sol: bool):
    """Log a new solution (better primal bound) for this particular instance.

    Args:
      log_entry: the dictionary with logging information.
      force_save_sol: to be used for final solutions to be recorded even if
        write_intermediate_sols is off. Otherwise we would record no actual
        solutions in the SolutionData.
    """
    sol = log_entry['best_primal_point']
    if sol:
      assert math.isclose(sol.objective_value, log_entry['primal_bound'])
    self._time_in_seconds.append(log_entry['solving_time'])
    self._calibrated_time.append(log_entry['solving_time_calibrated'])
    self._primal_bounds.append(log_entry['primal_bound'])
    if self._write_intermediate_sols or force_save_sol:
      if sol is None:
        raise ValueError('Trying to write full solution on None')
      self._original_solutions.append(sol)
    else:
      self._original_solutions.append(None)
    self._ensure_valid_primal_bound_update()
