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
"""Solvers used to solve MIPs."""

import abc
import collections as py_collections
from typing import Any, Dict, Optional, Tuple

from absl import logging
import ml_collections
import numpy as np

from neural_lns import calibration
from neural_lns import data_utils
from neural_lns import local_branching_data_generation as lns_data_gen
from neural_lns import mip_utils
from neural_lns import preprocessor
from neural_lns import sampling
from neural_lns import solution_data
from neural_lns import solving_utils


class BaseSolver(abc.ABC):
  """Base class for solvers.

  This class encapsulates an overall MIP solver / primal heuristic. We provide
  three implementations, one wrapping around any classical solver (e.g. SCIP),
  one implementing Neural Diving, and one implementing Neural Neighbourhood
  Selection.

  The API simply exposes a solve method which solves a given MIP and logs all
  solution information inside a BaseSolutionData object.
  """

  def __init__(self,
               solver_config: ml_collections.ConfigDict,
               sampler: Optional[sampling.BaseSampler] = None):
    self._solver_config = solver_config
    self._sampler = sampler

  @abc.abstractmethod
  def solve(
      self, mip: Any, sol_data: solution_data.BaseSolutionData,
      timer: calibration.Timer
  ) -> Tuple[solution_data.BaseSolutionData, Dict[str, Any]]:
    raise NotImplementedError('solve method should be implemented')


class SCIPSolver(BaseSolver):
  """Agent that solves MIP with SCIP."""

  def solve(
      self, mip: Any, sol_data: solution_data.BaseSolutionData,
      timer: calibration.Timer
  ) -> Tuple[solution_data.BaseSolutionData, Dict[str, Any]]:
    status, sol_data, _ = scip_solve(
        mip=mip,
        scip_solve_config=self._solver_config,
        sol_data=sol_data,
        timer=timer)
    stats = {}
    stats['solution_status'] = str(status)
    return sol_data, stats


class NeuralDivingSolver(BaseSolver):
  """Solver that implements Neural Diving."""

  def solve(
      self, mip: Any, sol_data: solution_data.BaseSolutionData,
      timer: calibration.Timer
  ) -> Tuple[solution_data.BaseSolutionData, Dict[str, Any]]:
    sub_mip, stats = predict_and_create_sub_mip(
        mip, self._sampler, self._solver_config.predict_config)
    status, sol_data, sub_mip_sol = scip_solve(
        sub_mip, self._solver_config.scip_solver_config, sol_data, timer)
    if self._solver_config.enable_restart:
      status, sol_data, _ = scip_solve(
          mip, self._solver_config.restart_scip_solver_config, sol_data, timer,
          sub_mip_sol)

    stats['solution_status'] = str(status)
    return sol_data, stats


class NeuralNSSolver(BaseSolver):
  """Solver that implements Neural Neighbourhood Selection."""

  def solve(
      self, mip: Any, sol_data: solution_data.BaseSolutionData,
      timer: calibration.Timer
  ) -> Tuple[solution_data.BaseSolutionData, Dict[str, Any]]:

    # First run Neural Diving to get an initial incumbent solution:
    diving_config = self._solver_config.diving_config
    sampler_name = diving_config.predict_config.sampler_config.name
    if sampler_name not in sampling.SAMPLER_DICT:
      # Just run pure SCIP on original MIP in this case:
      status, sol_data, incumbent_sol = scip_solve(
          mip, diving_config.scip_solver_config, sol_data, timer)
    else:
      diving_sampler = sampling.SAMPLER_DICT[
          diving_config.predict_config.sampler_config.name](
              self._solver_config.diving_model)

      sub_mip, _ = predict_and_create_sub_mip(mip, diving_sampler,
                                              diving_config.predict_config)
      _, sol_data, incumbent_sol = scip_solve(sub_mip,
                                              diving_config.scip_solver_config,
                                              sol_data, timer)

    if incumbent_sol is None:
      logging.warn('Did not find incumbent solution for MIP: %s, skipping',
                   mip.name)
      return sol_data, {}

    past_incumbents = py_collections.deque([incumbent_sol])

    # Extract the root features here, as we need to expand by adding the values
    # of the incumbent solution, which the model also saw during training.
    root_features = data_utils.get_features(mip)
    if root_features is None:
      logging.warn('Could not extract features from MIP: %s, skipping',
                   mip.name)
      return sol_data, {}

    dummy_columns = np.zeros((root_features['variable_features'].shape[0],
                              2 * lns_data_gen.NUM_PAST_INCUMBENTS + 1),
                             dtype=root_features['variable_features'].dtype)

    root_features['variable_features'] = np.concatenate(
        [root_features['variable_features'], dummy_columns], axis=1)

    # Enhance the features with the incumbent solution:
    features = lns_data_gen.enhance_root_features(root_features,
                                                  past_incumbents, mip)

    # The last column of enhanced features masks out continuous variables
    num_integer_variables = np.sum(features['variable_features'][:, -1])
    current_neighbourhood_size = int(self._solver_config.perc_unassigned_vars *
                                     num_integer_variables)
    sampler_params = self._solver_config.predict_config.sampler_config.params
    update_dict = {'num_unassigned_vars': int(current_neighbourhood_size)}
    sampler_params.update(update_dict)

    # Keep and return stats from first step
    sub_mip, stats = predict_and_create_lns_sub_mip(
        mip, self._sampler, features.copy(), self._solver_config.predict_config)

    for s in range(self._solver_config.num_solve_steps):
      incumbent_sol = past_incumbents[0]
      status, sol_data, improved_sol = scip_solve(
          sub_mip, self._solver_config.scip_solver_config, sol_data, timer,
          incumbent_sol)

      logging.info('NLNS step: %s, solution status: %s', s, status)
      if status in (mip_utils.MPSolverResponseStatus.OPTIMAL,
                    mip_utils.MPSolverResponseStatus.INFEASIBLE,
                    mip_utils.MPSolverResponseStatus.BESTSOLLIMIT):
        current_neighbourhood_size = min(
            current_neighbourhood_size * self._solver_config.temperature,
            0.6 * num_integer_variables)
      else:
        current_neighbourhood_size = max(
            current_neighbourhood_size // self._solver_config.temperature, 20)
      logging.info('Updated neighbourhood size to: %s',
                   int(current_neighbourhood_size))

      sampler_params = self._solver_config.predict_config.sampler_config.params
      update_dict = {'num_unassigned_vars': int(current_neighbourhood_size)}
      sampler_params.update(update_dict)
      logging.info('%s', self._solver_config.predict_config)

      if improved_sol is None:
        break
      # Add improved solution to buffer.
      past_incumbents.appendleft(improved_sol)
      if len(past_incumbents) > lns_data_gen.NUM_PAST_INCUMBENTS:
        past_incumbents.pop()

      # Recompute the last two columns of the features with new incumbent:
      features = lns_data_gen.enhance_root_features(root_features,
                                                    past_incumbents, mip)

      # Compute the next sub-MIP based on new incumbent:
      sub_mip, _ = predict_and_create_lns_sub_mip(
          mip, self._sampler, features.copy(),
          self._solver_config.predict_config)

    stats['solution_status'] = str(status)
    return sol_data, stats


def scip_solve(
    mip: Any,
    scip_solve_config: ml_collections.ConfigDict,
    sol_data: solution_data.BaseSolutionData,
    timer: calibration.Timer,
    best_known_sol: Optional[Any] = None
) -> Tuple[mip_utils.MPSolverResponseStatus, solution_data.BaseSolutionData,
           Optional[Any]]:
  """Uses SCIP to solve the MIP and writes solutions to SolutionData.

  Args:
    mip: MIP to be solved
    scip_solve_config: config for SCIPWrapper
    sol_data: SolutionData to write solving data to
    timer: timer to use to record real elapsed time and (possibly) calibrated
      elapsed time
    best_known_sol: previously known solution for the MIP to start solving
      process with

  Returns:
     Status of the solving process.
     SolutionData with filled solution data. All solutions are converted to the
       original space according to SolutionDataWrapper transform functions
     Best solution to the MIP passed to SCIP, not retransformed by SolutionData
       to the original space (this is convenient for restarts)
  """
  # Initialize SCIP solver and load the MIP
  mip_solver = solving_utils.Solver()
  mip_solver.load_model(mip)

  # Try to load the best known solution to SCIP
  if best_known_sol is not None:
    added = mip_solver.add_solution(best_known_sol)
    if added:
      logging.info('Added solution to SCIP with obj: %f',
                   best_known_sol.objective_value)
    else:
      logging.warn('Could not add solution to SCIP')

  # Solve the MIP with given config
  try:
    status = mip_solver.solve(scip_solve_config.params)
  finally:
    best_solution = mip_solver.get_best_solution()

  # Add final solution to the solution data
  if best_solution is not None:
    log_entry = {}
    log_entry['best_primal_point'] = best_solution
    log_entry['primal_bound'] = best_solution.objective_value
    log_entry['solving_time'] = timer.elapsed_real_time
    log_entry['solving_time_calibrated'] = timer.elapsed_calibrated_time
    sol_data.write(log_entry, force_save_sol=True)
  return status, sol_data, best_solution


def predict_and_create_sub_mip(
    mip: Any, sampler: sampling.BaseSampler,
    config: ml_collections.ConfigDict) -> Tuple[Any, Dict[str, Any]]:
  """Takes in a MIP and a config and outputs a sub-MIP.

  If the MIP is found infeasible or trivially optimal during feature extraction,
  then no SuperMIP reductions are applied and the original MIP is passed back.

  Args:
    mip: MIP that is used to produce a sub-MIP
    sampler: Sampler used to produce predictions
    config: config used to feature extraction and model sampling

  Returns:
    (sub-)MIP
    Dict with assignment stats:
      num_variables_tightened: how many variables were tightened in an assigment
      num_variables_cut: how many variables were used in an invalid cut, usually
        0 (if cut was enabled) or all of them (if cut was disabled).
  """
  # Step 1: Extract MIP features
  features = data_utils.get_features(
      mip, solver_params=config.extract_features_scip_config)
  if features is None:
    logging.warn('Could not extract features from MIP: %s, skipping', mip.name)
    return mip, {}

  # Step 2: Perform sampling
  node_indices = features['binary_variable_indices']
  var_names = features['variable_names']
  variable_lbs = features['variable_lbs']
  variable_ubs = features['variable_ubs']
  graphs_tuple = data_utils.get_graphs_tuple(features)

  assignment = sampler.sample(graphs_tuple, var_names, variable_lbs,
                              variable_ubs, node_indices,
                              **config.sampler_config.params)
  sub_mip = mip_utils.make_sub_mip(mip, assignment)

  return sub_mip


def predict_and_create_lns_sub_mip(
    mip: Any, sampler: sampling.BaseSampler, features: Any,
    config: ml_collections.ConfigDict) -> Tuple[Any, Dict[str, Any]]:
  """Produces a sub-MIP for LNS derived from model predictions.

  This function uses the provided sampler to predict which binary variables of
  the MIP should be unassigned. From this prediction we derive a sub-MIP where
  the remaining variables are fixed to the values provided in `incumbent_dict`.

  Args:
    mip: MIP that is used to produce a sub-MIP
    sampler: SuperMIP sampler used to produce predictions
    features: Model features used for sampling.
    config: config used to feature extraction and model sampling

  Returns:
    (sub-)MIP
    Dict with assignment stats:
      num_variables_tightened: how many variables were tightened in an assigment
      num_variables_cut: how many variables were used in an invalid cut, usually
        0 (if cut was enabled) or all of them (if cut was disabled).
  """
  node_indices = features['binary_variable_indices']
  var_names = features['variable_names']
  var_values = np.asarray([var_name.decode() for var_name in var_names])
  graphs_tuple = data_utils.get_graphs_tuple(features)

  assignment = sampler.sample(graphs_tuple, var_names, var_values, node_indices,
                              **config.sampler_config.params)
  sub_mip = mip_utils.make_sub_mip(mip, assignment)

  return sub_mip


SOLVING_AGENT_DICT = {
    'scip': SCIPSolver,
    'neural_diving': NeuralDivingSolver,
    'neural_ns': NeuralNSSolver,
}


def run_solver(
    mip: Any, solver_running_config: ml_collections.ConfigDict,
    solver: BaseSolver
) -> Tuple[solution_data.BaseSolutionData, Dict[str, Any]]:
  """End-to-end MIP solving with a Solver.

  Args:
    mip: MIP that is used to produce a sub-MIP
    solver_running_config: config to run the provided solver
    solver: initialized solver to be used

  Returns:
    SolutionData
    Dict with additional stats:
      solution_status: the returned status by the solver.
      elapsed_time_seconds: end-to-end time for instance in real time seconds.
      elapsed_time_calibrated: end-to-end time for instance in calibrated time.
    And for NeuralDivingSolver, additionally:
      num_variables_tightened: how many variables were tightened in an
        assigment
      num_variables_cut: how many variables were used in an invalid cut,
        usually 0 (if cut was enabled) or all of them (if cut was disabled).
  """

  # Stage 1: set up a timer
  timer = calibration.Timer()
  timer.start_and_wait()

  # Stage 2: presolve the original MIP instance
  presolver = None
  presolved_mip = mip
  if solver_running_config.preprocessor_configs is not None:
    presolver = preprocessor.Preprocessor(
        solver_running_config.preprocessor_configs)
    _, presolved_mip = presolver.presolve(mip)

  # Stage 3: setup solution data
  objective_type = max if mip.maximize else min
  sol_data = solution_data.SolutionData(
      objective_type=objective_type,
      write_intermediate_sols=solver_running_config.write_intermediate_sols)
  if presolver is not None:
    sol_data = solution_data.SolutionDataWrapper(
        sol_data, sol_transform_fn=presolver.get_original_solution)

  # Stage 4: Solve MIP
  sol_data, solve_stats = solver.solve(presolved_mip, sol_data, timer)

  timer.terminate_and_wait()
  solve_stats['elapsed_time_seconds'] = timer.elapsed_real_time
  solve_stats['elapsed_time_calibrated'] = timer.elapsed_calibrated_time
  return sol_data, solve_stats
