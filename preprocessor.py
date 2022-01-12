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
"""Wrapper APIs for MIP preprocessing."""

import abc
from typing import Optional, Tuple

from neural_lns import mip_utils


class Preprocessor(abc.ABC):
  """Class describing the API used to access a MIP presolver.

  This class should be used as a wrapper around any general presolving method
  MIPs, e.g. the presolver used in SCIP. The API only needs to expose a
  presolve method that turns a MPModel into a presolved MPModel, as well as a
  get_original_solution method that turns a solution to the presolved model to
  one a solution to the original.
  """

  def __init__(self, *args, **kwargs):
    """Initializes the preprocessor."""

  def presolve(
      self, mip: mip_utils.MPModel
  ) -> Tuple[mip_utils.MPSolverResponseStatus, Optional[mip_utils.MPModel]]:
    """Presolve the given MIP as MPModel.

    Args:
      mip: MPModel for MIP instance to presolve.

    Returns:
      status: A Status returned by the presolver.
      result: The MPModel of the presolved problem.
    """
    raise NotImplementedError('presolve method has to be implemented')

  def get_original_solution(
      self,
      solution: mip_utils.MPSolutionResponse) -> mip_utils.MPSolutionResponse:
    raise NotImplementedError(
        'get_original_solution method has to be implemented')
