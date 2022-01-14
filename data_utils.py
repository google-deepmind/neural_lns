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
"""Utility functions for feature extraction."""
import functools
from typing import Any, Dict, NamedTuple, Optional

from graph_nets import graphs
import ml_collections
import tensorflow.compat.v2 as tf

from neural_lns import mip_utils
from neural_lns import preprocessor
from neural_lns import solving_utils


BIAS_FEATURE_INDEX = 1
SOLUTION_FEATURE_INDEX = 14
BINARY_FEATURE_INDEX = 15

# Number of variable features without incumbent features.
NUM_ROOT_VARIABLE_FEATURES = 19

# Number of past incumbents to include in features.
NUM_PAST_INCUMBENTS = 3

# Total number of variable features.
NUM_VARIABLE_FEATURES = NUM_ROOT_VARIABLE_FEATURES + 2 * NUM_PAST_INCUMBENTS + 1

_INDICATOR_DIM = 1
_CON_FEATURE_DIM = 5

ORDER_TO_FEATURE_INDEX = {
    'coefficient': 6,
    'fractionality': 11,
}

# SCIP feature extraction parameters
SCIP_FEATURE_EXTRACTION_PARAMS = ml_collections.ConfigDict({
    'seed': 42,
    'time_limit_seconds': 60 * 10,
    'separating_maxroundsroot': 0,   # No cuts
    'conflict_enable': False,        # No additional cuts
    'heuristics_emphasis': 'off',    # No heuristics
})


class DatasetTuple(NamedTuple):
  state: Dict[str, tf.Tensor]
  graphs_tuple: graphs.GraphsTuple
  labels: tf.Tensor
  integer_labels: tf.Tensor
  integer_node_indices: tf.Tensor


def get_dataset_feature_metadata() -> Dict[str, tf.io.VarLenFeature]:
  """Returns the schema of the data for writing Neural LNS datasets."""
  features = {
      'constraint_features': tf.io.VarLenFeature(dtype=tf.string),
      'edge_features': tf.io.VarLenFeature(dtype=tf.string),
      'edge_indices': tf.io.VarLenFeature(dtype=tf.string),
      'variable_features': tf.io.VarLenFeature(dtype=tf.string),
      'variable_lbs': tf.io.VarLenFeature(dtype=tf.float32),
      'variable_ubs': tf.io.VarLenFeature(dtype=tf.float32),
      'constraint_feature_names': tf.io.VarLenFeature(dtype=tf.string),
      'variable_feature_names': tf.io.VarLenFeature(dtype=tf.string),
      'edge_features_names': tf.io.VarLenFeature(dtype=tf.string),
      'variable_names': tf.io.VarLenFeature(dtype=tf.string),
      'binary_variable_indices': tf.io.VarLenFeature(dtype=tf.int64),
      'all_integer_variable_indices': tf.io.VarLenFeature(dtype=tf.int64),
      'model_maximize': tf.io.VarLenFeature(dtype=tf.int64),
      'best_solution_labels': tf.io.VarLenFeature(dtype=tf.float32),
  }

  return features


def bnb_node_state_to_model_inputs(
    state: Dict[str, Any],
    node_depth: Optional[int] = None) -> graphs.GraphsTuple:
  """Convert a branch-and-bound node state into model inputs.

  Args:
    state: State information.
    node_depth: Depth of this search state.

  Returns:
    graph_tuple: The graph structure information.
  """
  variable_features = tf.where(
      tf.math.is_nan(state['variable_features']),
      tf.zeros_like(state['variable_features']),
      state['variable_features'])

  n_variables = tf.shape(variable_features)[0]
  variable_feature_dim = tf.shape(variable_features)[1]
  n_constraints = tf.shape(state['constraint_features'])[0]
  constraint_feature_dim = tf.shape(state['constraint_features'])[1]
  n_nodes = n_variables + n_constraints

  tf.Assert(constraint_feature_dim == _CON_FEATURE_DIM,
            [constraint_feature_dim])
  padded_variables = tf.pad(
      variable_features,
      [[0, 0], [0, constraint_feature_dim]],
      'CONSTANT')  # + constraint_feature_dim

  # Pad again with 1 to indicate variable corresponds to vertex.
  padded_variables = tf.pad(
      padded_variables,
      [[0, 0], [0, _INDICATOR_DIM]],
      'CONSTANT', constant_values=1.0)  # + 1

  padded_constraints = tf.pad(
      state['constraint_features'],
      [[0, 0], [variable_feature_dim, _INDICATOR_DIM]],
      'CONSTANT')  # + variable_feature_dim + 1

  nodes = tf.concat([padded_variables, padded_constraints], axis=0)
  edge_indices = tf.concat(
      [state['edge_indices'][:, :1] + tf.cast(n_variables, dtype=tf.int64),
       state['edge_indices'][:, 1:]], axis=1)

  edge_features = state['edge_features']
  node_features_dim = NUM_VARIABLE_FEATURES + _CON_FEATURE_DIM + 3

  graph_tuple = graphs.GraphsTuple(
      nodes=tf.cast(tf.reshape(nodes, [-1, node_features_dim]),
                    dtype=tf.float32),
      edges=tf.cast(edge_features, dtype=tf.float32),
      globals=tf.cast(node_depth, dtype=tf.float32),
      receivers=edge_indices[:, 0],  # constraint
      senders=edge_indices[:, 1],  # variables
      n_node=tf.reshape(n_nodes, [1]),
      n_edge=tf.reshape(tf.shape(state['edge_features'])[0], [1]))
  return graph_tuple


def convert_to_minimization(gt: graphs.GraphsTuple, state: Dict[str, Any]):
  """Changes the sign of the objective coefficients of all variable nodes.

  Args:
    gt: Input graph.
    state: Raw feature dictionary.

  Returns:
    graphs.GraphsTuple with updated nodes.
  """
  nodes = gt.nodes
  if tf.cast(state['model_maximize'], bool):
    num_vars = tf.shape(state['variable_features'])[0]
    feature_idx = ORDER_TO_FEATURE_INDEX['coefficient']
    indices = tf.stack([
        tf.range(num_vars),
        tf.broadcast_to(tf.constant(feature_idx), shape=[num_vars])
    ])
    indices = tf.transpose(indices)
    sign_change = tf.tensor_scatter_nd_update(
        tf.ones_like(nodes), indices,
        tf.broadcast_to(tf.constant(-1.0), shape=[num_vars]))
    nodes = nodes * sign_change

  return gt.replace(nodes=nodes)


def get_graphs_tuple(state: Dict[str, Any]) -> graphs.GraphsTuple:
  """Converts feature state into GraphsTuple."""
  state_with_bounds = state.copy()
  state_with_bounds['variable_features'] = tf.concat([
      state['variable_features'],
      tf.expand_dims(state['variable_lbs'], -1),
      tf.expand_dims(state['variable_ubs'], -1)
  ], -1)
  graphs_tuple = bnb_node_state_to_model_inputs(
      state_with_bounds, node_depth=1)
  graphs_tuple = convert_to_minimization(graphs_tuple, state_with_bounds)
  return graphs_tuple


def get_features(
    mip: mip_utils.MPModel,
    solver_params: ml_collections.ConfigDict = SCIP_FEATURE_EXTRACTION_PARAMS
    ) -> Optional[Dict[str, Any]]:
  """Extracts and preprocesses the features from the root of B&B tree."""
  mip_solver = solving_utils.Solver()
  presolver = preprocessor.Preprocessor()
  _, mip = presolver.presolve(mip)
  status = mip_solver.load_model(mip)
  features = None
  if status == mip_utils.MPSolverResponseStatus.NOT_SOLVED:
    features = mip_solver.extract_lp_features_at_root(solver_params)

  if features is not None and mip is not None:
    features['model_maximize'] = mip.maximize

  return features


def apply_feature_scaling(state, labels):
  """Scale variable bounds, solutions, coefficients and biases by sol norm.

  Out goal here is to scale continuous variables in such a way that we wouldn't
  change the integer feasible solutions to the MIP.
  In order to achieve that, we have to ensure that all constraints are scaled
  appropriately:
  a^Tx <= b can be rescaled without changes in the integer solutions via:
  (s * a_int)^Tx_int + a_cont^T(x_cont * s) <= s * b
  where
  - s = ||x_cont||^2,
  - a_int/cont are constraints coefficients corresponding to integer or
    continuous variables,
  - x_int/cont - solution values corresponding to integer or continuous
    variables.

  Args:
    state: dictionary with tensors corresponding to a single MIP instance
    labels: tensor with feasible solutions, including integer and continuous
    variables.

  Returns:
    state: dictionary with scaled tensors
    labels: tensor with scaled continuous solution values
  """
  sol = state['variable_features'][:, SOLUTION_FEATURE_INDEX]
  is_binary = state['variable_features'][:, BINARY_FEATURE_INDEX]
  is_non_integer = ~tf.cast(is_binary, tf.bool)
  continuous_sol = tf.boolean_mask(sol, is_non_integer)
  norm = tf.norm(continuous_sol)
  lbs = state['variable_lbs']
  ubs = state['variable_ubs']
  state['variable_lbs'] = tf.where(is_non_integer, lbs / norm, lbs)
  state['variable_ubs'] = tf.where(is_non_integer, ubs / norm, ubs)

  scaled_sol = tf.where(is_non_integer, sol / norm, sol)
  variable_features = tf.concat(
      [state['variable_features'][:, :SOLUTION_FEATURE_INDEX],
       tf.expand_dims(scaled_sol, axis=-1),
       state['variable_features'][:, SOLUTION_FEATURE_INDEX + 1:]],
      axis=1)
  state['variable_features'] = variable_features

  senders = state['edge_indices'][:, 1]
  is_integer_edge = tf.gather(~is_non_integer, senders)
  edges = tf.squeeze(state['edge_features'])
  scaled_edges = tf.where(is_integer_edge, edges / norm, edges)
  state['edge_features'] = tf.reshape(scaled_edges, [-1, 1])

  biases = state['constraint_features'][:, BIAS_FEATURE_INDEX]
  scaled_biases = biases / norm
  state['constraint_features'] = tf.concat([
      state['constraint_features'][:, :BIAS_FEATURE_INDEX],
      tf.reshape(scaled_biases, [-1, 1]),
      state['constraint_features'][:, BIAS_FEATURE_INDEX + 1:],
  ], axis=1)

  is_non_integer = tf.reshape(is_non_integer, [-1, 1])
  scaled_labels = tf.where(is_non_integer, labels / norm, labels)
  return state, scaled_labels


def decode_fn(record_bytes):
  """Decode a tf.train.Example.

   The list of (feature_name, feature_dtype, feature_ndim) is:
   [('variable_features', tf.float32, 2),
    ('binary_variable_indices', tf.int64, 1),
    ('model_maximize', tf.bool, 0),
    ('variable_names', tf.string, 1),
    ('constraint_features', tf.float32, 2),
    ('best_solution_labels', tf.float32, 1),
    ('variable_lbs', tf.float32, 1),
    ('edge_indices', tf.int64, 2),
    ('all_integer_variable_indices', tf.int64, 1),
    ('edge_features_names', tf.string, 0),
    ('variable_feature_names', tf.string, 0),
    ('constraint_feature_names', tf.string, 0),
    ('variable_ubs', tf.float32, 1),
    ('edge_features', tf.float32, 2)]

  Args:
     record_bytes: Serialised example.

  Returns:
    Deserialised example.
  """
  example = tf.io.parse_single_example(
      # Data
      record_bytes,

      # Schema
      get_dataset_feature_metadata()
  )

  # Parse all 2-D tensors and cast to the right dtype
  parsed_example = {}
  parsed_example['variable_features'] = tf.io.parse_tensor(tf.sparse.to_dense(
      example['variable_features'])[0], out_type=tf.float32)
  parsed_example['constraint_features'] = tf.io.parse_tensor(tf.sparse.to_dense(
      example['constraint_features'])[0], out_type=tf.float32)
  parsed_example['edge_indices'] = tf.io.parse_tensor(tf.sparse.to_dense(
      example['edge_indices'])[0], out_type=tf.int64)
  parsed_example['edge_features'] = tf.io.parse_tensor(tf.sparse.to_dense(
      example['edge_features'])[0], out_type=tf.float32)

  # Convert the remaining features to dense.
  for key, value in example.items():
    if key not in parsed_example:
      parsed_example[key] = tf.sparse.to_dense(value)

  return parsed_example


def extract_data(state: Dict[str, Any], scale_features: bool = False):
  """Create a DatasetTuple for each MIP instance."""
  num_vars = len(state['best_solution_labels'])
  labels = tf.reshape(state['best_solution_labels'], [num_vars, -1])

  if scale_features:
    state, labels = apply_feature_scaling(state, labels)

  if 'features_extraction_time' not in state:
    state['features_extraction_time'] = tf.constant(
        [], dtype=tf.float32)

  graphs_tuple = get_graphs_tuple(state)

  node_indices = tf.cast(state['binary_variable_indices'], tf.int32)

  # We allow filtering out instances that are invalid.
  valid_example = (tf.size(labels) > 0)

  if valid_example:
    int_labels = tf.gather(labels, node_indices)
    int_labels = tf.cast(tf.round(int_labels), tf.int32)
    int_labels = tf.cast(tf.expand_dims(int_labels, axis=-1), tf.int32)
  else:
    int_labels = tf.constant([], shape=[0, 0, 0], dtype=tf.int32)
    labels = tf.constant([], shape=[0, 0], dtype=tf.float32)

  return DatasetTuple(
      state=state,
      graphs_tuple=graphs_tuple,
      integer_node_indices=node_indices,
      labels=labels,
      integer_labels=int_labels)


def get_dataset(input_path: str,
                scale_features: bool = False,
                shuffle_size: int = 1000,
                num_epochs: Optional[int] = None) -> tf.data.Dataset:
  """Makes a tf.Dataset with correct preprocessing."""
  ds = tf.data.TFRecordDataset([input_path]).repeat(num_epochs)

  if shuffle_size > 0:
    ds = ds.shuffle(shuffle_size, reshuffle_each_iteration=True)

  data_fn = functools.partial(extract_data, scale_features=scale_features)
  return ds.map(decode_fn).map(data_fn)
