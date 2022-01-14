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
"""Training script for Neural Neighbourhood Search."""

import timeit
from typing import List, Tuple

from absl import app
from absl import flags
from absl import logging
import ml_collections
from ml_collections.config_flags import config_flags
import sonnet as snt
import tensorflow.compat.v2 as tf

from neural_lns import data_utils
from neural_lns import light_gnn

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    'config', 'neural_lns/config_train.py',
    'Training configuration.')


MIN_LEARNING_RATE = 1e-5


def train_and_evaluate(
    train_datasets: List[Tuple[str, str]],
    valid_datasets: List[Tuple[str, str]],
    strategy: tf.distribute.Strategy, learning_rate: float,
    model_dir: str, use_tf_function: bool, decay_steps: int,
    num_train_steps: int, num_train_run_steps: int, eval_every_steps: int,
    eval_steps: int, grad_clip_norm: float,
    model_config: ml_collections.ConfigDict):
  """The main training and evaluation loop."""
  if eval_every_steps % num_train_run_steps != 0:
    raise ValueError(
        'eval_every_steps is not divisible by num_train_run_steps')

  train_ds_all = []
  for path, _ in train_datasets:
    train_ds = data_utils.get_dataset(path)
    train_ds_all.append(train_ds)
  train_data = tf.data.Dataset.sample_from_datasets(train_ds_all)

  valid_ds_all = []
  for path, _ in valid_datasets:
    valid_ds = data_utils.get_dataset(path)
    valid_ds_all.append(valid_ds)
  valid_data = tf.data.Dataset.sample_from_datasets(valid_ds_all)

  with strategy.scope():
    model = light_gnn.get_model(**model_config.params)
    global_step = tf.Variable(
        0, trainable=False, name='global_step', dtype=tf.int64)
    lr_schedule = tf.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=learning_rate,
        decay_steps=decay_steps,
        decay_rate=0.9)
    optimizer = snt.optimizers.Adam(learning_rate)
    train_acc_metric = tf.keras.metrics.BinaryAccuracy(threshold=0.5)
    valid_acc_metric = tf.keras.metrics.BinaryAccuracy(threshold=0.5)
    train_auc_metric = tf.keras.metrics.AUC()
    valid_auc_metric = tf.keras.metrics.AUC()

  def train_step(train_inputs):
    """Perform a single training step. Returns the loss."""

    # step_fn is replicated when running with TPUStrategy.
    def step_fn(ds_tuple: data_utils.DatasetTuple):
      logging.info('retracing step_fn')

      with tf.GradientTape() as tape:
        logits = model(
            ds_tuple.graphs_tuple,
            is_training=True,
            node_indices=ds_tuple.integer_node_indices,
            labels=ds_tuple.integer_labels)
        local_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.cast(ds_tuple.integer_labels, tf.float32),
            logits=logits)
        tf.assert_equal(tf.shape(ds_tuple.integer_labels), tf.shape(logits))

        local_loss = tf.reduce_sum(local_loss, axis=[2], keepdims=True)
        local_loss = tf.reduce_mean(local_loss, axis=[0])
        local_loss = local_loss / strategy.num_replicas_in_sync

      tf.print('Local loss', local_loss)

      # We log AUC and ACC by comparing the greedy sample (always choose the
      # value with highest probability) with the best solution.
      _, proba = model.greedy_sample(ds_tuple.graphs_tuple,
                                     ds_tuple.integer_node_indices)
      proba = tf.reshape(proba, [-1])

      best_label = tf.reshape(ds_tuple.integer_labels[:, 0, :], [-1])
      train_acc_metric.update_state(best_label, proba)
      train_auc_metric.update_state(best_label, proba)

      replica_ctx = tf.distribute.get_replica_context()
      grads = tape.gradient(local_loss, model.trainable_variables)
      grads = replica_ctx.all_reduce('sum', grads)
      if grad_clip_norm > 0:
        grads, _ = tf.clip_by_global_norm(grads, grad_clip_norm)
      lr = tf.maximum(lr_schedule(optimizer.step), MIN_LEARNING_RATE)
      optimizer.learning_rate = lr
      optimizer.apply(grads, model.trainable_variables)
      global_step.assign_add(1)
      return local_loss

    losses = []
    for _ in range(num_train_run_steps):
      per_replica_losses = strategy.run(
          step_fn, args=(next(train_inputs),))
      loss = strategy.reduce(
          tf.distribute.ReduceOp.SUM,
          per_replica_losses,
          axis=None)
      losses.append(loss)
    return tf.reduce_mean(losses)

  def eval_step(eval_inputs):

    def step_fn(ds_tuple: data_utils.DatasetTuple):  # pylint: disable=missing-docstring
      logits = model(
          ds_tuple.graphs_tuple,
          is_training=False,
          node_indices=ds_tuple.integer_node_indices,
          labels=ds_tuple.integer_labels)

      local_loss = tf.nn.sigmoid_cross_entropy_with_logits(
          labels=tf.cast(ds_tuple.integer_labels, tf.float32),
          logits=logits)

      local_loss = tf.reduce_sum(local_loss, axis=[2], keepdims=True)
      local_loss = tf.reduce_mean(local_loss, axis=[0])
      # We scale the local loss here so we can later sum the gradients.
      # This is cheaper than averaging all gradients.
      local_loss = local_loss / strategy.num_replicas_in_sync

      # We log AUC and ACC by comparing the greedy sample (always choose the
      # value with highest probability) with the best solution.
      _, proba = model.greedy_sample(ds_tuple.graphs_tuple,
                                     ds_tuple.integer_node_indices)
      proba = tf.reshape(proba, [-1])
      best_label = tf.reshape(ds_tuple.integer_labels[:, 0, :], [-1])
      valid_acc_metric.update_state(best_label, proba)
      valid_auc_metric.update_state(best_label, proba)

      return local_loss

    valid_losses = []
    for _ in range(eval_steps):
      valid_losses_per_replica = strategy.run(
          step_fn, args=(next(eval_inputs),))
      valid_loss = strategy.reduce(
          tf.distribute.ReduceOp.SUM,
          valid_losses_per_replica,
          axis=None)
      valid_losses.append(valid_loss)
    return tf.reduce_mean(valid_losses)

  if use_tf_function:
    train_step = tf.function(train_step)
    eval_step = tf.function(eval_step)

  ckpt = tf.train.Checkpoint(
      model=model, optimizer=optimizer, global_step=global_step)
  ckpt_manager = tf.train.CheckpointManager(
      checkpoint=ckpt, directory=model_dir, max_to_keep=5)
  ckpt.restore(ckpt_manager.latest_checkpoint)
  if ckpt_manager.latest_checkpoint:
    logging.info('Restored from %s', ckpt_manager.latest_checkpoint)
  else:
    logging.info('Initializing from scratch.')

  train_inputs = iter(train_data)
  logging.info('Starting training...')
  while global_step.numpy() < num_train_steps:
    start = timeit.default_timer()
    loss = train_step(train_inputs)
    end = timeit.default_timer()

    step = global_step.numpy()
    train_acc = train_acc_metric.result().numpy()
    train_auc = train_auc_metric.result().numpy()
    train_acc_metric.reset_states()
    train_auc_metric.reset_states()

    logging.info(f'[{step}] loss = {loss.numpy():.4f}, ' +
                 f'acc = {train_acc:.4f} auc = {train_auc:.4f} ' +
                 f'steps_per_second = {num_train_run_steps / (end - start)}')

    if step % eval_every_steps == 0:
      model.save_model(model_dir)
      eval_inputs = iter(valid_data)
      valid_loss = eval_step(eval_inputs)

      valid_acc = valid_acc_metric.result().numpy()
      valid_auc = valid_auc_metric.result().numpy()
      valid_acc_metric.reset_states()
      valid_auc_metric.reset_states()

      logging.info(f'[Valid: {step}] acc = ' +
                   f'{valid_acc:.4f} auc = {valid_auc:.4f} ' +
                   f'loss = {valid_loss:.4f}')

  saved_ckpt = ckpt_manager.save()
  logging.info('Saved checkpoint: %s', saved_ckpt)


def main(_):
  flags_config = FLAGS.config
  gpus = tf.config.experimental.list_logical_devices(device_type='GPU')
  if gpus:
    logging.info('Found GPUs: %s', gpus)
    strategy = snt.distribute.Replicator([g.name for g in gpus])
  else:
    strategy = tf.distribute.OneDeviceStrategy('CPU')

  logging.info('Distribution strategy: %s', strategy)
  logging.info('Devices: %s', tf.config.list_physical_devices())

  train_and_evaluate(
      train_datasets=flags_config.train_datasets,
      valid_datasets=flags_config.valid_datasets,
      strategy=strategy,
      learning_rate=flags_config.learning_rate,
      model_dir=flags_config.work_unit_dir,
      use_tf_function=True,
      decay_steps=flags_config.decay_steps,
      num_train_steps=flags_config.num_train_steps,
      num_train_run_steps=flags_config.num_train_run_steps,
      eval_every_steps=flags_config.eval_every_steps,
      eval_steps=flags_config.eval_steps,
      grad_clip_norm=flags_config.grad_clip_norm,
      model_config=flags_config.model_config)


if __name__ == '__main__':
  tf.enable_v2_behavior()
  app.run(main)
