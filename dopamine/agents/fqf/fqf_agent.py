# coding=utf-8
# Copyright 2018 The Dopamine Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The implicit quantile networks (IQN) agent.

The agent follows the description given in "Implicit Quantile Networks for
Distributional RL" (Dabney et. al, 2018).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dopamine.agents.rainbow import rainbow_agent
from dopamine.discrete_domains import atari_lib
import tensorflow.compat.v1 as tf
import gin.tf
import os


@gin.configurable
class FullyParameterizedQuantileAgent(rainbow_agent.RainbowAgent):
  """An extension of Rainbow to perform implicit quantile regression."""
  
  def __init__(self,
               sess,
               num_actions,
               quantile_value_network=atari_lib.FullyParameterizedQuantileNetwork,
               fraction_proposal_network=atari_lib.FractionProposalNetork,
               kappa=1.0,
               num_quantile_samples=32,
               quantile_embedding_dim=64,
               entropy_coefficient=0.0,
               double_dqn=False,
               quantile_value_optimizer=tf.train.AdamOptimizer(
                 learning_rate=0.00025, epsilon=0.0003125),
               fraction_proposal_optimizer=tf.train.RMSPropOptimizer(
                 learning_rate=2.5e-9, decay=0.95, epsilon=0.00001,
                 centered=True),
               summary_writer=None,
               summary_writing_frequency=500):
    """Initializes the agent and constructs the Graph.

    Most of this constructor's parameters are IQN-specific hyperparameters whose
    values are taken from Dabney et al. (2018).

    Args:
      sess: `tf.Session` object for running associated ops.
      num_actions: int, number of actions the agent can take at any state.
      network: tf.Keras.Model, expects three parameters:
        (num_actions, quantile_embedding_dim, network_type). This class is used
        to generate network instances that are used by the agent. Each
        instantiation would have different set of variables. See
        dopamine.discrete_domains.atari_lib.NatureDQNNetwork as an example.
      kappa: float, Huber loss cutoff.
      num_quantile_samples: int, number of online quantile samples for loss
        estimation.
      num_quantile_samples: int, number of target quantile samples for loss
        estimation.
      num_quantile_samples: int, number of quantile samples for computing
        Q-values.
      quantile_embedding_dim: int, embedding dimension for the quantile input.
      double_dqn: boolean, whether to perform double DQN style learning
        as described in Van Hasselt et al.: https://arxiv.org/abs/1509.06461.
      summary_writer: SummaryWriter object for outputting training statistics.
        Summary writing disabled if set to None.
      summary_writing_frequency: int, frequency with which summaries will be
        written. Lower values will result in slower training.
    """
    self.quantile_value_network = quantile_value_network
    self.fraction_proposal_network = fraction_proposal_network
    self.quantile_value_optimizer = quantile_value_optimizer
    self.fraction_proposal_optimizer = fraction_proposal_optimizer
    self.kappa = kappa
    self.entropy_coefficient = entropy_coefficient
    # num_quantile_samples = k below equation (3) in the paper.
    self.num_quantile_samples = num_quantile_samples
    # quantile_embedding_dim = n above equation (4) in the paper.
    self.quantile_embedding_dim = quantile_embedding_dim
    # option to perform double dqn.
    self.double_dqn = double_dqn
    
    super(FullyParameterizedQuantileAgent, self).__init__(
      sess=sess,
      num_actions=num_actions,
      summary_writer=summary_writer,
      summary_writing_frequency=summary_writing_frequency)
  
  def _create_network(self, name):
    r"""Builds an Implicit Quantile ConvNet.

    Args:
      name: str, this name is passed to the tf.keras.Model and used to create
        variable scope under the hood by the tf.keras.Model.
    Returns:
      network: tf.keras.Model, the network instantiated by the Keras model.
    """
    network = self.quantile_value_network(self.num_actions,
                                          self.quantile_embedding_dim,
                                          self.num_quantile_samples, name=name)
    return network
  
  def _build_networks(self):
    """Builds the IQN computations needed for acting and training.

    These are:
      self.online_convnet: For computing the current state's quantile values.
      self.target_convnet: For computing the next state's target quantile
        values.
      self._net_outputs: The actual quantile values.
      self._q_argmax: The action maximizing the current state's Q-values.
      self._replay_net_outputs: The replayed states' quantile values.
      self._replay_next_target_net_outputs: The replayed next states' target
        quantile values.
    """
    self.tau_mlpnet = self.fraction_proposal_network(
      self.num_quantile_samples, name="FPN")
    self.online_convnet = self._create_network(name='Online')
    self.target_convnet = self._create_network(name='Target')
    
    # Compute the Q-values which are used for action selection in the current
    # state.
    self._net_outputs = self.online_convnet(
      self.state_ph, fraction_proposal_net=self.tau_mlpnet)
    
    # Shape of self._net_outputs.quantile_values:
    # num_quantile_samples x num_actions.
    # Shape of self._net_outputs.quantiles:
    # num_quantile_samples x 1.
    # Shape of self._net_outputs.fraction_proposal.taus:
    # batch_size x num_quantile_samples
    # 静态图构建时batch_size=1
    fraction_proposal = self._net_outputs.fraction_proposal
    self._net_delta_taus = tf.transpose(fraction_proposal.delta_taus)
    self._net_quantile_values = self._net_outputs.quantile_values
    self._q_values = tf.reduce_sum(tf.multiply(self._net_delta_taus,
                                               self._net_quantile_values),
                                   axis=0)
    self._q_argmax = tf.argmax(self._q_values, axis=0)
    
    self._replay_net_outputs = self.online_convnet(
      self._replay.states, fraction_proposal_net=self.tau_mlpnet)
    # Shape: (num_quantile_samples x batch_size) x num_actions.
    self._replay_net_quantile_values = self._replay_net_outputs.quantile_values
    # Shape: (num_quantile_samples x batch_size) x 1
    self._replay_net_quantiles = self._replay_net_outputs.quantiles
    # Shape: [batch_size x (num_quantile_samples-1), 1]
    vals = self._replay_net_outputs.fraction_proposal.taus[:, :-1]
    self._replay_net_taus = vals
    
    # Do the same for next states in the replay buffergather.
    # 共享quantiles
    self._replay_net_target_outputs = self.target_convnet(
      self._replay.next_states, proposed_quantiles=self._replay_net_quantiles)
    # Shape: (num_quantile_samples x batch_size) x num_actions.
    vals = self._replay_net_target_outputs.quantile_values
    self._replay_net_target_quantile_values = vals
    
    # Compute Q-values which are used for action selection for the next states
    # in the replay buffer. Compute the argmax over the Q-values.
    if self.double_dqn:
      outputs_action = self.online_convnet(
        self._replay.next_states, fraction_proposal_net=self.tau_mlpnet)
    else:
      outputs_action = self.target_convnet(
        self._replay.next_states, fraction_proposal_net=self.tau_mlpnet)
    
    # Shape: (num_quantile_samples x batch_size) x num_actions.
    target_quantile_values_action = outputs_action.quantile_values
    # Shape: num_quantile_samples x batch_size x num_actions.
    target_quantile_values_action = tf.reshape(target_quantile_values_action,
                                               [self.num_quantile_samples,
                                                self._replay.batch_size,
                                                self.num_actions])
    # Shape: batch_size x num_quantile_samples.
    target_delta_taus_action = outputs_action.fraction_proposal.delta_taus
    target_delta_taus_action = tf.transpose(target_delta_taus_action)[..., None]
    # 这里需要对Target Quantile_values做加权计算Q值，权重就是delta_taus
    # Shape: batch_size x num_actions.
    self._replay_net_target_q_values = tf.squeeze(tf.reduce_sum(
      tf.multiply(target_quantile_values_action, target_delta_taus_action),
      axis=0))
    self._replay_next_qt_argmax = tf.argmax(
      self._replay_net_target_q_values, axis=1)
    
    # 计算F_Z^{-1}(\tau)
    reshaped_replay_net_taus = tf.reshape(
      self._replay_net_taus,
      [(self.num_quantile_samples - 1) * self._replay.batch_size, 1])
    replay_tau_outputs = self.online_convnet(
      self._replay.states, proposed_quantiles=reshaped_replay_net_taus)
    # Shape: (num_quantile_samples-1 x batch_size) x num_actions.
    self._replay_net_tau_values = tf.stop_gradient(
      replay_tau_outputs.quantile_values)
  
  def _build_target_quantile_values_op(self):
    """Build an op used as a target for return values at given quantiles.

    Returns:
      An op calculating the target quantile return.
    """
    batch_size = tf.shape(self._replay.rewards)[0]
    # Shape of rewards: (num_quantile_samples x batch_size) x 1.
    rewards = self._replay.rewards[:, None]
    rewards = tf.tile(rewards, [self.num_quantile_samples, 1])
    
    is_terminal_multiplier = 1. - tf.to_float(self._replay.terminals)
    # Incorporate terminal state to discount factor.
    # size of gamma_with_terminal: (num_quantile_samples x batch_size) x 1.
    gamma_with_terminal = self.cumulative_gamma * is_terminal_multiplier
    gamma_with_terminal = tf.tile(gamma_with_terminal[:, None],
                                  [self.num_quantile_samples, 1])
    
    # Get the indices of the maximium Q-value across the action dimension.
    # Shape of replay_next_qt_argmax: (num_quantile_samples x batch_size) x 1.
    
    replay_next_qt_argmax = tf.tile(
      self._replay_next_qt_argmax[:, None], [self.num_quantile_samples, 1])
    
    # Shape of batch_indices: (num_quantile_samples x batch_size) x 1.
    batch_indices = tf.cast(tf.range(
      self.num_quantile_samples * batch_size)[:, None], tf.int64)
    
    # Shape of batch_indexed_target_values:
    # (num_quantile_samples x batch_size) x 2.
    batch_indexed_target_values = tf.concat(
      [batch_indices, replay_next_qt_argmax], axis=1)
    
    # Shape of next_target_values: (num_quantile_samples x batch_size) x 1.
    target_quantile_values = tf.gather_nd(
      self._replay_net_target_quantile_values,
      batch_indexed_target_values)[:, None]
    
    return rewards + gamma_with_terminal * target_quantile_values
  
  def _build_train_op(self):
    """Builds a training op.

    Returns:
      train_op: An op performing one step of training from replay data.
    """
    batch_size = tf.shape(self._replay.rewards)[0]
    
    target_quantile_values = tf.stop_gradient(
      self._build_target_quantile_values_op())
    # Reshape to self.num_quantile_samples x batch_size x 1 since this is
    # the manner in which the target_quantile_values are tiled.
    target_quantile_values = tf.reshape(target_quantile_values,
                                        [self.num_quantile_samples,
                                         batch_size, 1])
    # Transpose dimensions so that the dimensionality is batch_size x
    # self.num_quantile_samples x 1 to prepare for computation of
    # Bellman errors.
    # Final shape of target_quantile_values:
    # batch_size x num_quantile_samples x 1.
    target_quantile_values = tf.transpose(target_quantile_values, [1, 0, 2])
    
    # Shape of indices: (num_quantile_samples x batch_size) x 1.
    # Expand dimension by one so that it can be used to index into all the
    # quantiles when using the tf.gather_nd function (see below).
    indices = tf.range(self.num_quantile_samples * batch_size)[:, None]
    
    # Expand the dimension by one so that it can be used to index into all the
    # quantiles when using the tf.gather_nd function (see below).
    reshaped_actions = self._replay.actions[:, None]
    reshaped_actions = tf.tile(reshaped_actions, [self.num_quantile_samples, 1])
    # Shape of reshaped_actions: (num_quantile_samples x batch_size) x 2.
    reshaped_actions = tf.concat([indices, reshaped_actions], axis=1)
    
    chosen_action_quantile_values = tf.gather_nd(
      self._replay_net_quantile_values, reshaped_actions)
    # Reshape to self.num_quantile_samples x batch_size x 1 since this is the
    # manner in which the quantile values are tiled.
    chosen_action_quantile_values = tf.reshape(chosen_action_quantile_values,
                                               [self.num_quantile_samples,
                                                batch_size, 1])
    # Transpose dimensions so that the dimensionality is batch_size x
    # self.num_quantile_samples x 1 to prepare for computation of
    # Bellman errors.
    # Final shape of chosen_action_quantile_values:
    # batch_size x num_quantile_samples x 1.
    chosen_action_quantile_values = tf.transpose(
      chosen_action_quantile_values, [1, 0, 2])
    
    # Shape of bellman_erors and huber_loss:
    # batch_size x num_quantile_samples x num_quantile_samples x 1.
    val1 = target_quantile_values[:, :, None, :]
    val2 = chosen_action_quantile_values[:, None, :, :]
    bellman_errors = val1 - val2
    # The huber loss (see Section 2.3 of the paper) is defined via two cases:
    # case_one: |bellman_errors| <= kappa
    # case_two: |bellman_errors| > kappa
    huber_loss_case_one = tf.to_float(
      tf.abs(bellman_errors) <= self.kappa) * 0.5 * bellman_errors ** 2
    huber_loss_case_two = tf.to_float(
      tf.abs(bellman_errors) > self.kappa) * self.kappa * (
                              tf.abs(bellman_errors) - 0.5 * self.kappa)
    huber_loss = huber_loss_case_one + huber_loss_case_two
    
    # Reshape replay_quantiles to batch_size x num_quantile_samples x 1
    replay_quantiles = tf.reshape(
      self._replay_net_quantiles, [self.num_quantile_samples, batch_size, 1])
    replay_quantiles = tf.transpose(replay_quantiles, [1, 0, 2])
    
    # Tile by num_quantile_samples along a new dimension. Shape is now
    # batch_size x num_quantile_samples x num_quantile_samples x 1.
    # These quantiles will be used for computation of the quantile huber loss
    # below (see section 2.3 of the paper).
    replay_quantiles = tf.to_float(tf.tile(
      replay_quantiles[:, None, :, :], [1, self.num_quantile_samples, 1, 1]))
    # Shape: batch_size x num_quantile_samples x num_quantile_samples x 1.
    quantile_huber_loss = (tf.abs(replay_quantiles - tf.stop_gradient(
      tf.to_float(bellman_errors < 0))) * huber_loss) / self.kappa
    # Sum over current quantile value (num_quantile_samples) dimension,
    # average over target quantile value (num_quantile_samples) dimension.
    # Shape: batch_size x num_quantile_samples x 1.
    iqn_loss = tf.reduce_sum(quantile_huber_loss, axis=2)
    # Shape: batch_size x 1.
    iqn_loss = tf.reduce_mean(iqn_loss, axis=1)
    
    scope = tf.get_default_graph().get_name_scope()
    trainables_fpn = tf.get_collection(
      tf.GraphKeys.TRAINABLE_VARIABLES, scope=os.path.join(scope, 'Online/FPN'))
    trainables_online = tf.get_collection(
      tf.GraphKeys.TRAINABLE_VARIABLES, scope=os.path.join(scope, 'Online'))
    for w_fpn in trainables_fpn:
      trainables_online.remove(w_fpn)
    # 这里和上面选(x,a)对应的quantiles_value值的流程是一样的，区别在于维度少了1
    indices = tf.range((self.num_quantile_samples - 1) * batch_size)[:, None]
    reshaped_actions = self._replay.actions[:, None]
    reshaped_actions = tf.tile(reshaped_actions,
                               [self.num_quantile_samples - 1, 1])
    reshaped_actions = tf.concat([indices, reshaped_actions], axis=1)
    chosen_action_tau_values = tf.gather_nd(
      self._replay_net_tau_values, reshaped_actions)
    chosen_action_tau_values = tf.reshape(chosen_action_tau_values,
                                          [self.num_quantile_samples - 1,
                                           batch_size, 1])
    chosen_action_tau_values = tf.transpose(chosen_action_tau_values, [1, 0, 2])
    
    sub1 = chosen_action_tau_values - chosen_action_quantile_values[:, :-1, :]
    sub2 = chosen_action_tau_values - chosen_action_quantile_values[:, 1:, :]
    W1_partial_tau = tf.squeeze(sub1 + sub2)
    fraction_loss = tf.reduce_sum(self._replay_net_taus * W1_partial_tau,
                                  axis=1)
    fraction_entropies = self._replay_net_outputs.fraction_proposal.entropies
    
    # 加入PER
    if self._replay_scheme == 'prioritized':
      probs = self._replay.transition['sampling_probabilities']
      loss_weights = 1.0 / tf.sqrt(probs + 1e-10)
      loss_weights /= tf.reduce_max(loss_weights)
      update_priorities_op = self._replay.tf_set_priority(
        self._replay.indices, tf.sqrt(iqn_loss + 1e-10))
      iqn_loss = loss_weights * iqn_loss
      fraction_loss = loss_weights * fraction_loss
    else:
      update_priorities_op = tf.no_op()
    
    w1_loss = tf.reduce_sum(fraction_loss)
    if self.entropy_coefficient > 0:
      sum_fraction_entropies = tf.reduce_sum(fraction_entropies)
      w1_loss = w1_loss - self.entropy_coefficient * sum_fraction_entropies
    w2_loss = tf.reduce_mean(iqn_loss)
    update_w1_op = self.fraction_proposal_optimizer.minimize(
      loss=w1_loss, var_list=trainables_fpn)
    update_w2_op = self.quantile_value_optimizer.minimize(
      loss=w2_loss, var_list=trainables_online)
    
    with tf.control_dependencies([update_priorities_op]):
      if self.summary_writer is not None:
        with tf.variable_scope('Losses'):
          tf.summary.scalar('QuantileLoss', tf.reduce_mean(iqn_loss))
          tf.summary.scalar('FractionLoss', tf.reduce_sum(fraction_loss))
          tf.summary.scalar('FractionEntropy',
                            tf.reduce_sum(fraction_entropies))
      return update_w1_op, update_w2_op
