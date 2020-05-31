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
"""Atari-specific utilities including Atari-specific network architectures.

This includes a class implementing minimal Atari 2600 preprocessing, which
is in charge of:
  . Emitting a terminal signal when losing a life (optional).
  . Frame skipping and color pooling.
  . Resizing the image before it is provided to the agent.
## Networks
We are subclassing keras.models.Model in our network definitions. Each network
class has two main functions: `.__init__` and `.call`. When we create our
network the `__init__` function is called and necessary layers are defined. Once
we create our network, we can create the output operations by doing `call`s to
our network with different inputs. At each call, the same parameters will be
used.
More information about keras.Model API can be found here:
https://www.tensorflow.org/api_docs/python/tf/keras/models/Model
## Network Types
Network types are namedtuples that define the output signature of the networks
used. Please use the appropriate signature as needed when defining new networks.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math

import gin
import gym
from gym.spaces.box import Box
import numpy as np
import tensorflow as tf

import cv2

NATURE_DQN_OBSERVATION_SHAPE = (84, 84)  # Size of downscaled Atari 2600 frame.
NATURE_DQN_DTYPE = tf.uint8  # DType of Atari 2600 observations.
NATURE_DQN_STACK_SIZE = 4  # Number of frames in the state stack.

DQNNetworkType = collections.namedtuple('dqn_network', ['q_values'])
RainbowNetworkType = collections.namedtuple(
  'c51_network', ['q_values', 'logits', 'probabilities'])
ImplicitQuantileNetworkType = collections.namedtuple(
  'iqn_network', ['quantile_values', 'quantiles'])
FullyParameterizedQuantileNetworkType = collections.namedtuple(
  'fqf_network', ['quantile_values', 'quantiles', 'fraction_proposal'])
FractionProposalNetworkType = collections.namedtuple(
  'fp_network', ['delta_taus', 'taus', 'entropies'])


@gin.configurable
def create_atari_environment(game_name=None, sticky_actions=False):
  """Wraps an Atari 2600 Gym environment with some basic preprocessing.
  This preprocessing matches the guidelines proposed in Machado et al. (2017),
  "Revisiting the Arcade Learning Environment: Evaluation Protocols and Open
  Problems for General Agents".
  The created environment is the Gym wrapper around the Arcade Learning
  Environment.
  The main choice available to the user is whether to use sticky actions or not.
  Sticky actions, as prescribed by Machado et al., cause actions to persist
  with some probability (0.25) when a new command is sent to the ALE. This
  can be viewed as introducing a mild form of stochasticity in the environment.
  We use them by default.
  Args:
    game_name: str, the name of the Atari 2600 domain.
    sticky_actions: bool, whether to use sticky_actions as per Machado et al.
  Returns:
    An Atari 2600 environment with some standard preprocessing.
  """
  assert game_name is not None
  game_version = 'v0' if sticky_actions else 'v4'
  full_game_name = '{}NoFrameskip-{}'.format(game_name, game_version)
  env = gym.make(full_game_name)
  # Strip out the TimeLimit wrapper from Gym, which caps us at 100k frames. We
  # handle this time limit internally instead, which lets us cap at 108k frames
  # (30 minutes). The TimeLimit wrapper also plays poorly with saving and
  # restoring states.
  env = env.env
  env = AtariPreprocessing(env)
  return env


@gin.configurable(blacklist=['variables'])
def maybe_transform_variable_names(variables, legacy_checkpoint_load=False):
  """Maps old variable names to the new ones.
  The resulting dictionary can be passed to the tf.train.Saver to load
  legacy checkpoints into Keras models.
  Args:
    variables: list, of all variables to be transformed.
    legacy_checkpoint_load: bool, if True the variable names are mapped to
        the legacy names as appeared in `tf.slim` based agents. Use this if
        you want to load checkpoints saved before tf.keras.Model upgrade.
  Returns:
    dict or None, of <new_names, var>.
  """
  tf.logging.info('legacy_checkpoint_load: %s', legacy_checkpoint_load)
  if legacy_checkpoint_load:
    name_map = {}
    for var in variables:
      new_name = var.op.name.replace('bias', 'biases')
      new_name = new_name.replace('kernel', 'weights')
      name_map[new_name] = var
  else:
    name_map = None
  return name_map


class NatureDQNNetwork(tf.keras.Model):
  """The convolutional network used to compute the agent's Q-values."""
  
  def __init__(self, num_actions, name=None):
    """Creates the layers used for calculating Q-values.
    Args:
      num_actions: int, number of actions.
      name: str, used to create scope for network parameters.
    """
    super(NatureDQNNetwork, self).__init__(name=name)
    
    self.num_actions = num_actions
    # Defining layers.
    activation_fn = tf.keras.activations.relu
    # Setting names of the layers manually to make variable names more similar
    # with tf.slim variable names/checkpoints.
    self.conv1 = tf.keras.layers.Conv2D(32, [8, 8], strides=4, padding='same',
                                        activation=activation_fn, name='Conv')
    self.conv2 = tf.keras.layers.Conv2D(64, [4, 4], strides=2, padding='same',
                                        activation=activation_fn, name='Conv')
    self.conv3 = tf.keras.layers.Conv2D(64, [3, 3], strides=1, padding='same',
                                        activation=activation_fn, name='Conv')
    self.flatten = tf.keras.layers.Flatten()
    self.dense1 = tf.keras.layers.Dense(512, activation=activation_fn,
                                        name='fully_connected')
    self.dense2 = tf.keras.layers.Dense(num_actions, name='fully_connected')
  
  def call(self, state):
    """Creates the output tensor/op given the state tensor as input.
    See https://www.tensorflow.org/api_docs/python/tf/keras/Model for more
    information on this. Note that tf.keras.Model implements `call` which is
    wrapped by `__call__` function by tf.keras.Model.
    Parameters created here will have scope according to the `name` argument
    given at `.__init__()` call.
    Args:
      state: Tensor, input tensor.
    Returns:
      collections.namedtuple, output ops (graph mode) or output tensors (eager).
    """
    x = tf.cast(state, tf.float32)
    x = tf.div(x, 255.)
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.conv3(x)
    x = self.flatten(x)
    x = self.dense1(x)
    
    return DQNNetworkType(self.dense2(x))


class RainbowNetwork(tf.keras.Model):
  """The convolutional network used to compute agent's return distributions."""
  
  def __init__(self, num_actions, num_atoms, support, name=None):
    """Creates the layers used calculating return distributions.
    Args:
      num_actions: int, number of actions.
      num_atoms: int, the number of buckets of the value function distribution.
      support: tf.linspace, the support of the Q-value distribution.
      name: str, used to crete scope for network parameters.
    """
    super(RainbowNetwork, self).__init__(name=name)
    activation_fn = tf.keras.activations.relu
    self.num_actions = num_actions
    self.num_atoms = num_atoms
    self.support = support
    self.kernel_initializer = tf.keras.initializers.VarianceScaling(
      scale=1.0 / np.sqrt(3.0), mode='fan_in', distribution='uniform')
    # Defining layers.
    self.conv1 = tf.keras.layers.Conv2D(
      32, [8, 8], strides=4, padding='same', activation=activation_fn,
      kernel_initializer=self.kernel_initializer, name='Conv')
    self.conv2 = tf.keras.layers.Conv2D(
      64, [4, 4], strides=2, padding='same', activation=activation_fn,
      kernel_initializer=self.kernel_initializer, name='Conv')
    self.conv3 = tf.keras.layers.Conv2D(
      64, [3, 3], strides=1, padding='same', activation=activation_fn,
      kernel_initializer=self.kernel_initializer, name='Conv')
    self.flatten = tf.keras.layers.Flatten()
    self.dense1 = tf.keras.layers.Dense(
      512, activation=activation_fn,
      kernel_initializer=self.kernel_initializer, name='fully_connected')
    self.dense2 = tf.keras.layers.Dense(
      num_actions * num_atoms, kernel_initializer=self.kernel_initializer,
      name='fully_connected')
  
  def call(self, state):
    """Creates the output tensor/op given the state tensor as input.
    See https://www.tensorflow.org/api_docs/python/tf/keras/Model for more
    information on this. Note that tf.keras.Model implements `call` which is
    wrapped by `__call__` function by tf.keras.Model.
    Args:
      state: Tensor, input tensor.
    Returns:
      collections.namedtuple, output ops (graph mode) or output tensors (eager).
    """
    x = tf.cast(state, tf.float32)
    x = tf.div(x, 255.)
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.conv3(x)
    x = self.flatten(x)
    x = self.dense1(x)
    x = self.dense2(x)
    logits = tf.reshape(x, [-1, self.num_actions, self.num_atoms])
    probabilities = tf.keras.activations.softmax(logits)
    q_values = tf.reduce_sum(self.support * probabilities, axis=2)
    return RainbowNetworkType(q_values, logits, probabilities)


class ImplicitQuantileNetwork(tf.keras.Model):
  """The Implicit Quantile Network (Dabney et al., 2018).."""
  
  def __init__(self, num_actions, quantile_embedding_dim, name=None):
    """Creates the layers used calculating quantile values.
    Args:
      num_actions: int, number of actions.
      quantile_embedding_dim: int, embedding dimension for the quantile input.
      name: str, used to create scope for network parameters.
    """
    super(ImplicitQuantileNetwork, self).__init__(name=name)
    self.num_actions = num_actions
    self.quantile_embedding_dim = quantile_embedding_dim
    # We need the activation function during `call`, therefore set the field.
    self.activation_fn = tf.keras.activations.relu
    self.kernel_initializer = tf.keras.initializers.VarianceScaling(
      scale=1.0 / np.sqrt(3.0), mode='fan_in', distribution='uniform')
    # Defining layers.
    self.conv1 = tf.keras.layers.Conv2D(
      32, [8, 8], strides=4, padding='same', activation=self.activation_fn,
      kernel_initializer=self.kernel_initializer, name='Conv')
    self.conv2 = tf.keras.layers.Conv2D(
      64, [4, 4], strides=2, padding='same', activation=self.activation_fn,
      kernel_initializer=self.kernel_initializer, name='Conv')
    self.conv3 = tf.keras.layers.Conv2D(
      64, [3, 3], strides=1, padding='same', activation=self.activation_fn,
      kernel_initializer=self.kernel_initializer, name='Conv')
    self.flatten = tf.keras.layers.Flatten()
    self.dense1 = tf.keras.layers.Dense(
      512, activation=self.activation_fn,
      kernel_initializer=self.kernel_initializer, name='fully_connected')
    self.dense2 = tf.keras.layers.Dense(
      num_actions, kernel_initializer=self.kernel_initializer,
      name='fully_connected')
  
  def call(self, state, num_quantiles):
    """Creates the output tensor/op given the state tensor as input.
    See https://www.tensorflow.org/api_docs/python/tf/keras/Model for more
    information on this. Note that tf.keras.Model implements `call` which is
    wrapped by `__call__` function by tf.keras.Model.
    Args:
      state: `tf.Tensor`, contains the agent's current state.
      num_quantiles: int, number of quantile inputs.
    Returns:
      collections.namedtuple, that contains (quantile_values, quantiles).
    """
    batch_size = state.get_shape().as_list()[0]
    x = tf.cast(state, tf.float32)
    x = tf.div(x, 255.)
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.conv3(x)
    x = self.flatten(x)
    state_vector_length = x.get_shape().as_list()[-1]
    state_net_tiled = tf.tile(x, [num_quantiles, 1])
    quantiles_shape = [num_quantiles * batch_size, 1]
    quantiles = tf.random_uniform(
      quantiles_shape, minval=0, maxval=1, dtype=tf.float32)
    quantile_net = tf.tile(quantiles, [1, self.quantile_embedding_dim])
    pi = tf.constant(math.pi)
    quantile_net = tf.cast(tf.range(
      1, self.quantile_embedding_dim + 1, 1), tf.float32) * pi * quantile_net
    quantile_net = tf.cos(quantile_net)
    # Create the quantile layer in the first call. This is because
    # number of output units depends on the input shape. Therefore, we can only
    # create the layer during the first forward call, not during `.__init__()`.
    if not hasattr(self, 'dense_quantile'):
      self.dense_quantile = tf.keras.layers.Dense(
        state_vector_length, activation=self.activation_fn,
        kernel_initializer=self.kernel_initializer)
    quantile_net = self.dense_quantile(quantile_net)
    x = tf.multiply(state_net_tiled, quantile_net)
    x = self.dense1(x)
    quantile_values = self.dense2(x)
    return ImplicitQuantileNetworkType(quantile_values, quantiles)


@gin.configurable
class AtariPreprocessing(object):
  """A class implementing image preprocessing for Atari 2600 agents.
  Specifically, this provides the following subset from the JAIR paper
  (Bellemare et al., 2013) and Nature DQN paper (Mnih et al., 2015):
    * Frame skipping (defaults to 4).
    * Terminal signal when a life is lost (off by default).
    * Grayscale and max-pooling of the last two frames.
    * Downsample the screen to a square image (defaults to 84x84).
  More generally, this class follows the preprocessing guidelines set down in
  Machado et al. (2018), "Revisiting the Arcade Learning Environment:
  Evaluation Protocols and Open Problems for General Agents".
  It also provides random starting no-ops, which are used in the Rainbow, Apex
  and R2D2 papers.
  """
  
  def __init__(self, environment, frame_skip=4, terminal_on_life_loss=True,
               screen_size=84, max_random_noops=30):
    """Constructor for an Atari 2600 preprocessor.
    Args:
      environment: Gym environment whose observations are preprocessed.
      frame_skip: int, the frequency at which the agent experiences the game.
      terminal_on_life_loss: bool, If True, the step() method returns
        is_terminal=True whenever a life is lost. See Mnih et al. 2015.
      screen_size: int, size of a resized Atari 2600 frame.
      max_random_noops: int, maximum number of no-ops to apply at the beginning
        of each episode to reduce determinism. These no-ops are applied at a
        low-level, before frame skipping.
    Raises:
      ValueError: if frame_skip or screen_size are not strictly positive.
    """
    if frame_skip <= 0:
      raise ValueError('Frame skip should be strictly positive, got {}'.
                       format(frame_skip))
    if screen_size <= 0:
      raise ValueError('Target screen size should be strictly positive, got {}'.
                       format(screen_size))
    
    self.environment = environment
    self.terminal_on_life_loss = terminal_on_life_loss
    self.frame_skip = frame_skip
    self.screen_size = screen_size
    self.max_random_noops = max_random_noops
    
    obs_dims = self.environment.observation_space
    # Stores temporary observations used for pooling over two successive
    # frames.
    self.screen_buffer = [
      np.empty((obs_dims.shape[0], obs_dims.shape[1]), dtype=np.uint8),
      np.empty((obs_dims.shape[0], obs_dims.shape[1]), dtype=np.uint8)
    ]
    
    self.game_over = False
    self.lives = 0  # Will need to be set by reset().
  
  @property
  def observation_space(self):
    # Return the observation space adjusted to match the shape of the processed
    # observations.
    return Box(low=0, high=255, shape=(self.screen_size, self.screen_size, 1),
               dtype=np.uint8)
  
  @property
  def action_space(self):
    return self.environment.action_space
  
  @property
  def reward_range(self):
    return self.environment.reward_range
  
  @property
  def metadata(self):
    return self.environment.metadata
  
  def close(self):
    return self.environment.close()
  
  def apply_random_noops(self):
    """Steps self.environment with random no-ops."""
    if self.max_random_noops <= 0:
      return
    # Other no-ops implementations actually always do at least 1 no-op. We
    # follow them.
    no_ops = self.environment.np_random.randint(1, self.max_random_noops + 1)
    for _ in range(no_ops):
      _, _, game_over, _ = self.environment.step(0)
      if game_over:
        self.environment.reset()
  
  def apply_fire_reset_op(self):
    _, _, game_over, _ = self.environment.step(1)
    if game_over:
      self.environment.reset()
    _, _, game_over, _ = self.environment.step(2)
    if game_over:
      self.environment.reset()
  
  def reset(self):
    """Resets the environment.
    Returns:
      observation: numpy array, the initial observation emitted by the
        environment.
    """
    self.environment.reset()
    self.apply_random_noops()
    self.apply_fire_reset_op()
    
    self.lives = self.environment.ale.lives()
    self._fetch_grayscale_observation(self.screen_buffer[0])
    self.screen_buffer[1].fill(0)
    return self._pool_and_resize()
  
  def render(self, mode):
    """Renders the current screen, before preprocessing.
    This calls the Gym API's render() method.
    Args:
      mode: Mode argument for the environment's render() method.
        Valid values (str) are:
          'rgb_array': returns the raw ALE image.
          'human': renders to display via the Gym renderer.
    Returns:
      if mode='rgb_array': numpy array, the most recent screen.
      if mode='human': bool, whether the rendering was successful.
    """
    return self.environment.render(mode)
  
  def step(self, action):
    """Applies the given action in the environment.
    Remarks:
      * If a terminal state (from life loss or episode end) is reached, this may
        execute fewer than self.frame_skip steps in the environment.
      * Furthermore, in this case the returned observation may not contain valid
        image data and should be ignored.
    Args:
      action: The action to be executed.
    Returns:
      observation: numpy array, the observation following the action.
      reward: float, the reward following the action.
      is_terminal: bool, whether the environment has reached a terminal state.
        This is true when a life is lost and terminal_on_life_loss, or when the
        episode is over.
      info: Gym API's info data structure.
    """
    accumulated_reward = 0.
    
    for time_step in range(self.frame_skip):
      # We bypass the Gym observation altogether and directly fetch the
      # grayscale image from the ALE. This is a little faster.
      _, reward, game_over, info = self.environment.step(action)
      accumulated_reward += reward
      
      # 更新命数
      # for Qbert sometimes we stay in lives == 0 condtion for a few
      # frames so its important to keep lives > 0, so that we only reset
      # once the environment advertises done.
      new_lives = self.environment.ale.lives()
      life_loss = 0 < new_lives < self.lives
      self.lives = new_lives
      # 如果掉命就进行一次NOOP+FIRE操作获取新命
      if life_loss:
        self.environment.step(0)
        self.apply_fire_reset_op()
      
      if self.terminal_on_life_loss:
        is_terminal = game_over or life_loss
      
      else:
        is_terminal = game_over
      
      if is_terminal:
        break
      # We max-pool over the last two frames, in grayscale.
      elif time_step >= self.frame_skip - 2:
        t = time_step - (self.frame_skip - 2)
        self._fetch_grayscale_observation(self.screen_buffer[t])
    
    # Pool the last two observations.
    observation = self._pool_and_resize()
    
    self.game_over = game_over
    return observation, accumulated_reward, is_terminal, info
  
  def _fetch_grayscale_observation(self, output):
    """Returns the current observation in grayscale.
    The returned observation is stored in 'output'.
    Args:
      output: numpy array, screen buffer to hold the returned observation.
    Returns:
      observation: numpy array, the current observation in grayscale.
    """
    self.environment.ale.getScreenGrayscale(output)
    return output
  
  def _pool_and_resize(self):
    """Transforms two frames into a Nature DQN observation.
    For efficiency, the transformation is done in-place in self.screen_buffer.
    Returns:
      transformed_screen: numpy array, pooled, resized screen.
    """
    # Pool if there are enough screens to do so.
    if self.frame_skip > 1:
      np.maximum(self.screen_buffer[0], self.screen_buffer[1],
                 out=self.screen_buffer[0])
    
    transformed_image = cv2.resize(self.screen_buffer[0],
                                   (self.screen_size, self.screen_size),
                                   interpolation=cv2.INTER_LINEAR)
    int_image = np.asarray(transformed_image, dtype=np.uint8)
    return np.expand_dims(int_image, axis=2)


class FullyParameterizedQuantileNetwork(tf.keras.Model):
  
  def __init__(self, num_actions, quantile_embedding_dim, num_quantiles,
               name=None):
    """Creates the layers used calculating quantile values.
    Args:
      num_actions: int, number of actions.
      quantile_embedding_dim: int, embedding dimension for the quantile input.
      name: str, used to create scope for network parameters.
    """
    super(FullyParameterizedQuantileNetwork, self).__init__(name=name)
    self.num_actions = num_actions
    self.quantile_embedding_dim = quantile_embedding_dim
    self.default_num_quantiles = num_quantiles
    # We need the activation function during `call`, therefore set the field.
    self.activation_fn = tf.keras.activations.relu
    self.kernel_initializer = tf.keras.initializers.VarianceScaling(
      scale=1.0 / np.sqrt(3.0), mode='fan_in', distribution='uniform')
    # Defining layers.
    self.conv1 = tf.keras.layers.Conv2D(
      32, [8, 8], strides=4, padding='same', activation=self.activation_fn,
      kernel_initializer=self.kernel_initializer, name='Conv')
    self.conv2 = tf.keras.layers.Conv2D(
      64, [4, 4], strides=2, padding='same', activation=self.activation_fn,
      kernel_initializer=self.kernel_initializer, name='Conv')
    self.conv3 = tf.keras.layers.Conv2D(
      64, [3, 3], strides=1, padding='same', activation=self.activation_fn,
      kernel_initializer=self.kernel_initializer, name='Conv')
    self.flatten = tf.keras.layers.Flatten()
    self.dense1 = tf.keras.layers.Dense(
      512, activation=self.activation_fn,
      kernel_initializer=self.kernel_initializer, name='fully_connected')
    self.dense2 = tf.keras.layers.Dense(
      num_actions, kernel_initializer=self.kernel_initializer,
      name='fully_connected')
  
  def call(self, state, proposed_quantiles=None, fraction_proposal_net=None):
    # 确定quantiles的数量，如果是使用指定的quantiles就更改长度
    num_quantiles = self.default_num_quantiles
    batch_size = state.get_shape().as_list()[0]
    
    if proposed_quantiles is not None:
      # shape of proposed quantiles: [num_quantiles * batch_size, 1]
      num_quantiles = proposed_quantiles.get_shape().as_list()[0] // batch_size
    
    x = tf.cast(state, tf.float32)
    x = tf.div(x, 255.)
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.conv3(x)
    x = self.flatten(x)
    state_vector_length = x.get_shape().as_list()[-1]
    state_net_tiled = tf.tile(x, [num_quantiles, 1])
    quantiles_shape = [num_quantiles * batch_size, 1]
    if proposed_quantiles is not None:
      quantiles = tf.stop_gradient(proposed_quantiles)
    else:
      # shape of taus: batch_size x num_quantiles+1
      fraction_proposal = fraction_proposal_net(x)
      # 防止梯度BP到FPN网络
      taus_1_to_N = fraction_proposal.taus
      taus_0 = tf.zeros([batch_size, 1])
      taus_0_to_N = tf.concat([taus_0, taus_1_to_N], axis=1)
      quantiles = (taus_0_to_N[:, :-1] + taus_0_to_N[:, 1:]) / 2.0
      quantiles = tf.transpose(tf.stop_gradient(quantiles))
      quantiles = tf.reshape(quantiles, quantiles_shape)
    quantile_net = tf.tile(quantiles, [1, self.quantile_embedding_dim])
    pi = tf.constant(math.pi)
    quantile_net = tf.cast(tf.range(
      1, self.quantile_embedding_dim + 1, 1), tf.float32) * pi * quantile_net
    quantile_net = tf.cos(quantile_net)
    # Create the quantile layer in the first call. This is because
    # number of output units depends on the input shape. Therefore, we can only
    # create the layer during the first forward call, not during `.__init__()`.
    if not hasattr(self, 'dense_quantile'):
      self.dense_quantile = tf.keras.layers.Dense(
        state_vector_length, activation=self.activation_fn,
        kernel_initializer=self.kernel_initializer)
    quantile_net = self.dense_quantile(quantile_net)
    x = tf.multiply(state_net_tiled, quantile_net)
    x = self.dense1(x)
    quantile_values = self.dense2(x)
    ret = FullyParameterizedQuantileNetworkType(quantile_values, quantiles,
                                                None)
    if proposed_quantiles is None:
      return ret._replace(fraction_proposal=fraction_proposal)
    return ret


class FractionProposalNetork(tf.keras.Model):
  def __init__(self, num_actions, num_quantiles=32, name=None):
    super(FractionProposalNetork, self).__init__(name=name)
    # 我们约定FPN输出的是tau,tau的中点是quantiles
    self.num_quantiles = num_quantiles
    self.activation_fn = tf.nn.log_softmax
    self.kernel_initializer = tf.keras.initializers.VarianceScaling(
      scale=1.0 / np.sqrt(3.0), mode='fan_in', distribution='uniform')
    # paper里面τ的计算输入是(s,a)pair，但是作者说效率很低并且没有提升
    self.dense = tf.keras.layers.Dense(
      num_quantiles,
      activation=self.activation_fn,
      kernel_initializer=self.kernel_initializer)
  
  def call(self, state_embedding):
    x = tf.stop_gradient(state_embedding)
    log_probs = self.dense(x)
    delta_taus = tf.exp(log_probs)
    taus = tf.cumsum(delta_taus, axis=1)
    entropies = -tf.reduce_sum(log_probs * delta_taus, axis=1)
    return FractionProposalNetworkType(delta_taus=delta_taus, taus=taus,
                                       entropies=entropies)
