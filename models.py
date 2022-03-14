#!/usr/bin/python3

import tensorflow as tf;
import tensorflow_probability as tfp;

class GMM(tf.keras.layers.Layer):
  def __init__(self, kernel_num = 3, **kwargs):
    self.kernel_num = kernel_num;
    super(GMM, self).__init__(**kwargs);
  def build(self, input_shape):
    self.cats = self.add_weight(shape = (self.kernel_num, ), dtype = tf.float32, trainable = True, initializer = tf.keras.initializers.Constant(1./self.kernel_num), name = 'cats');
    self.locs = self.add_weight(shape = [self.kernel_num,] + list(input_shape)[1:], dtype = tf.float32, trainable = True, name = 'locs');
    self.scales = self.add_weight(shape = [self.kernel_num,] + list(input_shape)[1:], dtype = tf.float32, trainable = True, initializer = tf.keras.initializers.Constant(.1), name = 'scales');
  def call(self, inputs):
    gmm = tfp.distributions.Mixture(
      cat = tfp.distributions.Categorical(probs = self.cats),
      components = [
        tfp.distributions.Normal(loc = self.locs[i], scale = self.scales[i]) for i in range(self.kernel_num)
      ]);
    return gmm.prob(inputs);
  def get_config(self,):
    config = super(GMM, self).get_config();
    config['kernel_num'] = self.kernel_num;
    return config;
  @classmethod
  def from_config(cls, config):
    return cls(**config);

if __name__ == "__main__":
  gmm = GMM(3);
  import numpy as np;
  inputs = np.random.normal(size = (10));
  probs = gmm(inputs);
  print(probs.shape)
