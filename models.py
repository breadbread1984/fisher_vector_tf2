#!/usr/bin/python3

import tensorflow as tf;
import tensorflow_probability as tfp;

class GMMLayer(tf.keras.layers.Layer):
  def __init__(self, kernel_num = 3, **kwargs):
    self.kernel_num = kernel_num;
    super(GMMLayer, self).__init__(**kwargs);
  def build(self, input_shape):
    self.cats = self.add_weight(shape = (self.kernel_num, ), dtype = tf.float32, trainable = True, initializer = tf.keras.initializers.Constant(1./self.kernel_num), name = 'cats');
    self.locs = self.add_weight(shape = [self.kernel_num,] + list(input_shape)[1:], dtype = tf.float32, trainable = True, initializer = tf.keras.initializers.RandomNormal(), name = 'locs');
    self.scales = self.add_weight(shape = [self.kernel_num,] + list(input_shape)[1:], dtype = tf.float32, trainable = True, initializer = tf.keras.initializers.Constant(.1), name = 'scales');
  def call(self, inputs):
    gmm = tfp.distributions.Mixture(
      cat = tfp.distributions.Categorical(probs = self.cats),
      components = [
        tfp.distributions.MultivariateNormalDiag(loc = self.locs[i], scale_diag = self.scales[i]) for i in range(self.kernel_num)
      ]);
    return gmm.log_prob(inputs);
  def get_config(self,):
    config = super(GMMLayer, self).get_config();
    config['kernel_num'] = self.kernel_num;
    return config;
  @classmethod
  def from_config(cls, config):
    return cls(**config);

def get_gradient_vector(inputs, kernel_num = 5):
  gmm = GMMLayer(kernel_num);
  with tf.GradientTape() as g:
    outputs = gmm(inputs); # output.shape = (batch,)
  all_grads = list();
  grads = g.gradient(outputs, gmm.trainable_variables);
  print(grads, all_grads);

if __name__ == "__main__":
  import numpy as np;
  data = np.random.normal(size = (10,100));
  get_gradient_vector(data);
