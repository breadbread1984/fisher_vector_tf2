#!/usr/bin/python3

import tensorflow as tf;
import tensorflow_probability as tfp;

class GMMLayer(tf.keras.layers.Layer):
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
        tfp.distributions.MultivariateNormalDiag(loc = self.locs[i], scale_diag = self.scales[i]) for i in range(self.kernel_num)
      ]);
    return gmm.prob(inputs);
  def get_config(self,):
    config = super(GMM, self).get_config();
    config['kernel_num'] = self.kernel_num;
    return config;
  @classmethod
  def from_config(cls, config):
    return cls(**config);

def GMM(input_shape, kernel_num = 3):
  inputs = tf.keras.Input(input_shape); # inputs.shape = (batch, ...)
  probs = GMMLayer(kernel_num)(inputs); # probs.shape = (batch)
  log_probs = tf.keras.layers.Lambda(lambda x: tf.math.log(x))(probs); # log_probs.shape = (batch)
  return tf.keras.Model(inputs = inputs, outputs = log_probs);

def fisher_kernel(kernel_num = 3, dim = 10):
  gmm = GMM((dim,), kernel_num);
  inputs = tf.random.normal(loc = 5., scale = 3., shape = (1,dim,));
  with tf.GradientTape() as g:
    outputs = gmm(inputs); # log p(x)
  grads = g.gradient(outputs, gmm.trainable_variables);
  print(grads);

if __name__ == "__main__":
  fisher_kernel();
