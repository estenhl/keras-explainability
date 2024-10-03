import tensorflow as tf

from abc import ABC, abstractmethod
from tensorflow.keras.layers import Layer
from typing import List

class LRPLayer(Layer, ABC):
    def get_weights(self, *args, **kwargs):
        bias = self.layer.bias if self.layer.use_bias else None
        return bias, self.layer.weights[0]

    def __init__(self, layer: tf.Tensor, name: str = 'lrp', **kwargs):
        super().__init__(trainable=False, name=name)

        self.layer = layer

class StandardLRPLayer(LRPLayer, ABC):
    @abstractmethod
    def forward(a: tf.Tensor, w: tf.Tensor) -> tf.Tensor:
        pass

    @abstractmethod
    def backward(w: tf.Tensor, s: tf.Tensor) -> tf.Tensor:
        pass

    def __init__(self, layer: tf.Tensor, *, epsilon: float = None,
                 gamma: float = None, alpha: float = None, beta: float = None,
                 b: bool = False, flat: bool = False, ignore_bias: bool = False,
                 adjust_epsilon: bool = False, name: str = 'dense_lrp'):
        super().__init__(layer, name=name)

        if epsilon is not None:
            assert gamma is None, \
                'DenseLRP should not be used with both epsilon and gamma'
            assert alpha is None and beta is None, \
                'DenseLRP should not be used with both epsilon and alpha/beta'
        elif gamma is not None:
            assert alpha is None and beta is None, \
                'DenseLRP should not be used with both gamma and alpha/beta'

        assert alpha is None and beta is None or \
               alpha is not None and beta is not None, \
            'If alpha or beta is used, they should both be used'

        if alpha is not None:
            assert alpha == beta + 1, \
                'beta must be equal to alpha + 1'

        self.layer = layer
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.b = b
        self.flat = flat
        self.ignore_bias = ignore_bias
        self.adjust_epsilon = adjust_epsilon

    def compute_output_shape(self, input_shape):
        return self.layer.input.shape

    def call(self, inputs: List[tf.Tensor]) -> tf.Tensor:
        a, R = inputs
        R_in = R

        if self.b:
            a = tf.ones_like(a)

        bias, w = self.get_weights(a)

        if self.flat:
            a = tf.ones_like(a)
            w = tf.ones_like(w)

        if self.gamma:
            w = tf.where(w >= 0, tf.multiply(w, 1 + self.gamma), w)

        if self.alpha is not None and self.beta is not None:
            return self._compute_with_alpha_beta(a, w, R, bias)
        else:
            z = self.forward(a, w)

            if bias is not None and not self.ignore_bias:
                R = (R * z) / (z + bias)

            if self.epsilon:
                epsilon = tf.multiply(self.epsilon, tf.math.sign(z))
                z = tf.add(z, epsilon)

            s = R / z
            s = tf.where(z == 0, tf.zeros_like(s), s)

            c = self.backward(w, s)
            R = tf.multiply(a, c)

            # Ensures that the amount of relevance stays fixed between layers
            # even with epsilon
            if self.epsilon and self.adjust_epsilon:
                R = R * tf.reduce_sum(R_in) / tf.reduce_sum(R)

        return R
