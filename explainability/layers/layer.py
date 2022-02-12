import tensorflow as tf

from abc import ABC, abstractmethod
from tensorflow.keras.layers import Dense, Layer
from typing import List

class LRPLayer(Layer, ABC):
    @abstractmethod
    def forward(a: tf.Tensor, w: tf.Tensor) -> tf.Tensor:
        pass

    @abstractmethod
    def backward(w: tf.Tensor, s: tf.Tensor) -> tf.Tensor:
        pass

    def __init__(self, layer, *, epsilon: float = None, gamma: float = None,
                 name='dense_lrp'):
        super().__init__(trainable=False, name=name)

        assert not (epsilon is not None and gamma is not None), \
            'DenseLRP should not be used with both epsilon and gamma'

        #if hasattr(layer, 'use_bias') and layer.use_bias:
        #    raise NotImplementedError(('LRP for Dense layers with bias is not '
        #                               'implemented'))

        self.layer = layer
        self.epsilon = epsilon
        self.gamma = gamma

    def compute_output_shape(self, input_shape):
        return self.layer.input_shape

    def call(self, inputs: List[tf.Tensor]) -> tf.Tensor:
        a, R = inputs
        w = self.layer.weights[0]

        if self.gamma:
            w = tf.where(w >= 0, tf.multiply(w, 1 + self.gamma), w)

        z = self.forward(a, w)
        s = R / z

        if self.epsilon:
            z = tf.add(z, self.epsilon)

        s = R / z

        c = self.backward(w, s)
        R = tf.multiply(a, c)

        return R
