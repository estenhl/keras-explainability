import tensorflow as tf

from tensorflow.keras.layers import Activation, ReLU
from typing import List

from .layer import LRPLayer


class ReLULRP(LRPLayer):
    def __init__(self, layer, *args, name: str = 'relu_lrp', **kwargs):
        assert isinstance(layer, ReLU) or \
               (isinstance(layer, Activation) and \
                layer.activation.__name__ == 'relu'), \
            ('ReLULRP should only be called with a ReLU layer or an '
            'Activation layer with ReLU type')

        super().__init__(layer, *args, name=name, **kwargs)

    def call(self, inputs: List[tf.Tensor]) -> tf.Tensor:
        a, R = inputs

        return tf.where(a > 0, R, tf.zeros_like(R), name=self.name)
