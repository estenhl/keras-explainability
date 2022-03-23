import tensorflow as tf

from tensorflow.keras.layers import Add, Subtract
from typing import List

from .layer import LRPLayer


def _compute_add_lrp(a: tf.Tensor, b: tf.Tensor, R: tf.Tensor,
                     *, name: str, epsilon: float = 1e-9) -> tf.Tensor:
    forward = tf.add(a, b, name=f'{name}/forward')
    forward = tf.add(forward, epsilon, name=f'{name}/forward/epsilon')
    a = tf.divide(a, forward, name=f'{name}/a')
    a = tf.multiply(a, R, name=f'{name}/a/R')
    b = tf.divide(b, forward, name=f'{name}/b')
    b = tf.multiply(b, R, name=f'{name}/b/R')

    return [a, b]

class AddLRP(LRPLayer):
    def __init__(self, layer, *args, name: str = 'add_lrp', **kwargs):
        assert isinstance(layer, Add), \
            ('AddLRP should only be called with an Add layer')

        super().__init__(layer, *args, name=name, **kwargs)

    def call(self, inputs: List[tf.Tensor]) -> tf.Tensor:
        (a, b), R = inputs

        return _compute_add_lrp(a, b, R, name=self.name)

class SubtractLRP(LRPLayer):
    def __init__(self, layer, *args, name: str = 'add_lrp', **kwargs):
        assert isinstance(layer, Subtract), \
            ('SubtractLRP should only be called with an Subtract layer')

        super().__init__(layer, *args, name=name, **kwargs)

    def call(self, inputs: List[tf.Tensor]) -> tf.Tensor:
        (a, b), R = inputs
        b = tf.math.negative(b, name=f'{self.name}/negate')

        return _compute_add_lrp(a, b, R, name=self.name)
