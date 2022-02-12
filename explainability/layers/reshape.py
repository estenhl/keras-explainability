import tensorflow as tf

from tensorflow.keras.layers import Flatten, Reshape
from typing import List

from .layer import LRPLayer

class ReshapeLRP(LRPLayer):
    def __init__(self, layer, *args, name: str = 'reshape_lrp', **kwargs):
        assert isinstance(layer, (Flatten, Reshape)), \
            'ReshapeLRP should only be called with a Flatten or Reshape layer'

        super().__init__(layer, *args, name=name, **kwargs)

    def forward(self, a, w):
        raise NotImplementedError()

    def backward(self, w: tf.Tensor, s: tf.Tensor) -> tf.Tensor:
        raise NotImplementedError()

    def call(self, inputs: List[tf.Tensor]) -> tf.Tensor:
        a, R = inputs

        shape = [x if x != None else -1 for x in a.shape]
        print(f'Shape: {shape}')

        return tf.reshape(R, shape, name=self.name)
