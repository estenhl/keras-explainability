import tensorflow as tf

from typing import List

from .layer import LRPLayer


class NoOpLRP(LRPLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, a, w):
        raise NotImplementedError()

    def backward(self, w: tf.Tensor, s: tf.Tensor) -> tf.Tensor:
        raise NotImplementedError()

    def call(self, inputs: List[tf.Tensor]) -> tf.Tensor:
        _, R = inputs

        return R
