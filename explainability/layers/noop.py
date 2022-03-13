import tensorflow as tf

from typing import List

from .layer import LRPLayer


class NoOpLRP(LRPLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def call(self, inputs: List[tf.Tensor]) -> tf.Tensor:
        _, R = inputs

        return R
