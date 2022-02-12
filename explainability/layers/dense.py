import tensorflow as tf

from tensorflow.keras.layers import Dense

from .layer import LRPLayer

class DenseLRP(LRPLayer):
    @staticmethod
    def forward(a: tf.Tensor, w: tf.Tensor) -> tf.Tensor:
        return tf.tensordot(a, w, axes=1)

    @staticmethod
    def backward(w: tf.Tensor, s: tf.Tensor) -> tf.Tensor:
        # Manual dot-product to handle batch-dimension
        return tf.reduce_sum(tf.multiply(w, s), axis=-1)

    def __init__(self, layer, *args, name='dense_lrp', **kwargs):
        assert isinstance(layer, Dense), \
            'DenseLRP should only be called with a Dense layer'

        super().__init__(layer, *args, name=name, **kwargs)


