import tensorflow as tf

from tensorflow.keras.layers import Dense, LayerNormalization, BatchNormalization
from typing import Union

from .layer import StandardLRPLayer

class DenseLRP(StandardLRPLayer):
    @staticmethod
    def forward(a: tf.Tensor, w: tf.Tensor) -> tf.Tensor:
        x = tf.tensordot(a, w, axes=1)

        return x

    @staticmethod
    def backward(w: tf.Tensor, s: tf.Tensor) -> tf.Tensor:
        # Manual dot-product to handle batch-dimension
        return tf.reduce_sum(tf.multiply(w, s), axis=-1)

    def _compute_with_alpha_beta(self, a, w, R, bias) -> tf.Tensor:
        a = tf.reshape(a, (-1, a.shape[1], 1))
        aw = a * w

        pos = tf.where(aw > 0, aw, tf.zeros_like(aw))
        pos_sums = tf.reduce_sum(pos, axis=(0, 1))
        neg = tf.where(aw < 0, aw, tf.zeros_like(aw))
        neg_sums = tf.reduce_sum(neg, axis=(0, 1))

        if bias is not None:
            pos_sums = tf.add(pos_sums, tf.maximum(0., bias))
            neg_sums = tf.add(neg_sums, tf.minimum(0., bias))

        pos = tf.divide(pos, pos_sums)
        pos = tf.where(pos_sums != 0, pos, tf.zeros_like(pos))

        neg = tf.divide(neg, neg_sums)
        neg = tf.where(neg_sums != 0, neg, tf.zeros_like(neg))

        x = self.alpha * pos - self.beta * neg

        R = tf.expand_dims(R, axis=0)

        return tf.reduce_sum(x * R, axis=-1)

    def get_weights(self, inputs) -> tf.Tensor:
        bias, weights = super().get_weights()

        if self.norm is None:
            pass
        elif isinstance(self.norm, (BatchNormalization, LayerNormalization)):
            if isinstance(self.norm, BatchNormalization):
                raise NotImplementedError()
            elif isinstance(self.norm, LayerNormalization):
                output = self.layer(inputs)
                mean, var = tf.nn.moments(output, axes=[1], keepdims=True)

            weights = (self.norm.gamma * weights) / \
                            tf.sqrt(var + 1e-8)
            if bias is not None:
                bias = self.norm.beta + self.norm.gamma * ((bias - mean) / \
                        tf.sqrt(var + 1e-8))
        else:
            raise ValueError(f'Unknown normalization layer: {self.norm}')

        return bias, weights

    def compute_output_shape(self, input_shape):
        return (None, self.layer.input_spec.axes[-1])

    def __init__(self, layer, *args, name: str = 'dense_lrp',
                 norm: Union[BatchNormalization, LayerNormalization] = None,
                 **kwargs):
        assert isinstance(layer, Dense), \
            'DenseLRP should only be called with a Dense layer'

        self.norm = norm

        super().__init__(layer, *args, name=name, **kwargs)


