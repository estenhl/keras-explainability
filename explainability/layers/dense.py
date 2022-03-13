import tensorflow as tf

from tensorflow.keras.layers import Dense

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

    def _compute_with_alpha_beta(self, a, w, R) -> tf.Tensor:
        a = tf.reshape(a, (-1, a.shape[1], 1))
        aw = a * w

        pos = tf.where(aw > 0, aw, tf.zeros_like(aw))
        pos_sums = tf.reduce_sum(pos, axis=(0, 1))
        pos = tf.divide(pos, pos_sums)
        pos = tf.where(pos_sums != 0, pos, tf.zeros_like(pos))
        neg = tf.where(aw < 0, aw, tf.zeros_like(aw))
        neg_sums = tf.reduce_sum(neg, axis=(0, 1))
        neg = tf.divide(neg, neg_sums)
        neg = tf.where(neg_sums != 0, neg, tf.zeros_like(neg))

        x = self.alpha * pos - self.beta * neg

        R = tf.expand_dims(R, axis=0)

        return tf.reduce_sum(x * R, axis=-1)

    def __init__(self, layer, *args, name='dense_lrp', **kwargs):
        assert isinstance(layer, Dense), \
            'DenseLRP should only be called with a Dense layer'

        super().__init__(layer, *args, name=name, **kwargs)


