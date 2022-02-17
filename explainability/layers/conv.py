import tensorflow as tf

from tensorflow.keras.layers import Conv2D, Conv3D

from .layer import LRPLayer


class Conv2DLRP(LRPLayer):
    def __init__(self, layer, *args, name: str = 'conv2d_lrp', **kwargs):
        assert isinstance(layer, Conv2D), \
            'Conv2DLRP should only be called with a Conv2D layer'

        super().__init__(layer, *args, name=name, **kwargs)

    def forward(self, a, w):
        return tf.nn.conv2d(a, w, strides=self.layer.strides,
                            padding=self.layer.padding.upper())

    def backward(self, w: tf.Tensor, s: tf.Tensor) -> tf.Tensor:
        return tf.nn.conv2d_transpose(s, w, output_shape=s.shape[1:-1],
                                      strides=self.layer.strides,
                                      padding=self.layer.padding.upper())

    def _compute_with_alpha_beta(self, a, w, R) -> tf.Tensor:
        padding = self.layer.padding.upper()
        z = tf.nn.conv2d(a, w, padding=padding, strides=(1, 1))
        zpos = tf.nn.conv2d(a, tf.maximum(0., w), padding=padding, strides=(1, 1))
        zneg = tf.nn.conv2d(a, tf.minimum(0., w), padding=padding, strides=(1, 1))

        Rpos = R / (zpos + 1e-9)
        Rneg = R / (zneg + 1e-9)
        z = tf.add(z, 1e-9)

        cpos = tf.nn.conv2d_transpose(Rpos, tf.maximum(0., w), output_shape=a.shape[1:-1],
                                    strides=(1, 1, 1, 1),
                                    padding=padding)

        cneg = tf.nn.conv2d_transpose(Rneg, tf.minimum(0., w), output_shape=a.shape[1:-1],
                           strides=(1, 1, 1, 1),
                           padding=padding)

        c = self.alpha * cpos - self.beta * cneg
        R = tf.multiply(a, c)

        return R


class Conv3DLRP(LRPLayer):
    def __init__(self, layer, *args, name: str = 'conv3d_lrp', **kwargs):
        assert isinstance(layer, Conv3D), \
            'Conv2DLRP should only be called with a Conv3D layer'

        super().__init__(layer, *args, name=name, **kwargs)

    def forward(self, a, w):
        return tf.nn.conv3d(a, w, strides=self.layer.strides,
                            padding=self.layer.padding.upper())

    def backward(self, w: tf.Tensor, s: tf.Tensor) -> tf.Tensor:
        return tf.nn.conv3d_transpose(s, w, output_shape=s.shape[1:-1],
                                      strides=self.layer.strides,
                                      padding=self.layer.padding.upper())
