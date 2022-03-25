from abc import abstractproperty
import tensorflow as tf

from tensorflow.keras.layers import Conv2D, Conv3D
from typing import Callable, Tuple

from .layer import StandardLRPLayer


class ConvLRP(StandardLRPLayer):
    @abstractproperty
    def convolve(self) -> Callable:
        pass

    @abstractproperty
    def convolve_transpose(self) -> Callable:
        pass

    @abstractproperty
    def strides(self) -> Tuple[int]:
        pass

    @abstractproperty
    def input_shape(self):
        pass

    @property
    def padding(self) -> str:
        return self.layer.padding.upper()

    def forward(self, a, w):
        return self.__class__.convolve(a, w,
                                       strides=self.strides,
                                       padding=self.padding)

    def backward(self, w: tf.Tensor, s: tf.Tensor) -> tf.Tensor:
        return self.__class__.convolve_transpose(s, w,
                                                 output_shape=self.input_shape,
                                                 strides=self.layer.strides,
                                                 padding=self.padding)

    def _compute_with_alpha_beta(self, a, w, R) -> tf.Tensor:
        apos = tf.maximum(a, 0.)
        wpos = tf.maximum(w, 0.)
        aneg = tf.minimum(a, 0.)
        wneg = tf.minimum(w, 0.)

        zpospos = self.forward(apos, wpos)
        znegneg = self.forward(aneg, wneg)

        zpos = zpospos + znegneg

        zposneg = self.forward(apos, wneg)
        znegpos = self.forward(aneg, wpos)

        zneg = zposneg + znegpos

        Rpos = R / (zpos + 1e-9)
        Rneg = R / (zneg + 1e-9)

        # TODO: Only works for batches of size 1. Should raise error otherwise

        cpospos = self.backward(wpos, Rpos)
        cposneg = self.backward(wneg, Rpos)

        cnegneg = self.backward(wneg, Rneg)
        cnegpos = self.backward(wpos, Rneg)

        cpos = apos * cpospos + aneg * cposneg
        cneg = apos * cnegneg + aneg * cnegpos

        Rpos = tf.multiply(tf.cast(self.alpha, tf.float32), cpos)
        Rneg = tf.multiply(tf.cast(self.beta, tf.float32), cneg)

        return Rpos - Rneg


class Conv2DLRP(ConvLRP):
    convolve = tf.nn.conv2d
    convolve_transpose = tf.nn.conv2d_transpose

    @property
    def strides(self) -> Tuple[int]:
        return self.layer.strides

    @property
    def input_shape(self):
        return self.layer.input_shape[1:-1]

    def __init__(self, layer, *args, name: str = 'conv2d_lrp', **kwargs):
        assert isinstance(layer, Conv2D), \
            'Conv2DLRP should only be called with a Conv2D layer'

        super().__init__(layer, *args, name=name, **kwargs)


class Conv3DLRP(ConvLRP):
    convolve = tf.nn.conv3d
    convolve_transpose = tf.nn.conv3d_transpose

    @property
    def strides(self):
        return (1,) + self.layer.strides + (1,)

    @property
    def input_shape(self):
        return (1,) + self.layer.input_shape[1:]

    def __init__(self, layer, *args, name: str = 'conv3d_lrp', **kwargs):
        assert isinstance(layer, Conv3D), \
            'Conv3DLRP should only be called with a Conv3D layer'

        super().__init__(layer, *args, name=name, **kwargs)
