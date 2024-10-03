import tensorflow as tf

from abc import ABC
from enum import Enum
from tensorflow.keras.layers import AveragePooling2D, AveragePooling3D, \
                                    GlobalAveragePooling2D, \
                                    GlobalAveragePooling3D, \
                                    GlobalMaxPooling2D, GlobalMaxPooling3D, \
                                    MaxPooling2D, MaxPooling3D
from tensorflow.raw_ops import AvgPool3DGrad, AvgPoolGrad, MaxPoolGradV2, MaxPool3DGrad

from typing import List

from .layer import LRPLayer


class PoolingLRPLayer(LRPLayer, ABC):
    class Strategy(Enum):
        WINNER_TAKES_ALL = 'winner-takes-all'
        REDISTRIBUTE = 'redistribute'
        FLAT = 'flat'

    def _is_3d_pooling_layer(self) -> bool:
        layers = (MaxPooling3D, GlobalMaxPooling3D, AveragePooling3D,
                  GlobalAveragePooling3D)

        return isinstance(self.layer, layers)

    @property
    def strategy(self) -> Strategy:
        return self._strategy

    @strategy.setter
    def strategy(self, strategy):
        self._strategy = PoolingLRPLayer.Strategy(strategy)

    def compute_output_shape(self, input_shape):
        return self.layer.input.shape

    def __init__(self, layer, *args, strategy: str, batch_size: int = 1,
                 name: str = 'pooling_lrp', **kwargs):
        super().__init__(layer, *args, name=name, **kwargs)

        self.strategy = strategy
        self.batch_size = batch_size

    def _winner_takes_all(self, a: tf.Tensor, R: tf.Tensor, ksize, strides, padding) -> tf.Tensor:
        forward = tf.nn.max_pool(a, ksize=ksize, strides=strides,
                                 padding=padding, name=f'{self.name}/forward')

        ksize = [1] + ksize + [1]
        strides = [1] + strides + [1]

        # Global layer has squeezed away spatial dimensions
        if len(R.shape) != len(a.shape):
            extra_dims = len(a.shape) - len(R.shape)
            dims = [R.shape[0]] + [1] * extra_dims + [R.shape[-1]]
            dims = [dim if dim is not None else -1 for dim in dims]

            R = tf.reshape(R, dims, name=f'{self.name}/R/reshape')

        if len(a.shape) == 4:
            gradients = MaxPoolGradV2(orig_input=a,
                                      orig_output=forward,
                                      grad=R,
                                      ksize=ksize,
                                      strides=strides,
                                      padding=padding, data_format='NHWC')
        elif isinstance(self.layer, GlobalMaxPooling3D):
            gradients = MaxPool3DGrad(orig_input=a,
                                      orig_output=forward,
                                      grad=R,
                                      ksize=ksize,
                                      strides=strides,
                                      padding=padding,
                                      data_format='NDHWC')
        elif isinstance(self.layer, MaxPooling3D):
            gradients = MaxPool3DGrad(orig_input=a, orig_output=forward,
                                      grad=R, ksize=ksize,
                                      strides=strides, padding=padding,
                                      data_format='NDHWC')
        else:
            raise ValueError(f'Unable to handle layer {self.layer}')

        return gradients

    def _redistribute(self, a: tf.Tensor, R: tf.Tensor, ksize, strides, padding) -> tf.Tensor:
        z = tf.nn.avg_pool(a, ksize=ksize,
                              strides=strides,
                              padding=padding,
                              name=f'{self.name}/forward')
        s = R / z

        input_shape = (self.batch_size,) + a.shape[1:]
        ksize = [1] + ksize + [1]
        strides = [1] + strides + [1]
        padding = padding

        if self._is_3d_pooling_layer():
            c = AvgPool3DGrad(orig_input_shape=input_shape,
                             grad=s,
                             ksize=ksize,
                             strides=strides,
                             padding=padding,
                             name=f'{self.name}/backward')
        else:

            c = AvgPoolGrad(orig_input_shape=input_shape,
                            grad=s,
                            ksize=ksize,
                            strides=strides,
                            padding=padding,
                            name=f'{self.name}/backward')

        R = tf.multiply(a, c)

        return R

    def call(self, inputs: List[tf.Tensor]) -> tf.Tensor:
        a, R = inputs

        local = isinstance(self.layer, (AveragePooling2D, MaxPooling2D,
                                        AveragePooling3D, MaxPooling3D))

        ksize = list(self.layer.pool_size) if local \
                else a.shape[1:-1].as_list()
        strides = list(self.layer.strides) if local \
                  else list([1] * (len(a.shape) - 2))
        padding = self.layer.padding.upper() if local else 'VALID'


        if self.strategy == MaxPoolingLRP.Strategy.WINNER_TAKES_ALL:
            return self._winner_takes_all(a, R, ksize=ksize, strides=strides,
                                         padding=padding)
        elif self.strategy == MaxPoolingLRP.Strategy.REDISTRIBUTE:
            return self._redistribute(a, R, ksize=ksize, strides=strides,
                                      padding=padding)
        elif self.strategy == MaxPoolingLRP.Strategy.FLAT:
            return self._redistribute(tf.ones_like(a), R, ksize=ksize,
                                      strides=strides, padding=padding)
        else:
            raise NotImplementedError(('Only winner-take-all and redistribute '
                                       'strategy is implemented for LRP for '
                                       'maxpooling layers'))

class MaxPoolingLRP(PoolingLRPLayer):
    def __init__(self, layer, *args, strategy: str = 'winner-takes-all',
                 batch_size: int = 1,name: str = 'max_pooling_lrp', **kwargs):
        assert isinstance(layer, (GlobalMaxPooling2D, GlobalMaxPooling3D,
                                  MaxPooling2D, MaxPooling3D)), \
            ('MaxPoolingLRP should only be called with '
             'MaxPooling2D, GlobalMaxPooling2D layers')

        super().__init__(layer, *args, strategy=strategy,
                         batch_size=batch_size, name=name, **kwargs)


class AveragePoolingLRP(PoolingLRPLayer):
    def __init__(self, layer, *args, strategy: str = 'redistribute',
                 batch_size: int = 1, name: str = 'global_pooling_lrp',
                 **kwargs):
        assert isinstance(layer, (AveragePooling2D, AveragePooling3D,
                                  GlobalAveragePooling2D,
                                  GlobalAveragePooling3D)), \
            ('AveragePoolingLRP should only be called with '
             'AveragePooling2D, GlobalAveragePooling2D layers')

        super().__init__(layer, *args, strategy=strategy,
                         batch_size=batch_size, name=name, **kwargs)
