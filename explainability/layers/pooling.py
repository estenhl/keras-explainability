import tensorflow as tf

from enum import Enum
from tensorflow.keras.layers import AveragePooling2D, AveragePooling3D, \
                                    GlobalAveragePooling2D, \
                                    GlobalAveragePooling3D, \
                                    GlobalMaxPooling2D, GlobalMaxPooling3D, \
                                    MaxPooling2D, MaxPooling3D
from tensorflow.raw_ops import AvgPool3DGrad, AvgPoolGrad, MaxPoolGradV2, MaxPool3DGrad

from typing import List

from .layer import LRPLayer


class MaxPoolingLRP(LRPLayer):
    class Strategy(Enum):
        WINNER_TAKE_ALL = 'winner-take-all'
        REDISTRIBUTE = 'redistribute'

    def __init__(self, layer, *args, strategy: str = 'winner-take-all',
                 name: str = 'max_pooling_lrp', **kwargs):
        assert isinstance(layer, (GlobalMaxPooling2D, GlobalMaxPooling3D,
                                  MaxPooling2D, MaxPooling3D)), \
            ('MaxPoolingLRP should only be called with '
             'MaxPooling2D, GlobalMaxPooling2D layers')

        super().__init__(layer, *args, name=name, **kwargs)

        self.strategy = MaxPoolingLRP.Strategy(strategy)

    def call(self, inputs: List[tf.Tensor]) -> tf.Tensor:
        if self.strategy != MaxPoolingLRP.Strategy.WINNER_TAKE_ALL:
            raise NotImplementedError(('Only winner-take-all strategy is '
                                       'implemented for LRP for maxpooling '
                                       'layers'))

        a, R = inputs

        ksize = self.layer.pool_size \
                if isinstance(self.layer, (MaxPooling2D, MaxPooling3D)) \
                else a.shape[1:-1]
        strides = self.layer.strides \
                  if isinstance(self.layer, (MaxPooling2D, MaxPooling3D)) \
                  else tuple([1] * (len(a.shape) - 2))
        padding = self.layer.padding.upper() \
                  if isinstance(self.layer, (MaxPooling2D, MaxPooling3D)) \
                  else 'VALID'

        forward = tf.nn.max_pool(a, ksize=ksize, strides=strides,
                                 padding=padding, name=f'{self.name}/forward')

        ksize = (1,) + ksize + (1,)
        strides = (1,) + strides + (1,)

        # Global layer has squeezed away spatial dimensions
        if len(R.shape) != len(a.shape):
            extra_dims = len(a.shape) - len(R.shape)
            dims = [R.shape[0]] + [1] * extra_dims + [R.shape[-1]]
            dims = [dim if dim is not None else -1 for dim in dims]

            R = tf.reshape(R, dims, name=f'{self.name}/R/reshape')

        if len(a.shape) == 4:
            gradients = MaxPoolGradV2(orig_input=a, orig_output=forward,
                                      grad=R, ksize=ksize, strides=strides,
                                      padding=padding, data_format='NHWC')
        elif isinstance(self.layer, GlobalMaxPooling3D):
            gradients = MaxPool3DGrad(orig_input=a, orig_output=forward,
                                      grad=R, ksize=ksize.as_list(),
                                      strides=strides, padding=padding,
                                      data_format='NDHWC')
        elif isinstance(self.layer, MaxPooling3D):
            gradients = MaxPool3DGrad(orig_input=a, orig_output=forward,
                                      grad=R, ksize=ksize,
                                      strides=strides, padding=padding,
                                      data_format='NDHWC')
        else:
            raise ValueError(f'Unable to handle layer {self.layer}')

        return gradients

class AveragePoolingLRP(LRPLayer):
    def __init__(self, layer, *args, batch_size: int = 1,
                 name: str = 'global_pooling_lrp', **kwargs):
        assert isinstance(layer, (AveragePooling2D, AveragePooling3D,
                                  GlobalAveragePooling2D,
                                  GlobalAveragePooling3D)), \
            ('AveragePoolingLRP should only be called with '
             'AveragePooling2D, GlobalAveragePooling2D layers')

        super().__init__(layer, *args, name=name, **kwargs)

        self.batch_size = batch_size

    def forward(self, a: tf.Tensor) -> tf.Tensor:
        ksize = self.layer.pool_size \
                    if isinstance(self.layer, (AveragePooling2D,
                                               AveragePooling3D)) \
                    else a.shape[1:-1].as_list()
        strides = self.layer.strides \
                  if isinstance(self.layer, (AveragePooling2D,
                                             AveragePooling3D)) \
                  else a.shape[1:-1].as_list()
        padding = self.layer.padding.upper() \
                  if isinstance(self.layer, (AveragePooling2D,
                                             AveragePooling3D)) \
                  else 'VALID'

        return tf.nn.avg_pool(a, ksize=ksize,
                              strides=strides,
                              padding=padding,
                              name=f'{self.name}/forward')

    def backward(self, a: tf.Tensor, s: tf.Tensor) -> tf.Tensor:
        pool_size = self.layer.pool_size \
                    if isinstance(self.layer, (AveragePooling2D,
                                               AveragePooling3D)) \
                    else tuple(a.shape[1:-1].as_list())
        strides = self.layer.strides \
                  if isinstance(self.layer, (AveragePooling2D,
                                             AveragePooling3D)) \
                  else tuple(a.shape[1:-1].as_list())

        input_shape = (self.batch_size,) + a.shape[1:]
        ksize = (1,) + pool_size + (1,)
        strides = (1,) + strides + (1,)
        padding = self.layer.padding.upper() \
                  if isinstance(self.layer, (AveragePooling2D,
                                             AveragePooling3D)) \
                  else 'VALID'

        if isinstance(self.layer, (AveragePooling3D, GlobalAveragePooling3D)):
            return AvgPool3DGrad(orig_input_shape=input_shape,
                                 grad=s,
                                 ksize=ksize,
                                 strides=strides,
                                 padding=padding,
                                 name=f'{self.name}/backward')
        else:
            return AvgPoolGrad(orig_input_shape=input_shape,
                                 grad=s,
                                 ksize=ksize,
                                 strides=strides,
                                 padding=padding,
                                 name=f'{self.name}/backward')

    def call(self, inputs: List[tf.Tensor]) -> tf.Tensor:
        a, R = inputs

        expected_shape = (self.batch_size,) + a.shape[1:]
        tf.ensure_shape(a, expected_shape)

        z = self.forward(a)
        s = R / z

        c = self.backward(a, s)
        R = tf.multiply(a, c)

        return R
