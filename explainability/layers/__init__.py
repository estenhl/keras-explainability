from tensorflow.keras.layers import Activation, AveragePooling2D, \
                                    AveragePooling3D, BatchNormalization, \
                                    Conv2D, Conv3D, Dense, Dropout, Flatten, \
                                    GlobalAveragePooling2D, \
                                    GlobalAveragePooling3D, \
                                    GlobalMaxPooling3D, MaxPooling2D, \
                                    MaxPooling3D, Reshape, ReLU

from .activations import ReLULRP
from .conv import Conv2DLRP, Conv3DLRP
from .dense import DenseLRP
from .noop import NoOpLRP
from .normalization import BatchNormalizationLRP
from .pooling import AveragePoolingLRP, MaxPoolingLRP
from .reshape import ReshapeLRP

#Add

def LRP(layer, *args, **kwargs):
    if isinstance(layer, Activation) and layer.activation.__name__ == 'relu':
        return ReLULRP(layer, *args, **kwargs)
    elif isinstance(layer, AveragePooling2D):
        return AveragePoolingLRP(layer, *args, **kwargs)
    elif isinstance(layer, AveragePooling3D):
        return AveragePoolingLRP(layer, *args, **kwargs)
    elif isinstance(layer, BatchNormalization):
        return NoOpLRP(layer, *args, **kwargs)#BatchNormalizationLRP(layer, *args, **kwargs)
    elif isinstance(layer, Conv2D):
        return Conv2DLRP(layer, *args, **kwargs)
    elif isinstance(layer, Conv3D):
        return Conv3DLRP(layer, *args, **kwargs)
    elif isinstance(layer, Dense):
        return DenseLRP(layer, *args, **kwargs)
    elif isinstance(layer, Dropout):
        return NoOpLRP(layer, *args, **kwargs)
    elif isinstance(layer, GlobalAveragePooling2D):
        return AveragePoolingLRP(layer, *args, **kwargs)
    elif isinstance(layer, GlobalAveragePooling3D):
        return AveragePoolingLRP(layer, *args, **kwargs)
    elif isinstance(layer, GlobalMaxPooling3D):
        return MaxPoolingLRP(layer, *args, **kwargs)
    elif isinstance(layer, Flatten):
        return ReshapeLRP(layer, *args, **kwargs)
    elif isinstance(layer, (MaxPooling2D, MaxPooling3D)):
        return MaxPoolingLRP(layer, *args, **kwargs)
    elif isinstance(layer, Reshape):
        return ReshapeLRP(layer, *args, **kwargs)
    elif isinstance(layer, ReLU):
        return ReLULRP(layer, *args, **kwargs)
    else:
        raise NotImplementedError('LRP is not implemented for layer '
                                  f'{layer.__class__.__name__}')
