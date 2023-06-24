import os
import numpy as np
import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import BatchNormalization, Conv2D, Conv3D, \
                                    Dense, Input

from explainability.model.utils import fuse_batchnorm


"""
def test_fuse_batchnorm_dense():
    np.random.seed(42)

    i = Input((3,))
    x = Dense(3, activation=None)(i)

    bn = BatchNormalization(
        beta_initializer=tf.random_normal_initializer(mean=0, stddev=0.5,
                                                      seed=42),
        gamma_initializer=tf.random_normal_initializer(mean=1, stddev=0.5,
                                                       seed=42),
        moving_mean_initializer=tf.random_normal_initializer(mean=0, stddev=0.5,
                                                             seed=42),
        moving_variance_initializer=tf.random_normal_initializer(mean=1,
                                                                 stddev=0.5,
                                                                 seed=42)
    )(x)
    model_without_bn = Model(i, x)
    model_with_bn = Model(i, bn)

    weights = np.random.uniform(0, 1, (3, 3))
    biases = np.random.uniform(0, 1, 3)

    model_without_bn.layers[1].set_weights([weights, biases])
    model_with_bn.layers[1].set_weights([weights, biases])
    model_with_fused_bn = fuse_batchnorm(model_with_bn)

    assert model_with_bn != model_with_fused_bn, \
        'fuse_batchnorm overwrites the old model'

    data = np.reshape(np.linspace(0, 1, 3), (1, 3))

    predictions_without_bn = model_without_bn(data)
    predictions_with_bn = model_with_bn(data)
    predictions_with_fused_bn = model_with_fused_bn(data)

    assert not np.allclose(predictions_without_bn, predictions_with_bn,
                           atol=1e-2), \
        'Model with batchnorm behaves equivalently as model without'

    fused_bn = model_with_fused_bn.layers[-1]
    beta = fused_bn.beta.numpy()
    gamma = fused_bn.gamma.numpy()
    mean = fused_bn.moving_mean.numpy()
    var = fused_bn.moving_variance.numpy()

    assert np.all(beta == 0), \
        ('Model with fused batch normalization layers does not have identity '
         'batch norm betas')
    assert np.all(gamma == 1), \
        ('Model with fused batch normalization layers does not have identity '
         'batch norm gammas')
    assert np.all(mean == 0), \
        ('Model with fused batch normalization layers does not have identity '
         'batch norm moving means')
    assert np.all(var == 1), \
        ('Model with fused batch normalization layers does not have identity '
         'batch norm moving variances')

    assert np.allclose(predictions_with_bn, predictions_with_fused_bn,
                       atol=1e-2), \
        ('Model with fused batch normalization layers does not return same '
         'predictions as the original model')


def test_fuse_batchnorm_conv2d():
    np.random.seed(42)

    i = Input((3, 3, 3))
    x = Conv2D(2, (3, 3), activation=None)(i)

    bn = BatchNormalization(
        beta_initializer=tf.random_normal_initializer(mean=0, stddev=0.5,
                                                      seed=42),
        gamma_initializer=tf.random_normal_initializer(mean=1, stddev=0.5,
                                                       seed=42),
        moving_mean_initializer=tf.random_normal_initializer(mean=0, stddev=0.5,
                                                             seed=42),
        moving_variance_initializer=tf.random_normal_initializer(mean=1,
                                                                 stddev=0.5,
                                                                 seed=42)
    )(x)
    model_with_bn = Model(i, bn)

    weights = np.random.uniform(0, 1, (3, 3, 3, 2))
    biases = np.random.uniform(0, 1, 2)

    model_with_bn.layers[1].set_weights([weights, biases])
    model_with_fused_bn = fuse_batchnorm(model_with_bn)

    data = np.reshape(np.linspace(0, 1, 3*3*3), (1, 3, 3, 3))

    predictions_with_bn = model_with_bn(data)
    predictions_with_fused_bn = model_with_fused_bn(data)

    fused_bn = model_with_fused_bn.layers[-1]
    beta = fused_bn.beta.numpy()
    gamma = fused_bn.gamma.numpy()
    mean = fused_bn.moving_mean.numpy()
    var = fused_bn.moving_variance.numpy()

    assert np.all(beta == 0), \
        ('Model with fused batch normalization layers does not have identity '
         'batch norm betas')
    assert np.all(gamma == 1), \
        ('Model with fused batch normalization layers does not have identity '
         'batch norm gammas')
    assert np.all(mean == 0), \
        ('Model with fused batch normalization layers does not have identity '
         'batch norm moving means')
    assert np.all(var == 1), \
        ('Model with fused batch normalization layers does not have identity '
         'batch norm moving variances')

    # atol=1 because rounding errors increase as a function of the number of
    # inputs
    assert np.allclose(predictions_with_bn, predictions_with_fused_bn,
                       atol=1), \
        ('Model with fused batch normalization layers does not return same '
         'predictions as the original model')

"""
def test_fuse_batchnorm_conv3d():
    i = Input((3, 3, 3, 5))
    x = Conv3D(3, (3, 3, 3), activation=None)(i)

    bn = BatchNormalization(
        beta_initializer=tf.random_normal_initializer(mean=0, stddev=0.5,
                                                      seed=42),
        gamma_initializer=tf.random_normal_initializer(mean=1, stddev=0.5,
                                                       seed=42),
        moving_mean_initializer=tf.random_normal_initializer(mean=0, stddev=0.5,
                                                             seed=42),
        moving_variance_initializer=tf.random_normal_initializer(mean=1,
                                                                 stddev=0.5,
                                                                 seed=42)
    )(x)
    model_with_bn = Model(i, bn)

    weights = np.random.uniform(0, 1, (3, 3, 3, 5, 3))
    biases = np.random.uniform(0, 1, 3)

    model_with_bn.layers[1].set_weights([weights, biases])
    model_with_fused_bn = fuse_batchnorm(model_with_bn)

    data = np.reshape(np.arange(3*3*3*5), (1, 3, 3, 3, 5))

    predictions_with_bn = model_with_bn(data)
    predictions_with_fused_bn = model_with_fused_bn(data)

    fused_bn = model_with_fused_bn.layers[-1]
    beta = fused_bn.beta.numpy()
    gamma = fused_bn.gamma.numpy()
    mean = fused_bn.moving_mean.numpy()
    var = fused_bn.moving_variance.numpy()

    assert np.all(beta == 0), \
        ('Model with fused batch normalization layers does not have identity '
         'batch norm betas')
    assert np.all(gamma == 1), \
        ('Model with fused batch normalization layers does not have identity '
         'batch norm gammas')
    assert np.all(mean == 0), \
        ('Model with fused batch normalization layers does not have identity '
         'batch norm moving means')
    assert np.all(var == 1), \
        ('Model with fused batch normalization layers does not have identity '
         'batch norm moving variances')

    assert np.allclose(predictions_with_bn, predictions_with_fused_bn,
                       atol=10), \
        ('Model with fused batch normalization layers does not return same '
         'predictions as the original model')

"""
def test_fuse_batchnorm_real():
    model = ResNet50(weights='imagenet')
    image = np.load(os.path.join(os.path.dirname(__file__), 'data',
                                 'preprocessed_cat.npy'))
    image = np.expand_dims(image, 0)
    predictions = model.predict(image)[0]

    print(np.argmax(predictions))
    print(np.amax(predictions))

    model_with_fused_bn = fuse_batchnorm(model)
    predictions_with_fused_bn = model_with_fused_bn.predict(image)[0]

    assert np.allclose(predictions, predictions_with_fused_bn, atol=1e-5), \
        ('VGG19 with fused batchnorm layers does not return the correct '
         'predictions')

    for layer in model_with_fused_bn.layers:
        if isinstance(layer, BatchNormalization):
            assert np.all(layer.beta.numpy() == 0)
            assert np.all(layer.gamma.numpy() == 1)
            assert np.all(layer.moving_mean.numpy() == 0)
            assert np.all(layer.moving_variance.numpy() == 1)
"""
