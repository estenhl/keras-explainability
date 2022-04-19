import logging
import numpy as np
import tensorflow as tf

from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import BatchNormalization, Conv2D, Conv3D, Dense
from tensorflow.keras.models import clone_model
from typing import Union

from ...utils import infer_graph_structure

logformat = '%(asctime)s - %(levelname)s - %(name)s: %(message)s'
logging.basicConfig(format=logformat, level=logging.INFO)
logger = logging.getLogger(__name__)


def _fuse_layers(computational: Union[Dense, Conv2D, Conv3D],
                 batch_norm: BatchNormalization) -> None:
    betas = batch_norm.beta.numpy()
    gammas = batch_norm.gamma.numpy()
    means = batch_norm.moving_mean.numpy()
    variances = batch_norm.moving_variance.numpy()
    epsilon = batch_norm.epsilon

    batch_norm.beta.assign(tf.zeros_like(batch_norm.beta))
    batch_norm.gamma.assign(tf.ones_like(batch_norm.gamma))
    batch_norm.moving_mean.assign(tf.zeros_like(batch_norm.moving_mean))
    batch_norm.moving_variance.assign(tf.ones_like(batch_norm.moving_variance))

    if computational.use_bias:
        weights, biases = computational.get_weights()

        extra_dims = np.arange(len(weights.shape) - 1).tolist()

        expanded_gammas = np.expand_dims(gammas, extra_dims)
        expanded_variances = np.expand_dims(variances, extra_dims)

        weights = (expanded_gammas * weights) / \
                  np.sqrt(expanded_variances + epsilon)
        biases = betas + gammas * ((biases - means) / \
                 np.sqrt(variances + epsilon))

        computational.set_weights([
            weights,
            biases
        ])
    else:
        weights = computational.get_weights()[0]

        extra_dims = np.arange(len(weights.shape) - 1).tolist()

        expanded_gammas = np.expand_dims(gammas, extra_dims)
        expanded_variances = np.expand_dims(variances, extra_dims)

        weights = (expanded_gammas * weights) / \
                  np.sqrt(expanded_variances + epsilon)

        computational.set_weights([weights])


def _is_sequential(graph: np.ndarray):
    print(graph)
    print(graph[:,1:])
    print(np.sum(graph[:,1:], axis=0))
    return np.all(np.sum(graph[:,1:], axis=0) == 1)


def fuse_batchnorm(model: Model) -> Model:
    graph = infer_graph_structure(model)

    if not _is_sequential(graph):
        logger.warning(('Unable to clone Functional model. Original model '
                        'will be modified'))
    else:
        clone = clone_model(model)
        clone.set_weights(model.get_weights())
        model = clone


    layers = model.layers
    batch_norms = [i for i in range(len(layers)) \
                  if isinstance(layers[i], BatchNormalization)]

    for i in batch_norms:
        inputs = [layers[idx] for idx in np.where(graph[:,i] == 1)[0]]

        if len(inputs) != 1:
            raise ValueError('Unable to fuse BatchNormalization layer that '
                             'has multiple inputs')

        input_layer = inputs[0]

        if not isinstance(input_layer, (Conv2D, Conv3D, Dense)):
            raise ValueError('Unable to fuse BatchNormalization layer that '
                             'does not have a conv or dense layer as input')

        batch_norm_layer = layers[i]

        _fuse_layers(input_layer, batch_norm_layer)

    return model
