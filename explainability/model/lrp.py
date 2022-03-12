import tensorflow as tf

from tensorflow.keras import Model

from typing import Union

from .utils import remove_softmax
from ..layers import LRP


class LayerwiseRelevancePropagator(Model):
    def __init__(self, model: Model, *, epsilon: float = None,
                 gamma: float = None, alpha: float = None, beta: float = None,
                 layer: Union[int, str], idx: int):
        model = remove_softmax(model)

        if idx < 0:
            raise NotImplementedError(('Negative indexing for layers not '
                                       'implemented'))

        activations = [layer.output for layer in model.layers[:layer + 1]]

        assert len(activations[-1].shape) == 2, \
            'Unable to handle non-flat target layers'
        prev = tf.where(tf.range(activations[-1].shape[-1]) == idx,
                         activations[-1], tf.zeros_like(activations[-1]))

        for i in range(len(model.layers)-1, 0, -1):
            input = activations[i-1] if i > 1 \
                    else tf.ones_like(activations[i-1])
            prev = LRP(
                model.layers[i],
                epsilon=epsilon,
                gamma=gamma,
                alpha=alpha,
                beta=beta,
                name=f'{i}'
            )([activations[i-1], prev])

        super().__init__(model.input, prev)
