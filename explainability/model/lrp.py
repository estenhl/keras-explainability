import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import InputLayer, Dense, Lambda, LayerNormalization

from typing import Union

from .utils import fuse_batchnorm, remove_activation
from ..layers import get_lrp_layer, PoolingLRPLayer, StandardLRPLayer
from ..utils import infer_graph_structure, topological_sort
from ..utils.strategies import LRPStrategy


class LayerwiseRelevancePropagator(Model):
    def __init__(self, model: Model, *, layer: Union[int, str], idx: int,
                 include_prediction: bool = False,
                 epsilon: float = None, gamma: float = None,
                 alpha: float = None, beta: float = None,
                 strategy: LRPStrategy = None, name: str = 'LRP'):

        original_output = model.layers[layer].output
        model = Model(model.input, original_output)
        model = remove_activation(model, ['sigmoid', 'softmax'])
        model = fuse_batchnorm(model)

        if idx < 0:
            raise NotImplementedError('Negative node indexes (e.g. idx) not '
                                      'implemented')

        if strategy is not None:
            assert epsilon is None, \
                'Unable to instantiate LRP with both epsilon and strategy'

            assert gamma is None, \
                'Unable to instantiate LRP with both gamma and strategy'

            assert alpha is None, \
                'Unable to instantiate LRP with both alpha/beta and strategy'

        dependencies = infer_graph_structure(model)
        order = topological_sort(dependencies)
        order = order[::-1]
        output = model.output
        output_shape = (1,) + output.shape[1:]

        if len(output.shape) != 2:
            raise NotImplementedError('Unable to handle non-flat target layers')

        indexes = tf.range(output.shape[-1], name=f'{name}/output/indexes')
        mask = indexes == idx
        zeros = tf.zeros(output_shape, name=f'{name}/output/zeros')

        masked_output = Lambda(
            lambda params: tf.where(*params, name=f'{name}_output_masked'),
            name=f'{name}_output_mask_lambda',
            output_shape=output_shape
        )([mask, output, zeros])

        relevances = {
            output.name: masked_output
        }

        layers = [model.layers[i] for i in order]

        i = 0
        while i < len(layers):
            layer = layers[i]
            kwargs = {
                'name': f'{name}_{i}',
                'epsilon': epsilon,
                'gamma': gamma,
                'alpha': alpha,
                'beta':beta
            }

            if isinstance(layer, LayerNormalization):
                if isinstance(layers[i + 1], Dense):
                    inputs = layers[i + 1].input
                    outputs = layer.output
                    R = relevances[outputs.name]
                    kwargs['norm'] = layer
                    layer = layers[i + 1]
                else:
                    raise NotImplementedError('Unable to handle '
                                              'LayerNormalization preceded by '
                                              'something other than a Dense '
                                              'layer.')
                i += 2
            else:
                inputs = layer.input
                outputs = layer.output
                R = relevances[outputs.name]
                i += 1
            relevance = get_lrp_layer(
                layer,
                **kwargs
            )([inputs, R])

            if not isinstance(relevance, list):
                inputs = [inputs]
                relevance = [relevance]

            assert len(relevance) == len(inputs), \
                (f'Layer {layer} has {len(inputs)} inputs but '
                 f'{len(relevances)} relevances')

            for j in range(len(inputs)):
                if isinstance(inputs[j], list):
                    if len(inputs[j]) != 0:
                        raise ValueError('Inputs is a nested list where the '
                                         'inner list is not empty')
                    if not isinstance(layer, InputLayer):
                        raise ValueError('Inputs is a nested list but layer '
                                         f'is not an InputLayer ({layer})')
                    name = layer.name
                else:
                    name = inputs[j].name

                relevances[name] = relevance[j]

        inputs = model.inputs
        outputs = [relevances[layer.name] for layer in inputs] \
                  if isinstance(inputs, list) else \
                  relevances[inputs.name]

        if include_prediction:
            outputs = [original_output] + outputs

        super().__init__(inputs, outputs)

        if strategy is not None:
            standard_lrp_layers = []
            pooling_lrp_layers = []

            for i in range(len(self.layers)):
                if isinstance(self.layers[i], StandardLRPLayer):
                    standard_lrp_layers.append(self.layers[i])
                elif isinstance(self.layers[i], PoolingLRPLayer):
                    pooling_lrp_layers.append(self.layers[i])

            if strategy.layers is not None:
                assert len(strategy.layers) == len(standard_lrp_layers), \
                    ('Unable to instantiate LRP with strategy that does not have '
                    'a configuration for each standard LRP layer')

                configurations = strategy.layers[::-1]

                for i in range(len(configurations)):
                    for variable in configurations[i]:
                        assert hasattr(standard_lrp_layers[i], variable), \
                            ('LRPStrategy contains invalid configuration variable '
                            f'{variable} for layer '
                            f'{standard_lrp_layers[i].__class__.__name__}')
                        setattr(standard_lrp_layers[i], variable,
                                configurations[i][variable])

            if strategy.pooling is not None:
                assert len(strategy.pooling) == len(pooling_lrp_layers), \
                    ('Unable to instantiate LRP with strategy that does not have '
                    'a configuration for each standard LRP layer')

                configurations = strategy.pooling[::-1]

                for i in range(len(configurations)):
                    pooling_lrp_layers[i].strategy = configurations[i]['strategy']

LRP = LayerwiseRelevancePropagator

