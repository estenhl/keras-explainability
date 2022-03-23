import tensorflow as tf
from explainability.layers.layer import StandardLRPLayer

from tensorflow.keras import Model

from typing import Union

from .utils import remove_softmax
from ..layers import get_lrp_layer, StandardLRPLayer
from ..utils import infer_graph_structure, topological_sort
from ..utils.strategies import LRPStrategy


class LayerwiseRelevancePropagator(Model):
    def __init__(self, model: Model, *, layer: Union[int, str], idx: int,
                 epsilon: float = None, gamma: float = None,
                 alpha: float = None, beta: float = None,
                 ignore_input: bool = False, strategy: LRPStrategy = None,
                 name: str = 'LRP'):
        model = remove_softmax(model)

        if idx < 0:
            raise NotImplementedError(('Negative indexing for layers not '
                                       'implemented'))

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

        if len(output.shape) != 2:
            raise NotImplementedError('Unable to handle non-flat target layers')

        indexes = tf.range(output.shape[-1], name=f'{name}/output/indexes')
        mask = indexes == idx
        zeros = tf.zeros_like(output, name=f'{name}/output/zeros')
        masked_output =  tf.where(mask, output, zeros,
                                  name=f'{name}/output/masked')
        relevances = {
            output.ref(): masked_output
        }

        layers = [model.layers[i] for i in order]

        for i in range(len(layers)):
            layer = layers[i]
            layer_name = f'{name}/{i}'
            inputs = layer.input
            outputs = layer.output
            R = relevances[outputs.ref()]
            relevance = get_lrp_layer(
                layer,
                epsilon=epsilon,
                gamma=gamma,
                alpha=alpha,
                beta=beta,
                name=layer_name
            )([layer.input, R])

            if not isinstance(relevance, list):
                inputs = [inputs]
                relevance = [relevance]

            assert len(relevance) == len(inputs), \
                (f'Layer {layer} has {len(inputs)} inputs but '
                 f'{len(relevances)} relevances')

            for i in range(len(inputs)):
                relevances[inputs[i].ref()] = relevance[i]

        inputs = model.inputs
        outputs = [relevances[layer.ref()] for layer in inputs] \
                  if isinstance(inputs, list) else \
                  relevances[inputs.ref()]

        super().__init__(inputs, outputs)

        standard_lrp_layers = []

        for i in range(len(self.layers)):
            if isinstance(self.layers[i], StandardLRPLayer):
                standard_lrp_layers.append(self.layers[i])

        if strategy is not None:
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

LRP = LayerwiseRelevancePropagator
