import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import Activation, Add, Dense, Dropout, Input, \
                                    Subtract

from typing import Union

from ..layers import get_lrp_layer, DenseLRP, StandardLRPLayer
from ..utils import infer_graph_structure, topological_sort
from ..utils.strategies import LRPStrategy


class RestructuredLayerwiseRelevancePropagator(Model):
    def __init__(self, model: Model, *, layer: Union[int, str], idx: int,
                 bottleneck: int, epsilon: float = None, gamma: float = None,
                 alpha: float = None, beta: float = None,
                 strategy: LRPStrategy = None, threshold: bool = False,
                 name: str = 'LRP'):
        intermediate_layers = model.layers[bottleneck + 1:layer]

        for l in intermediate_layers:
            if not isinstance(l, (Activation, Dropout)):
                raise NotImplementedError(('Restructured LRP with layers '
                                           'between bottleneck and target is '
                                           'not implemented'))

        if strategy is not None:
            assert epsilon is None, \
                'Unable to instantiate LRP with both epsilon and strategy'

            assert gamma is None, \
                'Unable to instantiate LRP with both gamma and strategy'

            assert alpha is None, \
                'Unable to instantiate LRP with both alpha/beta and strategy'

        z = model.layers[bottleneck]

        assert isinstance(z, Dense), \
            ('Unable to instantiate RestructuredLRP with non-Dense '
             'bottleneck layer')
        assert z.activation is None or z.activation.__name__ == 'linear', \
            ('Unable to instantiate RestructuredLRP with bottleneck layer '
             'with a built-in activation')

        z = z.output
        a = Input((z.shape[-1]), name=f'{name}/a')

        first_component = Subtract(name=f'{name}/z-a')([z, a])
        first_component = Activation(
            'relu',
             name=f'{name}/z-a/relu'
        )(first_component)
        second_component_zeros = tf.zeros_like(z, name=f'{name}/-z/zeros')
        second_component = Subtract(
            name=f'{name}/-z'
        )([second_component_zeros, z])
        third_component = Add(name=f'{name}/a-z')([second_component, a])
        second_component = Activation(
            'relu',
            name=f'{name}/-z/relu'
        )(second_component)
        third_component = Activation(
            'relu',
            name=f'{name}/a-z/relu'
        )(third_component)

        restructured = Subtract(
            name=f'{name}/restructuring/second_component'
        )([second_component, third_component])
        restructured = Add(
            name=f'{name}/restructuring'
        )([first_component, restructured])

        model_inputs = [model.input, a]

        if threshold:
            t = Input((z.shape[-1]), name=f'{name}/threshold')
            threshold_zeros = tf.zeros_like(t, name=f'{name}/threshold/zeros')
            absolute = tf.abs(restructured,
                              name=f'{name}/restructuring/absolute')
            restructured = tf.where(absolute > t, restructured,
                                    threshold_zeros,
                                    name=f'{name}/restructuring/thresholded')
            model_inputs.append(t)

        output = Dense(
            model.layers[layer].output.shape[-1],
            name=f'{name}/output'
        )(restructured)

        restructured = Model(model_inputs, output)
        restructured.layers[-1].set_weights(model.layers[layer].get_weights())

        indexes = tf.range(output.shape[-1], name=f'{name}/output/indexes')
        mask = indexes == idx
        zeros = tf.zeros_like(output, name=f'{name}/output/zeros')
        masked_output =  tf.where(mask, output, zeros,
                                  name=f'{name}/output/masked')

        output_lrp = DenseLRP(
            restructured.layers[-1],
            name=f'{name}/output/lrp'
        )([restructured.layers[-1].input, masked_output])

        if len(output.shape) != 2:
            raise NotImplementedError('Unable to handle non-flat target layers')

        relevances = {
            z.ref(): output_lrp
        }

        model = Model(model.input, z)
        dependencies = infer_graph_structure(model)
        order = topological_sort(dependencies)
        order = order[::-1]

        layers = [model.layers[i] for i in order]

        for i in range(len(layers)):
            layer = layers[i]
            layer_name = f'{name}/{len(layers) - (i + 1)}'
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

        input = model.input
        outputs = relevances[input.ref()]
        inputs = [input, a]

        if threshold:
            inputs.append(t)

        super().__init__(model_inputs, outputs)

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

RestructuredLRP = RestructuredLayerwiseRelevancePropagator
