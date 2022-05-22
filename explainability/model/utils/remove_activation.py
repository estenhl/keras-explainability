from tensorflow.keras import Model
from tensorflow.keras.layers import Dense

from typing import List


def remove_activation(model: Model, activations: List[str]):
    prev = model.layers[-2].output
    output = model.layers[-1]

    if not isinstance(output, Dense):
        return model
    if output.activation.__name__ not in activations:
        return model

    layer = Dense(
        output.units,
        activation=None,
        use_bias=output.use_bias,
        kernel_initializer=output.kernel_initializer,
        bias_initializer=output.bias_initializer,
        kernel_regularizer=output.kernel_regularizer,
        bias_regularizer=output.bias_regularizer,
        activity_regularizer=output.kernel_regularizer,
        kernel_constraint=output.kernel_constraint,
        bias_constraint=output.bias_constraint,
        name=f'{output.name}_without_softmax'
    )(prev)

    model = Model(model.input, layer)

    model.layers[-1].set_weights(output.get_weights())

    return model

def remove_softmax(model: Model) -> Model:
    return remove_activation(model, activations=['softmax'])

def remove_sigmoid(model: Model) -> Model:
    return remove_activation(model, activations=['sigmoid'])
