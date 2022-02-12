"""Contains tests for the Conv2D-layer LRP implementations"""

import numpy as np

from tensorflow.keras import Model
from tensorflow.keras.layers import Activation, Input, ReLU

from explainability.layers import ReLULRP


def test_relu_lrp():
    input = Input((4,))
    layer = ReLU()(input)
    model = Model(input, layer)

    l = ReLULRP(model.layers[1])([input, np.asarray([1, 2, 3, 4])])
    explainer = Model(input, l)
    explanations = explainer(np.asarray([1, 2, -3, 4])).numpy()

    assert np.array_equal(explanations, [1, 2, 0, 4]), \
        'ReLULRP does not return the expected explanations'

def test_activation_relu_lrp():
    input = Input((4,))
    layer = Activation('relu')(input)
    model = Model(input, layer)

    l = ReLULRP(model.layers[1])([input, np.asarray([1, 2, 3, 4])])
    explainer = Model(input, l)
    explanations = explainer(np.asarray([1, 2, -3, 4])).numpy()

    assert np.array_equal(explanations, [1, 2, 0, 4]), \
        'ReLULRP does not return the expected explanations'
