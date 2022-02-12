"""Contains tests for the Conv2D-layer LRP implementations"""

import numpy as np

from tensorflow.keras import Model
from tensorflow.keras.layers import Reshape, Input

from explainability.layers import ReshapeLRP


def test_reshape_lrp():
    input = Input((4,4))
    layer = Reshape((2, 2, 2, 2))(input)
    model = Model(input, layer)

    l = ReshapeLRP(model.layers[1])([input, np.ones((2, 2, 2, 2))])
    explainer = Model(input, l)
    explanations = explainer(np.reshape(np.arange(4*4), (1, 4, 4))).numpy()

    print(explanations)

    assert np.array_equal(explanations, np.ones((1, 4, 4))), \
        'ReshapeLRP does not return the correct explanations'
