"""Contains tests for the Conv2D-layer LRP implementations"""

import numpy as np

from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Input

from explainability.layers import Conv2DLRP

def test_conv2d_lrp():
    input = Input((4, 4, 2))
    layer = Conv2D(2, (3, 3), use_bias=False, padding='SAME')
    x = layer(input)
    model = Model(input, x)
    weights = np.asarray([
            [[[1, 0], [0, 0]], [[0, 0], [0, 1]], [[0, 0], [0, 0]]],
            [[[0, 1], [0, 0]], [[1, 1], [0, 1]], [[0, 1], [0, 0]]],
            [[[0, 0], [0, 0]], [[0, 0], [0, 1]], [[1, 0], [0, 0]]]

    ])

    layer.set_weights([np.asarray(weights)])

    l = Conv2DLRP(model.layers[1])([input, np.ones((1, 4, 4, 2)) * 2])
    explainer = Model(input, l)

    channels = [np.arange(1, 17).reshape((1, 4, 4, 1)),
                np.arange(0.5, 16.5, 1).reshape((1, 4, 4, 1))]
    input = np.concatenate(channels, axis=-1)

    explanations = explainer(input)

    expected = np.asarray([
        [
            [[0.800, 0.165], [1.664, 0.317], [2.219, 0.401], [2.888, 0.575]],
            [[1.698, 1.694], [3.878, 1.353], [4.083, 1.244], [3.730, 1.494]],
            [[2.277, 1.430], [3.901, 1.167], [3.803, 1.144], [3.195, 1.409]],
            [[2.941, 1.082], [3.848, 0.876], [3.607, 0.863], [3.163, 1.073]]
        ]
    ])

    assert np.allclose(expected, explanations, 1e-2), \
        'Conv2DLRP does not return the correct explanations'
