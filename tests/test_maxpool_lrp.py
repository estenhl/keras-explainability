import numpy as np

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, GlobalMaxPooling2D, MaxPooling2D

from explainability.layers import MaxPoolingLRP


def test_maxpool_2d_lrp():
    input = Input((4, 4, 2))
    layer = MaxPooling2D((2, 2))(input)
    model = Model(input, layer)

    values = np.asarray([
        [
            [[2, 4], [6, 5], [3, 1], [0, 0]],
            [[4, 3], [0, 9], [1, 7], [0, 0]],
            [[8, 2], [7, 1], [2, 5], [0, 0]],
            [[0, 0], [0, 0], [0, 0], [0, 0]]
        ]
    ])

    relevances = np.ones((1, 2, 2, 2)) * np.reshape([1, 2], (1, 1, 2))
    l = MaxPoolingLRP(
        model.layers[1],
        strategy='winner-take-all'
    )([input, relevances])
    explainer = Model(input, l)
    explanations = explainer(values).numpy()

    expected = np.asarray([
        [
            [[0., 0.], [1., 0.], [1., 0.], [0., 0.]],
            [[0., 0.], [0., 2.], [0., 2.], [0., 0.]],
            [[1., 2.], [0., 0.], [1., 2.], [0., 0.]],
            [[0., 0.], [0., 0.], [0., 0.], [0., 0.]]
        ]
    ])

    assert np.array_equal(expected, explanations), \
        ('MaxPoolingLRP does not return the correct explanations for '
         'non-global 2D MaxPooling')

def test_global_maxpool_2d_lrp():
    input = Input((4, 4, 2))
    layer = GlobalMaxPooling2D()(input)
    model = Model(input, layer)

    values = np.asarray([
        [
            [[2, 4], [6, 5], [3, 1], [0, 0]],
            [[4, 3], [0, 9], [1, 7], [0, 0]],
            [[8, 2], [7, 1], [2, 5], [0, 0]],
            [[0, 0], [0, 0], [0, 0], [0, 0]]
        ]
    ])

    l = MaxPoolingLRP(
        model.layers[1],
        strategy='winner-take-all'
    )([input, np.asarray([[1., 2.]])])
    explainer = Model(input, l)
    explanations = explainer(values).numpy()

    expected = np.asarray([
        [
            [[0., 0.], [0., 0.], [0., 0.], [0., 0.]],
            [[0., 0.], [0., 2.], [0., 0.], [0., 0.]],
            [[1., 0.], [0., 0.], [0., 0.], [0., 0.]],
            [[0., 0.], [0., 0.], [0., 0.], [0., 0.]]
        ]
    ])

    print(explanations)
    print(explanations.shape)

    assert np.array_equal(expected, explanations), \
        ('MaxPoolingLRP does not return the correct explanations for '
         'non-global 2D MaxPooling')
