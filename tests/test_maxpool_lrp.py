import numpy as np

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, GlobalMaxPooling2D, \
                                    GlobalMaxPooling3D, MaxPooling2D, \
                                    MaxPooling3D

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
        strategy='winner-takes-all'
    )([input, relevances])
    explainer = Model(input, l)
    explanations = explainer.predict(values)

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
        strategy='winner-takes-all'
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

    assert np.array_equal(expected, explanations), \
        ('MaxPoolingLRP does not return the correct explanations for '
         'global 2D MaxPooling')

def test_maxpool_3d_lrp():
    input = Input((2, 2, 2, 2))
    layer = MaxPooling3D((2, 2, 2), padding='valid')(input)
    model = Model(input, layer)

    values = np.asarray([[
        [
            [[1, 0], [0, 2]],
            [[0, 0], [0, 0]]
        ],
        [
            [[0, 0], [0, 0]],
            [[0, 0], [0, 0]]
        ]
    ]])

    print(model.predict(values))

    l = MaxPoolingLRP(
        model.layers[1],
        strategy='winner-takes-all'
    )([input, np.asarray([[3., 5.]])])
    explainer = Model(input, l)
    explanations = explainer(values).numpy()

    expected = np.asarray([[
        [
            [[3, 0], [0, 5]],
            [[0, 0], [0, 0]]
        ],
        [
            [[0, 0], [0, 0]],
            [[0, 0], [0, 0]]
        ]
    ]])

    assert np.array_equal(expected, explanations), \
        ('MaxPoolingLRP does not return the correct explanations for '
         'non-global 3D MaxPooling')

def test_global_maxpool_3d_lrp():
    input = Input((2, 2, 2, 2))
    layer = GlobalMaxPooling3D()(input)
    model = Model(input, layer)

    values = np.asarray([[
        [
            [[1, 0], [0, 2]],
            [[0, 0], [0, 0]]
        ],
        [
            [[0, 0], [0, 0]],
            [[0, 0], [0, 0]]
        ]
    ]])

    l = MaxPoolingLRP(
        model.layers[1],
        strategy='winner-takes-all'
    )([input, np.asarray([[3., 5.]])])
    explainer = Model(input, l)
    explanations = explainer(values).numpy()

    expected = np.asarray([[
        [
            [[3, 0], [0, 5]],
            [[0, 0], [0, 0]]
        ],
        [
            [[0, 0], [0, 0]],
            [[0, 0], [0, 0]]
        ]
    ]])

    assert np.array_equal(expected, explanations), \
        ('MaxPoolingLRP does not return the correct explanations for '
         'global 3D MaxPooling')

def test_maxpool_redistribute_2d():
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
    ], dtype=np.float32)

    relevances = np.asarray([
        [[12, 21], [4, 8]],
        [[15, 3], [2, 5]]
    ], dtype=np.float32)

    l = MaxPoolingLRP(
        model.layers[1],
        strategy='redistribute'
    )([input, relevances])
    explainer = Model(input, l)
    explanations = explainer(values).numpy()

    assert np.array_equal(values, explanations), \
        ('MaxPoolingLRP does not return the correct explanations when using '
         'the redistribution strategy')

def test_maxpool_flat_2d():
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
    ], dtype=np.float32)

    relevances = np.asarray([
        [[12, 21], [4, 8]],
        [[15, 3], [2, 5]]
    ], dtype=np.float32)

    l = MaxPoolingLRP(
        model.layers[1],
        strategy='flat'
    )([input, relevances])
    explainer = Model(input, l)
    explanations = explainer.predict(values)

    expected = np.asarray([
        [
            [[12, 21], [12, 21], [4, 8], [4, 8]],
            [[12, 21], [12, 21], [4, 8], [4, 8]],
            [[15, 3], [15, 3], [2, 5], [2, 5]],
            [[15, 3], [15, 3], [2, 5], [2, 5]]
        ]
    ], dtype=np.float32) / 4.0

    assert np.array_equal(expected, explanations), \
        ('MaxPoolingLRP does not return the correct explanations when using '
         'the flat strategy')

