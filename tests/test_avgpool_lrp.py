import numpy as np

from tensorflow.keras import Model
from tensorflow.keras.layers import AveragePooling2D, AveragePooling3D, GlobalAveragePooling2D, GlobalAveragePooling3D, \
                                    Input

from explainability.layers import AveragePoolingLRP


def test_avgpool_2d_lrp():
    input = Input((4, 4, 2))
    layer = AveragePooling2D((2, 2))(input)
    model = Model(input, layer)

    values = np.asarray([
        [
            [[1, 16], [2, 15], [3, 14], [4, 13]],
            [[5, 12], [6, 11], [7, 10], [8, 9]],
            [[9, 8], [10, 7], [11, 6], [12, 5]],
            [[13, 4], [14, 3], [15, 2], [16, 1]]
        ]
    ])

    relevances = np.asarray([
        [
            [[1., 5.], [2., 6.]],
            [[3., 7.], [4., 8.]]
        ]
    ])
    l = AveragePoolingLRP(
        model.layers[1]
    )([input, relevances])
    explainer = Model(input, l)
    explanations = explainer(values).numpy()

    expected = np.asarray([
        [
            [[1./14, 80./54], [2./14, 75./54], [6./22, 84./46], [8./22, 78./46]],
            [[5./14, 60./54], [6./14, 55./54], [14./22, 60./46], [16./22, 54./46]],
            [[27./46, 56./22], [30./46, 49./22], [44./54, 48./14], [48./54, 40./14]],
            [[39./46, 28./22], [42./46, 21./22], [60./54, 16./14], [64./54, 8./14]]
        ]
    ])

    assert np.allclose(expected, explanations, 1e-3), \
         ('AveragePoolingLRP does not return the correct explanations for '
          'non-global 2D AveragePooling')

def test_avgpool_2d_lrp_invalid_batch_size():
    input = Input((4, 4, 2))
    layer = AveragePooling2D((2, 2))(input)
    model = Model(input, layer)

    values = np.asarray([
        [
            [[1, 16], [2, 15], [3, 14], [4, 13]],
            [[5, 12], [6, 11], [7, 10], [8, 9]],
            [[9, 8], [10, 7], [11, 6], [12, 5]],
            [[13, 4], [14, 3], [15, 2], [16, 1]]
        ]
    ])

    relevances = np.asarray([
        [
            [[1., 5.], [2., 6.]],
            [[3., 7.], [4., 8.]]
        ]
    ])
    l = AveragePoolingLRP(
        model.layers[1],
        batch_size=1
    )([input, relevances])
    explainer = Model(input, l)

    exception = False

    try:
        explainer(np.concatenate([values, values], axis=0)).numpy()
    except Exception:
        exception = True

    assert exception, \
        ('Running AveragePoolingLRP with batch that does not match the given '
         'batch_size does not raise an exception')

def test_global_avgpool_2d_lrp():
    input = Input((4, 4, 2))
    layer = GlobalAveragePooling2D()(input)
    model = Model(input, layer)

    values = np.asarray([
        [
            [[1, 16], [2, 15], [3, 14], [4, 13]],
            [[5, 12], [6, 11], [7, 10], [8, 9]],
            [[9, 8], [10, 7], [11, 6], [12, 5]],
            [[13, 4], [14, 3], [15, 2], [16, 1]]
        ]
    ])

    relevances = np.asarray([
        [
            [[4., 8.]]
        ]
    ])
    l = AveragePoolingLRP(
        model.layers[1]
    )([input, relevances])
    explainer = Model(input, l)
    explanations = explainer(values).numpy()

    expected = values / np.sum(values, axis=(0, 1, 2))
    expected = expected * relevances

    assert np.allclose(expected, explanations, 1e-3), \
         ('AveragePoolingLRP does not return the correct explanations for '
          'global 2D AveragePooling')

def test_avgpool_3d_lrp():
    ksize = 3
    windows = 3
    input = Input((ksize * windows, ksize * windows, ksize * windows, 2))
    layer = AveragePooling3D((ksize, ksize, ksize))(input)
    model = Model(input, layer)

    values = np.ones((1, ksize * windows, ksize * windows, ksize * windows, 2))

    relevances = np.ones((1, windows, windows, windows, 2))*ksize**3
    l = AveragePoolingLRP(
        model.layers[1]
    )([input, relevances])
    explainer = Model(input, l)
    explanations = explainer(values).numpy()

    assert np.allclose(values, explanations, 1e-3), \
         ('AveragePoolingLRP does not return the correct explanations for '
          'non-global 3D AveragePooling')

def test_global_avgpool_3d_lrp():
    ksize = 5
    windows = 1
    input = Input((ksize * windows, ksize * windows, ksize * windows, 2))
    layer = GlobalAveragePooling3D()(input)
    model = Model(input, layer)

    values = np.ones((1, ksize * windows, ksize * windows, ksize * windows, 2))

    relevances = np.ones((1, windows, windows, windows, 2))*ksize**3
    l = AveragePoolingLRP(
        model.layers[1]
    )([input, relevances])
    explainer = Model(input, l)
    explanations = explainer(values).numpy()

    assert np.allclose(values, explanations, 1e-3), \
         ('AveragePoolingLRP does not return the correct explanations for '
          'global 3D AveragePooling')
