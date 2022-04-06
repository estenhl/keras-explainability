"""Contains tests for the Conv2D-layer LRP implementations"""

import numpy as np

from tensorflow.keras import Model
from tensorflow.keras.layers import Conv3D, Flatten, Input

from explainability import LayerwiseRelevancePropagator
from explainability.layers import Conv3DLRP

def test_conv3d_lrp():
    input = Input((4, 4, 4, 1))
    x = Conv3D(2, (3, 3, 3), activation=None, padding='SAME')(input)
    x = Flatten()(x)

    model = Model(input, x)

    weights = np.arange(-27, 27).reshape((3, 3, 3, 1, 2))
    model.layers[1].set_weights([
        weights,
        np.zeros(2)
    ])

    data = np.arange(4*4*4)
    data = np.asarray([x if x % 2 == 0 else -x for x in data])
    data = np.reshape(data, (1, 4, 4, 4, 1))

    explainer = LayerwiseRelevancePropagator(model, layer=2, idx=0,
                                             epsilon=1e-15)
    explanations = explainer(data)

    assert -132 == np.sum(explanations), \
        'Conv3D with a non-existing epsilon returns the wrong explanations'
    assert -1 == explanations[0,0,0,1,0], \
        'Conv3D with a non-existing epsilon returns the wrong explanations'
    assert 20 == explanations[0,0,1,0,0], \
        'Conv3D with a non-existing epsilon returns the wrong explanations'
    assert -35 == explanations[0,0,1,1,0], \
        'Conv3D with a non-existing epsilon returns the wrong explanations'
    assert 272 == explanations[0,1,0,0,0], \
        'Conv3D with a non-existing epsilon returns the wrong explanations'
    assert -323 == explanations[0,1,0,1,0], \
        'Conv3D with a non-existing epsilon returns the wrong explanations'
    assert 460 == explanations[0,1,1,0,0], \
        'Conv3D with a non-existing epsilon returns the wrong explanations'
    assert -525 == explanations[0,1,1,1,0], \
        'Conv3D with a non-existing epsilon returns the wrong explanations'

def test_conv3d_lrp_epsilon():
    input = Input((4, 4, 4, 1))
    x = Conv3D(2, (3, 3, 3), activation=None, padding='SAME')(input)
    x = Flatten()(x)

    model = Model(input, x)

    weights = np.arange(-27, 27).reshape((3, 3, 3, 1, 2))
    model.layers[1].set_weights([
        weights,
        np.zeros(2)
    ])

    data = np.arange(4*4*4)
    data = np.asarray([x if x % 2 != 0 else -x for x in data])
    data = np.reshape(data, (1, 4, 4, 4, 1))

    explainer = LayerwiseRelevancePropagator(model, layer=2, idx=0,
                                             epsilon=1e-1)
    explanations = explainer(data)

    assert np.isclose(131.9000, np.sum(explanations), 1e-5), \
        'Conv3D with epsilon returns the wrong explanations'
    assert np.isclose(0.99924, explanations[0,0,0,1,0], 1e-5), \
        'Conv3D with epsilon returns the wrong explanations'
    assert np.isclose(-19.98486, explanations[0,0,1,0,0], 1e-5), \
        'Conv3D with epsilon returns the wrong explanations'
    assert np.isclose(34.97350, explanations[0,0,1,1,0], 1e-5), \
        'Conv3D with epsilon returns the wrong explanations'
    assert np.isclose(-271.7941, explanations[0,1,0,0,0], 1e-5), \
        'Conv3D with epsilon returns the wrong explanations'
    assert np.isclose(322.7555, explanations[0,1,0,1,0], 1e-5), \
        'Conv3D with epsilon returns the wrong explanations'
    assert np.isclose(-459.65176, explanations[0,1,1,0,0], 1e-5), \
        'Conv3D with epsilon returns the wrong explanations'
    assert np.isclose(524.60254, explanations[0,1,1,1,0], 1e-5), \
        'Conv3D with epsilon returns the wrong explanations'
