"""Contains tests for the Dense-layer LRP implementations"""

import numpy as np

from tensorflow.keras.initializers import Constant
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input

from explainability.layers import DenseLRP
from explainability import LayerwiseRelevancePropagator


def test_dense_lrp():
    """Tests that the regular Dense-layer LRP implementation returns
    the expected explanations
    """
    input = Input((2,))
    layer = Dense(3, use_bias=False,
                  kernel_initializer=Constant(value=[[1., 2., 3.],
                                                     [2., 3., 4.]]))
    model = Model(input, layer(input))

    relevances = np.asarray([[24., 52., 90.]], dtype=np.float32)
    l = DenseLRP(model.layers[1])([input, relevances])

    explainer = Model(input, l)
    explanations = explainer(np.asarray([[2., 3.]], dtype=np.float32))

    assert np.array_equal(explanations, [[52., 114.]]), \
        'Default DenseLRP does not return the correct explanations'

def test_sequential_dense_lrp():
    """Tests that two regular Dense-layer LRPs returns the expected
    explanations
    """
    input = Input((2,))
    first = Dense(3, use_bias=False,
                  kernel_initializer=Constant(value=[[1., 2., 3.], [2., 3., 4.]]))(input)
    second = Dense(1, use_bias=False,
                   kernel_initializer=Constant(value=[[3.], [4.], [5.]]))(first)
    model = Model(input, second)

    l1 = DenseLRP(model.layers[2], name='l1')([model.layers[1].output, model.output])
    l2 = DenseLRP(model.layers[1], name='l2')([model.layers[0].output, l1])

    explainer = Model(input, l2)
    explanations = explainer(np.asarray([[2., 3.]], dtype=np.float32))

    assert np.array_equal(explanations, [[52., 114.]]), \
        ('Two default DenseLRPs in sequence does not return the '
         'correct explanations')

def test_dense_model():
    """Tests that a model comprised of only dense layers returns the
    expected result
    """
    input = Input((2,))
    first = Dense(3, use_bias=False,
                  kernel_initializer=Constant(value=[[1., 2., 3.], [2., 3., 4.]]))(input)
    second = Dense(1, use_bias=False,
                   kernel_initializer=Constant(value=[[3.], [4.], [5.]]))(first)
    model = Model(input, second)

    explainer = LayerwiseRelevancePropagator(model)
    explanations = explainer(np.asarray([[2., 3.]], dtype=np.float32))

    assert np.array_equal(explanations, [[52., 114.]]), \
        ('Two default DenseLRPs in a LRP-wrapper does not return the '
         'correct explanations')

def test_dense_lrp_epsilon():
    """Tests that the epsilon-variant of the Dense-layer LRP
    implementation returns the expected explanations
    """
    input = Input((2,))
    layer = Dense(3, use_bias=False,
                  kernel_initializer=Constant(value=[[1., 2., 3.],
                                                     [2., 3., 4.]]))
    model = Model(input, layer(input))

    relevances = np.asarray([[24., 52., 90.]], dtype=np.float32)
    l = DenseLRP(model.layers[1], epsilon=1)([input, relevances])

    explainer = Model(input, l)
    explanations = explainer(np.asarray([[2., 3.]], dtype=np.float32))

    assert np.allclose(explanations, [[48.611, 106.270]], 1e-3), \
        'Epsilon-DenseLRP does not return the correct explanations'

def test_dense_lrp_gamma():
    """Tests that the gamma-variant of the Dense-layer LRP
    implementation returns the expected explanations
    """
    input = Input((2,))
    layer = Dense(3, use_bias=False,
                  kernel_initializer=Constant(value=[[1., -2., 3.],
                                                     [3., 4., 5.]]))
    model = Model(input, layer(input))

    relevances = np.asarray([[3., 4., 5.]], dtype=np.float32)
    l = DenseLRP(model.layers[1], gamma=0.25)([input, relevances])

    explainer = Model(input, l)
    explanations = explainer(np.asarray([[1., 2.]], dtype=np.float32))

    assert np.allclose(explanations, [[0.582, 11.417]], 1e-3), \
        'Gamma-DenseLRP does not return the correct explanations'

def test_dense_lrp_epsilon_and_gamma():
    """Tests that instantiating a DenseLRP with both epsilon and gamma
    raises an exception
    """
    input = Input((2,))
    layer = Dense(3, use_bias=False,
                  kernel_initializer=Constant(value=[[1., -2., 3.],
                                                     [3., 4., 5.]]))
    model = Model(input, layer(input))

    exception = False

    try:
        DenseLRP(model.layers[1], epsilon=0.25, gamma=0.25)
    except Exception:
        exception = True

    assert exception, \
        ('Instantiating a DenseLRP-layer with both epsilon and gamma does not '
         'raise an exception')
