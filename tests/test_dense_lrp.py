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

    explainer = LayerwiseRelevancePropagator(model, layer=2, idx=0)
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
    l = DenseLRP(model.layers[1], epsilon=1.)([input, relevances])

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

def test_dense_lrp_relu():
    input = Input((3,))
    x = Dense(3, activation='relu')(input)
    x = Dense(1, activation=None)(x)
    model = Model(input, x)
    model.layers[1].set_weights([
        np.asarray([
            [1, 1, -1],
            [-1, -1, -1],
            [1, -1, 1]
        ]),
        np.zeros(3)
    ])
    model.layers[2].set_weights([np.reshape(np.arange(1, 4), (3, 1)), np.zeros(1)])

    explainer = LayerwiseRelevancePropagator(model, layer=2, idx=0)
    explanations = explainer(np.asarray([[1, 2, -3]], dtype=np.float32))

    assert np.array_equal([[2, -4, 6]], explanations), \
        'DenseLRP does not handle built-in relu'

def test_dense_alpha_beta():
    input = Input((3,))
    x = Dense(3, activation='relu')(input)
    x = Dense(1, activation=None)(x)
    model = Model(input, x)
    model.layers[1].set_weights([
        np.asarray([
            [1, 1, -1],
            [-1, -1, -1],
            [1, -1, 1]
        ]),
        np.zeros(3)
    ])
    model.layers[2].set_weights([np.reshape(np.arange(1, 4), (3, 1)), np.zeros(1)])

    for a in range(2, 5):
        b = a - 1
        explainer = LayerwiseRelevancePropagator(model, layer=2, idx=0,
                                                 alpha=a, beta=b)
        output = explainer(np.asarray([[1, 2, -3]]))
        m = Model(model.input, model.layers[1].output)

        assert np.array_equal([[a**2, -4 * b * a, 3*a**2]], output), \
            (f'LRP with alpha={a} and beta={b} does not return the correct '
             'explanations')

def test_dense_alpha_1_beta_0_model():
    input = Input((3,))
    x = Dense(3, activation='relu')(input)
    x = Dense(1, activation=None)(x)
    model = Model(input, x)
    model.layers[1].set_weights([
        np.asarray([
            [1, 1, -1],
            [-1, -1, -1],
            [1, -1, 1]
        ]),
        np.zeros(3)
    ])
    model.layers[2].set_weights([np.reshape(np.arange(1, 4), (3, 1)), np.zeros(1)])

    explainer = LayerwiseRelevancePropagator(model, layer=2, idx=0, alpha=1, beta=0)
    explanations = explainer(np.asarray([[1, 2, -3]], dtype=np.float32))

    assert np.array_equal([[1, 0, 3]], explanations), \
        'DenseLRP does not handle alpha/beta'

def test_dense_lrp_b():
    input = Input((3,))
    x = Dense(1, activation=None)(input)
    model = Model(input, x)
    model.layers[1].set_weights([
        np.asarray([[1], [2], [3]]),
        np.zeros(1)
    ])

    explainer = DenseLRP(model.layers[1])
    explanations = explainer([np.asarray([[1., 2., 3.]]),
                              np.asarray([14.])])[0].numpy()

    assert np.array_equal([1., 4., 9.], explanations), \
        'Default DenseLRP returns the wrong explanations'

    explainer = DenseLRP(model.layers[1], b=True)
    explanations = explainer([np.asarray([[1., 2., 3.]]),
                              np.asarray([14.])])[0].numpy()

    expected = np.asarray([14./6, (14.*2)/6, (14.*3)/6])

    assert np.allclose(expected, explanations, atol=1e-8), \
        'DenseLRP with b=True returns the wrong explanations'


def test_dense_flat():
    inputs = Input((4,))
    x = Dense(2, activation=None)(inputs)
    model = Model(inputs, x)

    model.layers[1].set_weights([
        np.asarray([[-1, 0], [1, 2], [3, 4], [5, 6]]),
        np.zeros(2)
    ])

    explainer = DenseLRP(model.layers[-1])
    explanations = explainer([np.asarray([[1., 2., 3., 4.]]),
                              np.asarray([[10., 20.]])]).numpy()

    expected = np.asarray([[-1/3, 8/3, 9, 56/3]])

    assert np.allclose(expected, explanations, atol=1e-5), \
        'Default DenseLRP returns the wrong explanations'

    explainer = DenseLRP(model.layers[-1], flat=True)
    explanations = explainer([np.asarray([[1., 2., 3., 4.]]),
                              np.asarray([[10., 20.]])]).numpy()

    expected = np.ones((1, 4)) * 30/4

    assert np.allclose(expected, explanations, atol=1e-5), \
        'DenseLRP with flat=True returns the wrong explanations'


def test_dense_bias_ab_negative():
    inputs = Input((4,))
    x = Dense(1, activation=None)(inputs)
    model = Model(inputs, x)

    model.layers[1].set_weights([
        np.asarray([[-1.], [2.], [-3.], [4.]]),
        np.asarray([4.])
    ])

    data = np.asarray([[-1., -2., 3., 4.]])

    print(model.predict(data))

    explainer = DenseLRP(model.layers[-1], alpha=2, beta=1)
    explanations = explainer([data, [[8.]]])

    expected = np.asarray([[16/21., -32/13., -72/13., 256/21]])

    assert np.allclose(expected, explanations, atol=1e-5), \
        ('Dense layer with bias and alpha=2, beta=1 and negative values in '
         'both input and weights does not return the correct explanation')


