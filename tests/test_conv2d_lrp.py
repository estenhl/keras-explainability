"""Contains tests for the Conv2D-layer LRP implementations"""

import numpy as np

from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Input

from explainability.layers import Conv2DLRP

# def test_conv2d_lrp():
#     input = Input((4, 4, 2))
#     layer = Conv2D(2, (3, 3), use_bias=False, padding='SAME')
#     x = layer(input)
#     model = Model(input, x)
#     weights = np.asarray([
#             [[[1, 0], [0, 0]], [[0, 0], [0, 1]], [[0, 0], [0, 0]]],
#             [[[0, 1], [0, 0]], [[1, 1], [0, 1]], [[0, 1], [0, 0]]],
#             [[[0, 0], [0, 0]], [[0, 0], [0, 1]], [[1, 0], [0, 0]]]

#     ])

#     layer.set_weights([np.asarray(weights)])

#     l = Conv2DLRP(model.layers[1])([input, np.ones((1, 4, 4, 2)) * 2])
#     explainer = Model(input, l)

#     channels = [np.arange(1, 17).reshape((1, 4, 4, 1)),
#                 np.arange(0.5, 16.5, 1).reshape((1, 4, 4, 1))]
#     input = np.concatenate(channels, axis=-1)

#     explanations = explainer(input)

#     expected = np.asarray([
#         [
#             [[0.800, 0.165], [1.664, 0.317], [2.219, 0.401], [2.888, 0.575]],
#             [[1.698, 1.694], [3.878, 1.353], [4.083, 1.244], [3.730, 1.494]],
#             [[2.277, 1.430], [3.901, 1.167], [3.803, 1.144], [3.195, 1.409]],
#             [[2.941, 1.082], [3.848, 0.876], [3.607, 0.863], [3.163, 1.073]]
#         ]
#     ])

#     assert np.allclose(expected, explanations, 1e-2), \
#         'Conv2DLRP does not return the correct explanations'

#     assert np.allclose(expected, explanations, 1e-2), \
#         'Conv2D with alpha/beta does not return the correct explanations'

# def test_conv2d_flat():
#     input = Input((4, 4, 1))
#     x = Conv2D(1, (3, 3), activation=None, padding='SAME')(input)
#     model = Model(input, x)

#     model.layers[-1].set_weights([
#         np.asarray([
#             [[[-1.]], [[1.]], [[-1.]]],
#             [[[1.]], [[-1.]], [[1.]]],
#             [[[-1.]], [[1.]], [[-1.]]]
#         ]),
#         np.zeros(1)
#     ])

#     data = np.asarray([
#         [-5., 4., -3., 2.],
#         [-1., 0., 1., -2.],
#         [3., -4., 5., -6.],
#         [7., -8., 9., -10.]
#     ]).reshape((1, 4, 4, 1))

#     R = np.asarray([
#         [-1., 2., 3., 4.],
#         [5., 0., 7., 8.],
#         [9., 10., -11., 12.],
#         [13., 14., 15., -16.]
#     ]).reshape((1, 4, 4, 1))

#     explainer = Conv2DLRP(model.layers[-1], flat=True)
#     explanations = explainer([data, R]).numpy()

#     assert np.allclose(0.9166666, explanations[0,0,0,0], atol=1e-5), \
#         'Conv2DLRP with flat=True does not return the correct explanations'
#     assert np.allclose(4.833333, explanations[0,2,2,0], atol=1e-5), \
#         'Conv2DLRP with flat=True does not return the correct explanations'

# def test_conv2d_alpha_1_beta_0():
#     input = Input((4, 4, 1))
#     x = Conv2D(1, (3, 3), padding='VALID')(input)
#     model = Model(input, x)

#     weights = np.asarray([1 if i % 2 == 0 else -1 for i in range(9)])
#     weights = np.reshape(weights, (3, 3, 1, 1))

#     model.layers[-1].set_weights([
#         weights,
#         np.zeros(1)
#     ])

#     data = np.asarray([x if x < 8 else -x for x in np.arange(16)])
#     data = data.astype(np.float32)
#     data = np.reshape(data, (1, 4, 4, 1))

#     explainer = Conv2DLRP(model.layers[-1], alpha=1, beta=0)
#     R = np.asarray([[1.0, 2.0, 3.0, 4.0]])
#     R = np.reshape(R, (2, 2, 1))
#     explanations = explainer([data, R])

#     expected = np.asarray([
#         [0, 2/20, 2/16, 6/20],
#         [12/41, 5/16+20/46, 12/20+18/41, 28/46],
#         [24/41, 9/16+36/46, 1+30/41, 44/46],
#         [0, 39/41, 56/46, 0]
#     ])
#     expected = np.reshape(expected, (1, 4, 4, 1))

#     assert np.allclose(expected, explanations, atol=1e-5), \
#         'Conv2D with alpha=1, beta=0 does not return the correct explanations'

# def test_conv2d_alpha_2_beta_1():
#     input = Input((4, 4, 1))
#     x = Conv2D(1, (3, 3), padding='VALID')(input)
#     model = Model(input, x)

#     weights = np.asarray([1 if i % 2 == 0 else -1 for i in range(9)])
#     weights = np.reshape(weights, (3, 3, 1, 1))

#     model.layers[-1].set_weights([
#         weights,
#         np.zeros(1)
#     ])

#     data = np.asarray([x if x < 8 else -x for x in np.arange(16)])
#     data = data.astype(np.float32)
#     data = np.reshape(data, (1, 4, 4, 1))

#     explainer = Conv2DLRP(model.layers[-1], alpha=2, beta=1)
#     R = np.asarray([[1.0, 2.0, 3.0, 4.0]])
#     R = np.reshape(R, (2, 2, 1))
#     explanations = explainer([data, R])

#     expected = np.asarray([
#         [0, -1/29+4/20, 4/16-4/34, 12/20],
#         [-4/29+24/41, 10/16-10/34-15/40+40/46, -6/29+24/20+36/41-24/44, -14/34+56/46],
#         [-8/29+48/41, 18/16-18/34-27/40+72/46, -10/29+40/20+60/41-40/44, -22/34+88/46],
#         [-36/40, 78/41-52/44, -42/40+112/46, -60/44]
#     ])
#     expected = np.reshape(expected, (1, 4, 4, 1))

#     assert np.allclose(expected, explanations, atol=1e-5), \
#         'Conv2D with alpha=1, beta=0 does not return the correct explanations'


# def test_conv2d_bias():
#     input = Input((3, 3, 1))
#     x = Conv2D(1, (3, 3), activation=None, padding='VALID')(input)
#     model = Model(input, x)

#     weights = np.reshape(np.arange(1, 10).astype(np.float32), (3, 3, 1, 1))
#     biases = np.ones(1) * 100

#     model.layers[-1].set_weights([
#         weights,
#         biases
#     ])

#     data = np.reshape(np.arange(1, 10).astype(np.float32), (1, 3, 3, 1))

#     explainer = Conv2DLRP(model.layers[-1])
#     explanations = explainer([data, [[385.]]])

#     expected = np.asarray([[
#         [[1.], [4.], [9.]],
#         [[16.], [25.], [36.]],
#         [[49.], [64.], [81.]]
#     ]])

#     assert np.array_equal(expected, explanations), \
#         ('Conv2DLRP does not return the correct explanations when the layer '
#          'has bias')


def test_conv2d_bias_ab_negative():
    input = Input((3, 3, 1))
    x = Conv2D(1, (3, 3), activation=None, padding='VALID')(input)
    model = Model(input, x)

    weights = np.reshape(np.arange(1, 10).astype(np.float32), (3, 3, 1, 1))
    biases = np.ones(1) * 100

    model.layers[-1].set_weights([
        weights,
        biases
    ])

    data = np.asarray([[
        [[1.], [-2.], [3.]],
        [[-4.], [5.], [-6.]],
        [[7.], [-8.], [9.]]
    ]])

    print(model.predict(data))

    explainer = Conv2DLRP(model.layers[-1], alpha=2, beta=1)
    explanations = explainer([data, np.asarray([[145.]])])

    expected = np.asarray([
        [1/165.*2*145, -4/120.*145, 9/165.*2*145],
        [-16/120.*145., 25/165.*2*145., -36/120*145.],
        [49/165.*2*145, -64/120.*145, 81/165.*2*145]
    ])
    expected = np.reshape(expected, (1, 3, 3, 1))

    print(expected)
    print(explanations)

    assert np.allclose(expected, explanations, atol=1e-5), \
        ('Conv2DLRP with alpha=2, beta=1 and negative values does not return '
         'the correct explanations when the layer has bias')
