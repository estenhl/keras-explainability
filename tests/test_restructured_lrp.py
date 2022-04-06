import numpy as np
from explainability.layers.layer import StandardLRPLayer
from explainability.utils.strategies.lrp_strategy import LRPStrategy

from tensorflow.keras import Model
from tensorflow.keras.layers import Activation, Conv2D, Dense, Flatten, Input

from explainability import RestructuredLRP


# def test_restructured_lrp():
#     input = Input((3,))
#     x = Dense(3, activation=None)(input)
#     x = Activation('relu')(x)
#     x = Dense(1, activation=None)(x)
#     model = Model(input, x)

#     model.layers[1].set_weights([
#         np.reshape(np.arange(9), (3, 3)),
#         np.zeros(3)
#     ])

#     model.layers[3].set_weights([
#         np.ones((3, 1)),
#         np.zeros(1)
#     ])

#     data = np.reshape(np.arange(1, 4), (1, 3))

#     lrp = RestructuredLRP(model, layer=3, idx=0, bottleneck=1)

#     a = np.asarray([[22, 30, 40]])
#     explanations = lrp.predict([data, a])

#     expected = np.asarray([[-0.22222, -0.61111, -1.16666]])

#     assert np.allclose(expected, explanations, atol=1e-5), \
#         'RestructuredLRP does not return the correct explanations'


# def test_restructured_lrp_strategy():
#     strategy = LRPStrategy(
#         layers=[
#             {'flat': True},
#             {'alpha': 1, 'beta': 0},
#             {'epsilon': 0.25},
#             {'epsilon': 0.25}
#         ]
#     )

#     input = Input((3,))
#     x = Dense(3, activation=None)(input)
#     x = Dense(3, activation=None)(x)
#     x = Dense(3, activation=None)(x)
#     x = Dense(1, activation=None)(x)
#     model = Model(input, x)
#     lrp = RestructuredLRP(model, layer=4, idx=0, bottleneck=3, strategy=strategy)

#     standard_layers = [20, 19, 18, 17]
#     for i in range(len(standard_layers)):
#         configuration = strategy.layers[i]
#         layer = lrp.layers[standard_layers[i]]
#         for key in configuration:
#             assert configuration[key] == getattr(layer, key), \
#                 'RestructuredLRP with strategy does not work'


# def test_restructured_lrp_non_dense_bottleneck():
#     input = Input((3,3,1))
#     x = Conv2D(3, (3, 3), activation=None)(input)
#     x = Flatten()(x)
#     x = Activation('relu')(x)
#     x = Dense(1, activation=None)(x)
#     model = Model(input, x)

#     exception = False

#     try:
#         RestructuredLRP(model, layer=3, idx=0, bottleneck=1)
#     except Exception:
#         exception = True

#     assert exception, \
#         ('Creating a RestructuredLRP with a non-Dense bottleneck layer does '
#          'not raise an error')

# def test_restructured_lrp_built_in_activation():
#     input = Input((3,))
#     x = Dense(3, activation='relu')(input)
#     x = Dense(1, activation=None)(x)
#     model = Model(input, x)

#     exception = False

#     try:
#         RestructuredLRP(model, layer=2, idx=0, bottleneck=1)
#     except Exception:
#         exception = True

#     assert exception, \
#         ('Instantiating a RestructuredLRP with a bottleneck layer with a '
#          'built-in activation does not raise an error')

# def test_disconnected_bottleneck_and_target():
#     input = Input((3,))
#     x = Dense(3, activation=None)(input)
#     x = Dense(3, activation=None)(x)
#     x = Dense(1, activation=None)(x)
#     model = Model(input, x)

#     exception = False

#     try:
#         RestructuredLRP(model, layer=3, idx=0, bottleneck=1)
#     except Exception:
#         exception = True

#     assert exception, \
#         ('Instantiating a RestructuredLRP where the bottleneck is not '
#          'directly connected to the target layer does not raise an error')

# def test_disconnected_bottleneck_and_target_relu():
#     input = Input((3,))
#     x = Dense(3, activation=None)(input)
#     x = Activation('relu')(x)
#     x = Dense(1, activation=None)(x)
#     model = Model(input, x)

#     exception = False

#     try:
#         RestructuredLRP(model, layer=3, idx=0, bottleneck=1)
#     except Exception as e:
#         print(e)
#         exception = True

#     assert not exception, \
#         ('Instantiating a RestructuredLRP with a ReLU between the bottleneck '
#          'and the target raises an error')

def test_restructured_lrp_with_threshold():
    input = Input((3,))
    x = Dense(3, activation=None)(input)
    x = Activation('relu')(x)
    x = Dense(1, activation=None)(x)
    model = Model(input, x)

    model.layers[1].set_weights([
        np.reshape(np.arange(9), (3, 3)),
        np.zeros(3)
    ])

    model.layers[3].set_weights([
        np.ones((3, 1)),
        np.zeros(1)
    ])

    data = np.reshape(np.arange(1, 4), (1, 3))

    lrp = RestructuredLRP(model, layer=3, idx=0, bottleneck=1, threshold=True)

    a = np.asarray([[22, 30, 40]])
    t = np.asarray([[3., 3., 3.]])
    explanations = lrp.predict([data, a, t])

    expected = np.asarray([[-0.22222, -1.11111, -2.66666]])

    assert np.allclose(expected, explanations, atol=1e-5), \
        ('RestructuredLRP with threshold does not return the correct '
         'explanations')
