import numpy as np
import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import Activation, Add, Dense, Input, \
                                    LayerNormalization, Subtract

from explainability import LRP


# def test_non_sequential_model():
#     i1 = Input((5,))
#     d1 = Dense(5)(i1)
#     c1 = tf.constant(np.arange(-1, 4).reshape((1, -1)).astype(np.float32))
#     a = Add()([d1, c1])
#     d2 = Dense(5)(a)
#     i2 = Input((5,))
#     s = Subtract()([d2, i2])
#     o = Dense(1)(s)

#     model = Model([i1, i2], o)

#     model.layers[1].set_weights([
#         np.eye(5, dtype=np.float32),
#         np.zeros(5, dtype=np.float32)
#     ])
#     model.layers[3].set_weights([
#        np.eye(5, dtype=np.float32),
#        np.zeros(5, dtype=np.float32)
#     ])
#     model.layers[-1].set_weights([
#        np.arange(1, 6, dtype=np.float32).reshape((5, 1)),
#        np.zeros(1, dtype=np.float32)
#     ])

#     explainer = LRP(model, layer=6, idx=0)
#     explanations = explainer([
#         np.arange(1, 6, dtype=np.float32).reshape((1, 5)),
#         np.arange(-2, 3, dtype=np.float32).reshape((1, 5))
#     ])

#     expected = np.asarray([[0., 4., 9., 16., 25.]])

#     assert np.array_equal(expected, explanations[0]), \
#         ('LRP with non-sequential model does not return the correct '
#          'explanations')

# def test_lrp_negative_layer():
#     i = Input((3,))
#     x = Dense(3)(i)
#     x = Dense(1)(x)
#     model = Model(i, x)

#     lrp = LRP(model, layer=-1, idx=0)
#     dense_layers = [l for l in lrp.layers if isinstance(l, Dense)]

#     assert 2 == len(dense_layers), \
#         ('Indexing an LRP with -1 does not provide explanations for the last '
#          'layer')

# def test_lrp_not_last_layer():
#     i = Input((3,))
#     x = Dense(3)(i)
#     x = Dense(3)(x)
#     model = Model(i, x)

#     model.layers[1].set_weights([
#         np.asarray([[0, 0, 1], [0, 1, 0], [1, 0, 0]]),
#         np.zeros(3)
#     ])
#     model.layers[2].set_weights([
#         np.ones((3, 3)),
#         np.zeros(3)
#     ])

#     data = np.asarray([[1, 2, 3]])

#     lrp = LRP(model, layer=1, idx=0)
#     explanations = lrp.predict(data)

#     assert np.array_equal(np.asarray([[0., 0., 3.]]), explanations), \
#         ('LRP which does not target the last layer does not return the '
#          'correct explanations')

def test_lrp_layer_normalization():
    betas = [1, 0]
    gammas = [1, 0.5]
    i = Input((3,))
    x = Dense(3)(i)
    x = LayerNormalization(
        beta_initializer=tf.constant_initializer(betas[0]),
        gamma_initializer=tf.constant_initializer(gammas[0])
    )(x)
    x = Activation('relu')(x)
    x = Dense(2)(x)
    x = LayerNormalization(
        beta_initializer=tf.constant_initializer(betas[1]),
        gamma_initializer=tf.constant_initializer(gammas[1]))(x)
    x = Activation('relu')(x)
    x = Dense(1)(x)

    weights = [
        np.asarray([
            [1, -1, 0],
            [-1, -1, 1],
            [1, 0, 1]
        ]),
        np.asarray([
            [1, 0],
            [0, -1],
            [-1, 1]
        ]),
        np.asarray([[1], [1]]),
    ]

    model = Model(i, x)
    model.layers[1].set_weights([
        weights[0],
        np.zeros(3)
    ])
    model.layers[4].set_weights([
        weights[1],
        np.zeros(2)
    ])
    model.layers[7].set_weights([
        weights[2],
        np.zeros(1)
    ])

    inputs = np.asarray([[1, 2, 3]])

    prev = inputs
    outputs = []
    for i in range(len(model.layers)):
        prev = model.layers[i](prev)
        outputs.append(prev)

    means = []
    stds = []
    normalized = []
    modified_weights = []
    modified_biases = []

    # Manually adjust the weights and betas for all dense layers
    # according to https://arxiv.org/pdf/2002.11018.pdf
    for weight_idx, layer_idx in [(0, 1), (1, 4)]:
        means.append(np.mean(outputs[layer_idx]))
        stds.append(np.std(outputs[layer_idx] - means[-1]))
        normalized.append((outputs[layer_idx] - means[-1]) / stds[-1])
        modified_weights.append((gammas[weight_idx] * weights[weight_idx]) /
                                stds[-1])
        modified_biases.append(betas[weight_idx] + gammas[weight_idx] * \
                               ((0 - means[-1]) / stds[-1]))

    i = Input((3,))
    x = Dense(3)(i)
    x = Activation('relu')(x)
    x = Dense(2)(x)
    x = Activation('relu')(x)
    x = Dense(1)(x)
    modified_model = Model(i, x)

    modified_model.layers[1].set_weights([
        modified_weights[0],
        np.asarray([modified_biases[0]] * 3)
    ])
    modified_model.layers[3].set_weights([
        modified_weights[1],
        np.asarray([modified_biases[1]] * 2)
    ])
    modified_model.layers[5].set_weights([
        weights[2],
        np.zeros(1)
    ])

    assert np.allclose(model.predict(inputs),
                       modified_model.predict(inputs), atol=1e-3), \
        ('Manually configured modified model is wrong')

    lrp = LRP(modified_model, layer=-1, idx=0)
    expected = lrp.predict(inputs)

    modified_lrp = LRP(model, layer=-1, idx=0)
    explanation = modified_lrp.predict(inputs)

    assert np.allclose(expected, explanation, atol=1e-3), \
        ('LRP with LayerNormalization does not return the correct '
         'explanations')
