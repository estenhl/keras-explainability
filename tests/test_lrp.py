import numpy as np
import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import Add, Dense, Input, Subtract

from explainability import LRP


def test_non_sequential_model():
    i1 = Input((5,))
    d1 = Dense(5)(i1)
    c1 = tf.constant(np.arange(-1, 4).reshape((1, -1)).astype(np.float32))
    a = Add()([d1, c1])
    d2 = Dense(5)(a)
    i2 = Input((5,))
    s = Subtract()([d2, i2])
    o = Dense(1)(s)

    model = Model([i1, i2], o)

    model.layers[1].set_weights([
        np.eye(5, dtype=np.float32),
        np.zeros(5, dtype=np.float32)
    ])
    model.layers[3].set_weights([
       np.eye(5, dtype=np.float32),
       np.zeros(5, dtype=np.float32)
    ])
    model.layers[-1].set_weights([
       np.arange(1, 6, dtype=np.float32).reshape((5, 1)),
       np.zeros(1, dtype=np.float32)
    ])

    explainer = LRP(model, layer=6, idx=0)
    explanations = explainer([
        np.arange(1, 6, dtype=np.float32).reshape((1, 5)),
        np.arange(-2, 3, dtype=np.float32).reshape((1, 5))
    ])

    expected = np.asarray([[0., 4., 9., 16., 25.]])

    assert np.array_equal(expected, explanations[0]), \
        ('LRP with non-sequential model does not return the correct '
         'explanations')

def test_lrp_negative_layer():
    i = Input((3,))
    x = Dense(3)(i)
    x = Dense(1)(x)
    model = Model(i, x)

    lrp = LRP(model, layer=-1, idx=0)
    dense_layers = [l for l in lrp.layers if isinstance(l, Dense)]

    assert 2 == len(dense_layers), \
        ('Indexing an LRP with -1 does not provide explanations for the last '
         'layer')

def test_lrp_not_last_layer():
    i = Input((3,))
    x = Dense(3)(i)
    x = Dense(3)(x)
    model = Model(i, x)

    model.layers[1].set_weights([
        np.asarray([[0, 0, 1], [0, 1, 0], [1, 0, 0]]),
        np.zeros(3)
    ])
    model.layers[2].set_weights([
        np.ones((3, 3)),
        np.zeros(3)
    ])

    data = np.asarray([[1, 2, 3]])

    lrp = LRP(model, layer=1, idx=0)
    explanations = lrp.predict(data)

    assert np.array_equal(np.asarray([[0., 0., 3.]]), explanations), \
        ('LRP which does not target the last layer does not return the '
         'correct explanations')



