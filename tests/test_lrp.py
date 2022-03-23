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

    print(explanations)

    for i in range(len(explainer.layers)):
        print(f'{i}: {explainer.layers[i]}')

    m = Model(explainer.input, explainer.layers[12].output)
    print(m([
        np.arange(1, 6, dtype=np.float32).reshape((1, 5)),
        np.arange(-2, 3, dtype=np.float32).reshape((1, 5))
    ]))

    expected = np.asarray([[0., 4., 9., 16., 25.]])

