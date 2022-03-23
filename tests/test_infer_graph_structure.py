import numpy as np
import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import Add, Dense, Input, Subtract

from explainability.utils import infer_graph_structure


def test_infer_graph_structure():
    i1 = Input((5,))
    d1 = Dense(5)(i1)
    c1 = tf.constant(np.arange(5).reshape((-1, 1)).astype(np.float32))
    a = Add()([d1, c1])
    d2 = Dense(5)(a)
    i2 = Input((5,))
    s = Subtract()([d2, i2])
    o = Dense(1)(s)

    model = Model([i1, i2], o)

    for i in range(len(model.layers)):
        print(f'{i}: {model.layers[i]}')

    dependencies = infer_graph_structure(model)

    expected = np.asarray([
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0]
    ])

    assert np.array_equal(expected, dependencies), \
        'infer_graph_structure does not infer the correct structure'
