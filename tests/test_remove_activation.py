import numpy as np

from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input

from explainability import LRP
from explainability.model.utils import remove_activation, remove_sigmoid, \
                                       remove_softmax


def test_remove_softmax():
    input = Input((10,))
    output = Dense(10, activation='softmax')(input)
    model = Model(input, output)

    model.layers[1].set_weights([
        np.reshape(np.arange(0, 1, 0.01) * -1, (10, 10)),
        np.zeros(10)
    ])

    dataset = np.reshape(np.arange(10), (-1, 10))
    predictions = model.predict(dataset)

    assert np.all(predictions >= 0), \
        'Model with softmax output doesn\'t transform values'

    model = remove_softmax(model)

    predictions = model.predict(dataset)
    expected = np.asarray([[
        -28.5, -28.95, -29.399998, -29.849998, -30.3, -30.75, -31.199999,
        -31.649998, -32.1, -32.550003
    ]])

    assert np.allclose(expected, predictions, 1e-4), \
        'Model returns the wrong predictions after remove_softmax'

def test_remove_softmax_implicit():
    input = Input((10,))
    output = Dense(10, activation='softmax')(input)
    model = Model(input, output)

    model.layers[1].set_weights([
        np.reshape(np.arange(0, 1, 0.01) * -1, (10, 10)),
        np.zeros(10)
    ])

    dataset = np.reshape(np.arange(10), (-1, 10))
    predictions = model.predict(dataset)

    assert np.all(predictions >= 0), \
        'Model with softmax output doesn\'t transform values'

    model = remove_activation(model, ['sigmoid', 'softmax'])

    predictions = model.predict(dataset)
    expected = np.asarray([[
        -28.5, -28.95, -29.399998, -29.849998, -30.3, -30.75, -31.199999,
        -31.649998, -32.1, -32.550003
    ]])

    assert np.allclose(expected, predictions, 1e-4), \
        'Model returns the wrong predictions after remove_activation'

def test_remove_sigmoid():
    input = Input(3)
    output = Dense(1, activation='sigmoid')(input)
    model = Model(input, output)

    model.layers[1].set_weights([
        np.ones((3, 1)),
        np.zeros(1)
    ])

    model = remove_sigmoid(model)

    predictions = model.predict(np.ones((1, 3)) * -1)

    assert np.array_equal([[-3.]], predictions), \
        'remove_sigmoid does not remove sigmoid'

def test_remove_sigmoid_implicit():
    input = Input(3)
    output = Dense(1, activation='sigmoid')(input)
    model = Model(input, output)

    model.layers[1].set_weights([
        np.ones((3, 1)),
        np.zeros(1)
    ])

    model = remove_activation(model, ['softmax', 'sigmoid'])

    predictions = model.predict(np.ones((1, 3)) * -1)

    assert np.array_equal([[-3.]], predictions), \
        'remove_activation does not remove sigmoid'

def test_remove_softmax_model():
    input = Input((10,))
    output = Dense(10, activation='softmax')(input)
    model = Model(input, output)
    lrp = LRP(model, layer=-1, idx=0)

    assert 'linear' == lrp.layers[1].activation.__name__, \
        'LRP does not automatically remove softmax activation'

def test_remove_sigmoid_model():
    input = Input(3)
    output = Dense(1, activation='sigmoid')(input)
    model = Model(input, output)
    lrp = LRP(model, layer=-1, idx=0)

    assert 'linear' == lrp.layers[1].activation.__name__, \
        'LRP does not automatically remove sigmoid activation'
