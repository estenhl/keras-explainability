import numpy as np

from tensorflow.keras import Model
from tensorflow.keras.layers import Add, Input

from explainability.layers import AddLRP


def test_add_positive():
    i1 = Input((3,))
    i2 = Input((3,))
    a = Add()([i1, i2])
    model = Model([i1, i2], a)

    explainer = AddLRP(model.layers[-1])
    explanations = explainer([[np.arange(0, 3, dtype=np.float32),
                               np.arange(1, 4, dtype=np.float32)],
                              np.arange(1, 7, 2, dtype=np.float32)])

    assert isinstance(explanations, list), \
        'AddLRP does not return a list of explanations'
    assert len(explanations) == 2, \
        'AddLRP does not return two explanations'
    assert np.array_equal([0, 1, 2], explanations[0]), \
        'AddLRP does not return correct explanations for first element'
    assert np.array_equal([1, 2, 3], explanations[1]), \
        'AddLRP does not return correct explanations for second element'

def test_add_positive_relevance():
    i1 = Input((3,))
    i2 = Input((3,))
    a = Add()([i1, i2])
    model = Model([i1, i2], a)

    explainer = AddLRP(model.layers[-1])
    explanations = explainer([[np.arange(0, 3, dtype=np.float32),
                               np.arange(1, 4, dtype=np.float32)],
                              np.ones(3, dtype=np.float32)])

    assert isinstance(explanations, list), \
        'AddLRP does not return a list of explanations'
    assert len(explanations) == 2, \
        'AddLRP does not return two explanations'
    assert np.allclose([0, 1/3, 2/5], explanations[0], atol=1e-5), \
        'AddLRP does not return correct explanations for first element'
    assert np.allclose([1, 2/3, 3/5], explanations[1], atol=1e-5), \
        'AddLRP does not return correct explanations for second element'
