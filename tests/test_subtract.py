import numpy as np

from tensorflow.keras import Model
from tensorflow.keras.layers import Subtract, Input

from explainability.layers import SubtractLRP


def test_subtract():
    i1 = Input((4,))
    i2 = Input((4,))
    a = Subtract()([i1, i2])
    model = Model([i1, i2], a)

    explainer = SubtractLRP(model.layers[-1])
    explanations = explainer([[np.asarray([-4., -1., 1., 4.]),
                               np.asarray([2., 2., 2., 2.])],
                              np.ones(4, dtype=np.float32)])

    assert isinstance(explanations, list), \
        'SubtractLRP does not return a list of explanations'
    assert len(explanations) == 2, \
        'SubtractLRP does not return two explanations'
    assert np.allclose([2/3, 1/3, -1, 2], explanations[0], atol=1e-5), \
        'SubtractLRP does not return correct explanations for first element'
    assert np.allclose([1/3, 2/3, 2, -1], explanations[1], atol=1e-5), \
        'SubtractLRP does not return correct explanations for second element'
