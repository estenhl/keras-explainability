"""Contains tests for the Conv2D-layer LRP implementations"""

import numpy as np

from tensorflow.keras.initializers import Constant
from tensorflow.keras import Model
from tensorflow.keras.layers import BatchNormalization, Input

from explainability.layers import BatchNormalizationLRP


def test_batchnormalization_lrp():
    input = Input((3, 3, 2))

    values = np.reshape(np.arange(3*3*2), (1, 3, 3, 2))
    means = Constant(2)
    vars = Constant(3)

    layer = BatchNormalization(
        moving_mean_initializer=means,
        moving_variance_initializer=vars
    )(input)
    model = Model(input, layer)
    relevances = model(values).numpy()

    l = BatchNormalizationLRP(model.layers[1])([input, relevances])
    explainer = Model(input, l)
    explanations = explainer(values).numpy()

    assert np.allclose(values, explanations, atol=1e-3), \
        'BatchNormalizationLRP does not return the expected explanations'
