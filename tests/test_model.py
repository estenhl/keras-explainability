import os
import numpy as np

from skimage.io import imread
from skimage.transform import resize
from tensorflow.keras import Model
from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import Activation, Dense, Flatten, Input

from explainability import LayerwiseRelevancePropagator


def test_vgg19_cat():
    cwd = os.path.dirname(os.path.realpath(__file__))
    image = os.path.join(cwd, 'data', 'cat.jpg')
    expected = os.path.join(cwd, 'data', 'cat_explanations.npy')

    assert os.path.isfile(image), \
        'Unable to test VGG19 explanations without cat image'

    assert os.path.isfile(expected), \
        'Unable to test VGG19 explanations without expected explanations'

    image = imread(image)
    image = resize(image, (224, 224, 3), preserve_range=True)
    image = image.astype(np.uint8)

    model = VGG19(weights='imagenet')

    explainer = LayerwiseRelevancePropagator(model, layer=25, idx=281, alpha=1,
                                             beta=0)
    explanations = explainer(np.reshape(image, (1, 224, 224, 3))).numpy()

    expected = np.load(expected)

    assert np.allclose(expected, explanations, 1e-8), \
        ('LayerwiseRelevancePropagator does not return the expected '
         'explanations for a cat image using VGG19')

