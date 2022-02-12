from tensorflow.keras import Model

from ..layers import LRP


class LayerwiseRelevancePropagator(Model):
    def __init__(self, model: Model):
        activations = [layer.output for layer in model.layers]

        prev = model.output
        for i in range(len(model.layers)-1, 0, -1):
            prev = LRP(model.layers[i], name=f'{i}')([activations[i-1], prev])

        super().__init__(model.input, prev)
