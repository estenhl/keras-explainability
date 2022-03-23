import numpy as np
import tensorflow as tf

from tensorflow.keras import Model
from typing import List


def _is_neighbour(outputs: List[tf.Tensor], inputs: List[tf.Tensor]) -> bool:
    outputs = outputs if isinstance(outputs, list) else [outputs]
    outputs = [output.ref() for output in outputs]
    inputs = inputs if isinstance(inputs, list) else [inputs]
    inputs = [input.ref() for input in inputs]

    return len(set(outputs)& set(inputs)) > 0

def infer_graph_structure(model: Model) -> np.ndarray:
    inputs = [layer.input for layer in model.layers]
    outputs = [layer.output for layer in model.layers]

    neighbours = np.asarray([[_is_neighbour(outputs[i], inputs[j]) \
                              if i != j else False for j in range(len(inputs))]
                             for i in range(len(inputs))]).astype(np.uint8)

    return neighbours
