import numpy as np


def topological_sort(dependencies: np.ndarray) -> np.ndarray:
    order = []

    while len(order) < len(dependencies):
        for i in range(len(dependencies)):
            if i in order:
                continue
            if np.sum(dependencies[:,i]) == 0:
                order.append(i)
                dependencies[i,:] = 0
                break

    return np.asarray(order)
