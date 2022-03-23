import numpy as np

from explainability.utils import topological_sort

def test_topological_sort():
    dependencies = np.asarray([
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0]
    ])

    expected = [
        [0, 1, 2, 3, 4, 5, 6],
        [0, 1, 2, 4, 3, 5, 6]
    ]

    order = topological_sort(dependencies)

    assert any([np.array_equal(order, e) for e in expected]), \
        'topological_sort does not sort dependencies correctly'
