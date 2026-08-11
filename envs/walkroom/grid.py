import itertools
import numpy as np


def _initialize_grid_python(size, dim, seed):
    rng = np.random.default_rng(seed)
    data = np.empty(size ** dim, dtype=np.uint8)

    # Match the original extension's iteration order: the first coordinate
    # changes fastest in the flattened output before the final reshape.
    for i, reversed_coord in enumerate(itertools.product(range(size), repeat=dim)):
        coord = reversed_coord[::-1]
        pos = sum(coord) + rng.normal(0.0, size / 10.0)
        pos = size - 1 - pos
        data[i] = np.uint8(np.rint(np.clip(pos, 0.0, size - 1)))

    return data


def initialize_grid(size, dim, seed):
    return _initialize_grid_python(size, dim, seed)
