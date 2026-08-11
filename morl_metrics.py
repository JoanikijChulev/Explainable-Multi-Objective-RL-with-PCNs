from functools import lru_cache

import numpy as np


EXPECTED_UTILITY_WEIGHT_SAMPLES = 10000
EXPECTED_UTILITY_WEIGHT_SEED = 0


def _clean_metric_points(points):
    points = np.asarray(points, dtype=np.float32)
    if points.ndim == 1:
        points = points[None, :]
    if len(points) == 0:
        return points
    return points[~np.isnan(points).any(axis=1)]


@lru_cache(maxsize=None)
def _expected_utility_weights(n_objectives, n_weight_samples, seed):
    if n_objectives <= 0:
        raise ValueError('n_objectives must be positive')
    if n_objectives == 1:
        return np.ones((n_weight_samples, 1), dtype=np.float32)

    rng = np.random.default_rng(seed)
    return rng.dirichlet(
        np.ones(n_objectives, dtype=np.float32),
        size=n_weight_samples,
    ).astype(np.float32)


def expected_utility(
    points,
    n_weight_samples=EXPECTED_UTILITY_WEIGHT_SAMPLES,
    seed=EXPECTED_UTILITY_WEIGHT_SEED,
):
    points = _clean_metric_points(points)
    if len(points) == 0:
        return np.nan

    weights = _expected_utility_weights(points.shape[-1], int(n_weight_samples), int(seed))
    utilities = points @ weights.T
    return float(np.mean(np.max(utilities, axis=0)))
