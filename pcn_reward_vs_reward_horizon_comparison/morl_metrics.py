from functools import lru_cache

import numpy as np
from pygmo import hypervolume


EXPECTED_UTILITY_WEIGHT_SAMPLES = 10000
EXPECTED_UTILITY_WEIGHT_SEED = 0


def _clean_metric_points(points):
    points = np.asarray(points, dtype=np.float32)
    if points.ndim == 1:
        points = points[None, :]
    if len(points) == 0:
        return points
    return points[~np.isnan(points).any(axis=1)]


def _non_dominated_metric_points(points):
    keep = np.ones(points.shape[0], dtype=bool)
    for index, candidate in enumerate(points):
        if keep[index]:
            keep[keep] = np.any(points[keep] > candidate, axis=1)
            keep[index] = True
    return points[keep]


def hypervolume_score(points, ref_point, *, return_excluded=False):
    """Compute maximization hypervolume inside a fixed reference region.

    A point contributes only when it is no worse than ``ref_point`` in every
    objective. Non-dominated points outside that region are excluded instead
    of being passed to PyGMO, which would reject the complete set. Points on a
    reference boundary have zero volume and are omitted from the numerical
    calculation without changing the score.
    """
    points = _clean_metric_points(points)
    if len(points) == 0:
        score = np.nan
        excluded = 0
    else:
        ref_point = np.asarray(ref_point, dtype=np.float32)
        if ref_point.ndim != 1:
            raise ValueError('ref_point must be a one-dimensional vector')
        if points.shape[-1] != len(ref_point):
            raise ValueError(
                f'point dimension {points.shape[-1]} does not match '
                f'reference dimension {len(ref_point)}'
            )

        front = _non_dominated_metric_points(points)
        in_reference_region = np.all(front >= ref_point, axis=1)
        excluded = int(np.count_nonzero(~in_reference_region))

        # A point equal to the reference on any objective spans a zero-width
        # box in that objective and therefore contributes zero hypervolume.
        contributing = front[
            in_reference_region & np.all(front > ref_point, axis=1)
        ]
        if len(contributing) == 0:
            score = 0.0
        else:
            score = float(
                hypervolume(-contributing.astype(np.float64)).compute(
                    -ref_point.astype(np.float64)
                )
            )

    if return_excluded:
        return score, excluded
    return score


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
