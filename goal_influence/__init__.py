"""Goal-influence analysis tools.

The implementation remains in :mod:`goal_influence.goal_influence`. Keeping
this folder as a package supports both direct script commands and ``python -m``
execution without duplicating any analysis code.
"""

from importlib import import_module
from pathlib import Path

_PUBLIC_API = {
    "DEFAULT_FRACTIONS",
    "all_scores",
    "auroc",
    "explain_trajectory",
    "generic_baseline",
    "generic_directions",
    "goal_influence_at_state",
    "grid_ground_truth",
    "js_divergence",
    "kl_divergence",
    "load_run",
    "main",
    "plot_profile",
    "query_probs_batch",
    "rankdata",
    "spearman",
    "tv_distance",
}


def goal_output_artifacts_dir():
    """Return the goal-influence package's local artifact directory."""
    path = Path(__file__).resolve().parent / "goal_output_artifacts"
    path.mkdir(parents=True, exist_ok=True)
    return path


__all__ = sorted(_PUBLIC_API | {"goal_output_artifacts_dir"})


def __getattr__(name):
    if name not in _PUBLIC_API:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(".goal_influence", __name__), name)


def __dir__():
    return sorted(set(globals()) | _PUBLIC_API)
