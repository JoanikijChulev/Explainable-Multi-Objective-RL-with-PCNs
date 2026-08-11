from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from artifacts import plots_dir
from paper_plot_utils import save_paper_figure


def _load_scalar(log_file, tag):
    if tag not in log_file:
        return None, None
    data = log_file[tag][:]
    if len(data) == 0:
        return None, None
    return data[:, 0], data[:, 1]


def _load_ndarray(log_file, tag):
    step_key = f'{tag}/step'
    array_key = f'{tag}/ndarray'
    if step_key not in log_file or array_key not in log_file:
        return None, None
    steps = log_file[step_key][:]
    arrays = log_file[array_key][:]
    if len(steps) == 0:
        return None, None
    return steps, arrays


def _save_line_plot(steps, values, title, ylabel, output_path):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(steps, values, linewidth=2)
    ax.set_title(title)
    ax.set_xlabel('Timesteps')
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_paper_figure(fig, output_path)
    plt.close(fig)


def _save_eval_return_plot(steps, values, output_path):
    mean_returns = np.nanmean(values, axis=1)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for objective in range(mean_returns.shape[-1]):
        ax.plot(steps, mean_returns[:, objective], linewidth=2, label=f'Objective {objective}')
    ax.set_title('Evaluation Return vs Timesteps')
    ax.set_xlabel('Timesteps')
    ax.set_ylabel('Mean Evaluated Return')
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    save_paper_figure(fig, output_path)
    plt.close(fig)


def _filter_non_dominated(points):
    from eval_pcn import non_dominated

    points = np.asarray(points)
    points = points[~np.isnan(points).any(axis=1)]
    if len(points) == 0:
        return points
    return non_dominated(points)


def _clean_points(points):
    points = np.asarray(points)
    if len(points) == 0:
        return points
    return points[~np.isnan(points).any(axis=1)]


def _split_non_dominated(points):
    from eval_pcn import non_dominated

    points = _clean_points(points)
    if len(points) == 0:
        return points, points

    _, mask = non_dominated(points, return_indexes=True)
    return points[mask], points[~mask]


def _sort_front(points):
    if len(points) == 0:
        return points
    return points[np.argsort(points[:, 0])]


def _select_diverse_front_points(points, max_points=5):
    points = np.asarray(points)
    if len(points) <= max_points:
        return points

    normalized = (points - points.min(axis=0)) / (np.ptp(points, axis=0) + 1e-8)
    centroid = normalized.mean(axis=0)
    selected = [int(np.argmax(np.linalg.norm(normalized - centroid, axis=1)))]

    while len(selected) < max_points:
        remaining = np.array([index for index in range(len(points)) if index not in selected], dtype=np.int32)
        distances = normalized[remaining, None, :] - normalized[np.asarray(selected), :]
        min_distances = np.min(np.linalg.norm(distances, axis=-1), axis=1)
        selected.append(int(remaining[np.argmax(min_distances)]))

    return points[np.asarray(selected)]


def _objective_label(objective_labels, index):
    if objective_labels is not None and index < len(objective_labels):
        return objective_labels[index]
    return f'Objective {index} Return'


def _style_3d_pareto_axes(ax, objective_labels=None):
    z_label = _objective_label(objective_labels, 2)
    ax.set_xlabel(_objective_label(objective_labels, 0), labelpad=8)
    ax.set_ylabel(_objective_label(objective_labels, 1), labelpad=8)
    ax.set_zlabel('')
    # Matplotlib's native 3D z-axis label is often clipped or hidden when
    # exported with tight bounding boxes. Add a vector side label to make the
    # vertical objective readable in paper figures.
    ax.text2D(1.09, 0.52, z_label, transform=ax.transAxes, rotation=90, va='center', ha='center')
    # Match the historical Minecart eval plot viewpoint while keeping the semantic
    # labels introduced for paper-ready figures.
    ax.view_init(elev=30, azim=-60)


def _save_3d_paper_figure(fig, output_path):
    fig.subplots_adjust(left=0.02, right=0.78, bottom=0.08, top=0.92)
    save_paper_figure(fig, output_path, pad_inches=0.08)


def _save_pareto_front_plot(values, output_path, objective_labels=None):
    pareto_front = _filter_non_dominated(values[-1])
    if len(pareto_front) == 0:
        return
    n_objectives = pareto_front.shape[-1]

    if n_objectives == 2:
        pareto_front = _sort_front(pareto_front)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(pareto_front[:, 0], pareto_front[:, 1], marker='o')
        ax.set_xlabel('Objective 0 Return')
        ax.set_ylabel('Objective 1 Return')
    elif n_objectives == 3:
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(pareto_front[:, 0], pareto_front[:, 1], pareto_front[:, 2], s=40)
        _style_3d_pareto_axes(ax, objective_labels)
    else:
        fig, ax = plt.subplots(figsize=(9, 5))
        objective_indexes = np.arange(n_objectives)
        pareto_front = _select_diverse_front_points(pareto_front, max_points=5)
        for point in pareto_front:
            ax.plot(objective_indexes, point, alpha=0.4)
        ax.set_xticks(objective_indexes)
        ax.set_xlabel('Objective Index')
        ax.set_ylabel('Return')

    ax.set_title('Pareto Front')
    ax.grid(True, alpha=0.3)
    if n_objectives == 3:
        _save_3d_paper_figure(fig, output_path)
    else:
        fig.tight_layout()
        save_paper_figure(fig, output_path)
    plt.close(fig)


def save_eval_pareto_front_comparison(logged_front, achieved_front, output_path, objective_labels=None):
    logged_front = _filter_non_dominated(logged_front)
    achieved_front, dominated_achieved_front = _split_non_dominated(achieved_front)
    if len(logged_front) == 0 and len(achieved_front) == 0 and len(dominated_achieved_front) == 0:
        return

    fronts = [front for front in (logged_front, achieved_front, dominated_achieved_front) if len(front) > 0]
    n_objectives = fronts[0].shape[-1]

    if n_objectives == 2:
        fig, ax = plt.subplots(figsize=(6, 6))
        if len(logged_front) > 0:
            logged_front = _sort_front(logged_front)
            ax.plot(logged_front[:, 0], logged_front[:, 1], marker='o', label='Logged Front')
        if len(achieved_front) > 0:
            achieved_front = _sort_front(achieved_front)
            ax.plot(achieved_front[:, 0], achieved_front[:, 1], marker='x', color='C1', label='Achieved Pareto')
        if len(dominated_achieved_front) > 0:
            dominated_achieved_front = _sort_front(dominated_achieved_front)
            ax.plot(
                dominated_achieved_front[:, 0],
                dominated_achieved_front[:, 1],
                marker='x',
                color='C3',
                linestyle='None',
                label='Dominated Achieved',
            )
        ax.set_xlabel('Objective 0 Return')
        ax.set_ylabel('Objective 1 Return')
    elif n_objectives == 3:
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection='3d')
        if len(logged_front) > 0:
            ax.scatter(logged_front[:, 0], logged_front[:, 1], logged_front[:, 2], s=40, label='Logged Front')
        if len(achieved_front) > 0:
            ax.scatter(
                achieved_front[:, 0],
                achieved_front[:, 1],
                achieved_front[:, 2],
                s=40,
                marker='x',
                color='C1',
                label='Achieved Pareto',
            )
        if len(dominated_achieved_front) > 0:
            ax.scatter(
                dominated_achieved_front[:, 0],
                dominated_achieved_front[:, 1],
                dominated_achieved_front[:, 2],
                s=40,
                marker='x',
                color='C3',
                label='Dominated Achieved',
            )
        _style_3d_pareto_axes(ax, objective_labels)
    else:
        fig, ax = plt.subplots(figsize=(9, 5))
        objective_indexes = np.arange(n_objectives)

        def _plot_front_summary(front, color, label, line_width, line_alpha, fill_alpha, zorder):
            if len(front) == 0:
                return
            if len(front) <= 5:
                for point in front:
                    ax.plot(
                        objective_indexes,
                        point,
                        alpha=line_alpha,
                        linewidth=line_width,
                        color=color,
                        zorder=zorder,
                    )
                ax.plot([], [], color=color, alpha=line_alpha, linewidth=line_width, label=label)
            else:
                front_min = np.min(front, axis=0)
                front_med = np.median(front, axis=0)
                front_max = np.max(front, axis=0)
                ax.fill_between(
                    objective_indexes,
                    front_min,
                    front_max,
                    alpha=fill_alpha,
                    color=color,
                    zorder=zorder - 1,
                )
                ax.plot(
                    objective_indexes,
                    front_med,
                    linewidth=line_width,
                    alpha=line_alpha,
                    color=color,
                    label=label,
                    zorder=zorder,
                )

        _plot_front_summary(
            logged_front,
            'C0',
            'Logged Front',
            line_width=5,
            line_alpha=0.3,
            fill_alpha=0.1,
            zorder=2,
        )
        _plot_front_summary(
            achieved_front,
            'C1',
            'Achieved Pareto',
            line_width=1,
            line_alpha=1,
            fill_alpha=0.3,
            zorder=3,
        )
        ax.set_xticks(objective_indexes)
        ax.set_xlabel('Objective Index')
        ax.set_ylabel('Return')

    ax.set_title('Logged Pareto Front vs Achieved Results')
    ax.grid(True, alpha=0.3)
    ax.legend()
    if n_objectives == 3:
        _save_3d_paper_figure(fig, output_path)
    else:
        fig.tight_layout()
        save_paper_figure(fig, output_path)
    plt.close(fig)





def save_training_plots(run_dir):
    run_dir = Path(run_dir)
    output_dir = plots_dir(run_dir, create=True)
    log_path = run_dir / 'log.h5'
    if not log_path.is_file():
        return

    with h5py.File(log_path, 'r') as log_file:
        scalar_plots = [
            ('train/episode_length', 'Episode Length vs Timesteps', 'Episode Length', 'episode_length_vs_timesteps.png'),
            ('train/loss', 'Loss vs Timesteps', 'Loss', 'loss_vs_timesteps.png'),
            ('train/entropy', 'Entropy vs Timesteps', 'Entropy', 'entropy_vs_timesteps.png'),
            ('train/hypervolume', 'Hypervolume vs Timesteps', 'Hypervolume', 'hypervolume_vs_timesteps.png'),
            ('eval/expected_utility', 'Expected Utility vs Timesteps', 'Expected Utility', 'expected_utility_vs_timesteps.png'),
        ]
        for tag, title, ylabel, filename in scalar_plots:
            steps, values = _load_scalar(log_file, tag)
            if steps is not None:
                _save_line_plot(steps, values, title, ylabel, output_dir / filename)

        steps, values = _load_ndarray(log_file, 'eval/return/value')
        if steps is not None:
            _save_eval_return_plot(steps, values, output_dir / 'evaluation_return_vs_timesteps.png')

        _, values = _load_ndarray(log_file, 'train/leaves/r')
        if values is not None:
            _save_pareto_front_plot(values, output_dir / 'pareto_front.png')
