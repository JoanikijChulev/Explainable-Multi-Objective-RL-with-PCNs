from pathlib import Path

import h5py
import numpy as np


FINAL_COMMANDS_GROUP = 'train/final_commands'
FINAL_RETURNS_DATASET = f'{FINAL_COMMANDS_GROUP}/returns'
FINAL_HORIZONS_DATASET = f'{FINAL_COMMANDS_GROUP}/horizons'


def _commands_from_replay_entries(replay_entries):
    if not replay_entries:
        raise ValueError('cannot save final commands from an empty experience replay')

    returns = []
    horizons = []
    for entry in replay_entries:
        transitions = entry[2]
        if not transitions:
            raise ValueError('experience replay contains an empty episode')
        returns.append(np.asarray(transitions[0].reward, dtype=np.float32))
        horizons.append(len(transitions))

    try:
        returns = np.stack(returns, axis=0).astype(np.float32, copy=False)
    except ValueError as exc:
        raise ValueError('experience replay contains inconsistent reward dimensions') from exc
    if returns.ndim != 2:
        raise ValueError(f'expected vector returns, got array shape {returns.shape}')

    horizons = np.asarray(horizons, dtype=np.float32)
    return returns, horizons


def _write_resizable_dataset(group, name, values):
    expected_tail = values.shape[1:]
    if name in group:
        dataset = group[name]
        can_resize = (
            dataset.ndim == values.ndim
            and dataset.shape[1:] == expected_tail
            and dataset.maxshape is not None
            and dataset.maxshape[0] is None
        )
        if not can_resize:
            del group[name]
            dataset = None
    else:
        dataset = None

    if dataset is None:
        dataset = group.create_dataset(
            name,
            shape=values.shape,
            maxshape=(None,) + expected_tail,
            dtype=values.dtype,
        )
    else:
        dataset.resize(values.shape)
    dataset[...] = values


def write_final_replay_commands(run_dir, replay_entries, step):
    """Write the latest paired return/horizon commands, replacing any prior snapshot."""
    returns, horizons = _commands_from_replay_entries(replay_entries)
    log_path = Path(run_dir) / 'log.h5'
    if not log_path.is_file():
        raise FileNotFoundError(f'cannot save final commands; run log does not exist: {log_path}')

    with h5py.File(log_path, 'r+') as log:
        group = log.require_group(FINAL_COMMANDS_GROUP)
        _write_resizable_dataset(group, 'returns', returns)
        _write_resizable_dataset(group, 'horizons', horizons)
        group.attrs['step'] = int(step)
        group.attrs['count'] = len(returns)
        group.attrs['selection'] = 'best half of final experience replay'
        log.flush()
