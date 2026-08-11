import json
import re
from datetime import datetime
from pathlib import Path


RUN_METADATA_FILENAME = 'run_config.json'
TRAINING_STATE_FILENAME = 'training_state.pt'
_VALID_RUN_NAME = re.compile(r'[A-Za-z0-9][A-Za-z0-9._-]*')


def repo_root():
    return Path(__file__).resolve().parent


def output_artifacts_dir():
    path = repo_root() / 'output_artifacts'
    path.mkdir(parents=True, exist_ok=True)
    return path


def checkpoints_dir(run_dir, create=False):
    path = Path(run_dir) / 'checkpoints'
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def training_state_path(run_dir):
    return Path(run_dir) / TRAINING_STATE_FILENAME


def plots_dir(run_dir, create=False):
    path = Path(run_dir) / 'plots'
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def evaluations_dir(run_dir, create=False):
    path = Path(run_dir) / 'evaluations'
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def create_evaluation_dir(run_dir):
    root = evaluations_dir(run_dir, create=True)
    path = root / f'eval_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    path.mkdir(parents=True, exist_ok=False)
    return path.resolve()


def animation_render_dir(run_dir, create=False):
    path = Path(run_dir) / 'animation_render'
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def create_animation_render_dir(run_dir):
    root = animation_render_dir(run_dir, create=True)
    path = root / f'render_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    path.mkdir(parents=True, exist_ok=False)
    return path.resolve()


def custom_runs_dir(run_dir, create=False):
    path = Path(run_dir) / 'custom_runs'
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def create_custom_run_dir(run_dir):
    root = custom_runs_dir(run_dir, create=True)
    path = root / f'custom_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    path.mkdir(parents=True, exist_ok=False)
    return path.resolve()


def validate_run_name(run_name):
    if not _VALID_RUN_NAME.fullmatch(run_name):
        raise ValueError(
            'run names must start with a letter or number and use only letters, numbers, ".", "_" or "-"'
        )
    return run_name


def create_run_dir(run_name):
    run_name = validate_run_name(run_name)
    run_dir = output_artifacts_dir() / run_name
    if run_dir.exists():
        raise FileExistsError(f'run directory already exists: {run_dir}')
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir.resolve()


def resolve_run_dir(run):
    candidate = Path(run)
    if not candidate.is_absolute() and len(candidate.parts) == 1:
        artifact_dir = output_artifacts_dir() / run
        if artifact_dir.is_dir():
            return artifact_dir.resolve()
    if candidate.is_dir():
        return candidate.resolve()
    raise FileNotFoundError(f'could not find run directory: {run}')


def write_run_metadata(run_dir, metadata):
    metadata_path = Path(run_dir) / RUN_METADATA_FILENAME
    with metadata_path.open('w', encoding='utf-8') as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write('\n')
    return metadata_path


def load_run_metadata(run_dir):
    metadata_path = Path(run_dir) / RUN_METADATA_FILENAME
    if not metadata_path.is_file():
        return {}
    with metadata_path.open('r', encoding='utf-8') as handle:
        return json.load(handle)
