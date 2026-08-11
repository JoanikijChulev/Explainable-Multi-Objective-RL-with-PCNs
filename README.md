# Explainable Multi-Objective Reinforcement Learning with Pareto-Conditioned Policies

An end-to-end research codebase for training Pareto-Conditioned Networks (PCNs), explaining their decisions, exploring trained policies interactively, and comparing reward-only conditioning against reward-plus-horizon conditioning.

The repository brings four connected capabilities into one tree:

- controlled reward-only (`R`) versus reward-and-horizon (`RH`) PCN training and evaluation.
- reward-conditioned PCN training, resumption, evaluation, and rollout rendering;
- command-space counterfactual explanations with CF-ZOO and Carlini-Wagner (C&W);
- post-hoc goal-influence signals based on KL, total variation, Jensen-Shannon, and action-flip behavior;

All generated data stays in component-specific artifact directories. The browser application runs locally on `127.0.0.1` and includes 30 model checkpoints: an `R` and `RH` model for each of 15 environments.

## System map

~~~text
environment + PCN training
          |
          v
checkpoint + log + run metadata
          |
          +--> evaluation and rollout rendering
          |
          +--> CF-ZOO / C&W command counterfactuals
          |
          +--> goal-influence profiles and faithfulness analysis
~~~

## Repository layout

| Path | Purpose |
| --- | --- |
| [`pcn/`](pcn/) | Root reward-conditioned PCN models, training loop, replay handling, and environment construction. |
| [`envs/`](envs/) | Local environments: Collect Two, Branch Path, Reward Line, Three Tree, and WalkRoom. |
| [`main_pcn.py`](main_pcn.py) | Train a new root PCN, warm-start from a checkpoint, or resume a complete training run. |
| [`eval_pcn.py`](eval_pcn.py) | Replay logged Pareto commands and compare requested with achieved returns. |
| [`render_pcn.py`](render_pcn.py) | Render selected policy rollouts to frames and video artifacts. |
| [`custom_run_pcn.py`](custom_run_pcn.py) | Prompt for a checkpoint and arbitrary desired-return command, then render the rollout. |
| [`interactive_pcn_zoo_cf.py`](interactive_pcn_zoo_cf.py) | Prompt-driven CF-ZOO command-space counterfactual explanation. |
| [`interactive_pcn_cw.py`](interactive_pcn_cw.py) | Prompt-driven C&W command-space counterfactual explanation through ART. |
| [`evaluate_pcn_zoo_counterfactuals.py`](evaluate_pcn_zoo_counterfactuals.py) | Batch CF-ZOO evaluation over runs, front rows, timesteps, and foil actions. |
| [`diagnose_zoo_failures.py`](diagnose_zoo_failures.py) | Dense command-grid diagnosis of unsuccessful batch cases. |
| [`evaluate_cw_vs_zoo_paper_experiments.py`](evaluate_cw_vs_zoo_paper_experiments.py) | Paired C&W-versus-ZOO benchmarking and landscape experiments. |
| [`generate_showcase_landscapes.py`](generate_showcase_landscapes.py) | Select and plot representative real command-space landscapes. |
| [`goal_influence/`](goal_influence/) | Single-run and cross-environment goal-influence analysis. |
| [`interactive_app/`](interactive_app/) | Local browser app, environment descriptions, bundled checkpoints, and verified fronts. |
| [`pcn_reward_vs_reward_horizon_comparison/`](pcn_reward_vs_reward_horizon_comparison/) | Self-contained `R` versus `RH` training and comparison implementation. |
| [`output_artifacts/`](output_artifacts/) | Root training runs and root explanation/evaluation artifacts. |
| [`environment.yml`](environment.yml) | Recommended Conda environment. |
| [`requirements.txt`](requirements.txt) | Pinned pip-installable dependencies. |

The comparison folder intentionally contains its own `pcn/`, `envs/`, artifact helpers, metrics, and CLI entry points. Run its commands from inside that folder so its local modules are imported.

## Requirements

Recommended setup:

- Windows, Linux, or macOS with a 64-bit Python environment;
- Python 3.12;
- Miniforge, Mambaforge, or Conda;
- FFmpeg available on `PATH` for MP4 export;
- a CUDA-capable PyTorch installation is optional.

PyGMO does not provide a suitable Windows PyPI wheel for this setup, so the Conda environment is the most reliable installation route.

## Installation

### Recommended Conda environment

From the repository root:

~~~bash
conda env create -f environment.yml
conda activate pcn-cfzoo
python -m pip check
~~~

### Pip-oriented environment

Install Python, PyGMO, and FFmpeg with Conda first, then install the pinned pip packages:

~~~bash
conda create -n pcn-cfzoo-pip -c conda-forge python=3.12 pygmo=2.19.7 ffmpeg
conda activate pcn-cfzoo-pip
python -m pip install -r requirements.txt
python -m pip check
~~~

The code selects CUDA when available and otherwise runs on CPU. Large training runs and dense counterfactual studies can be substantially slower on CPU.

## Quick start

### Use the bundled Minecart run

The root artifact bundle includes an archived Minecart checkpoint, training log, evaluation outputs, counterfactual outputs, and plots:

~~~bash
python eval_pcn.py Minecart --checkpoint 10
python render_pcn.py Minecart --checkpoint 10 --policy-index 0
python custom_run_pcn.py Minecart
python goal_influence/goal_influence.py --run Minecart --checkpoint 10 --front 0
~~~

### Launch the interactive application

~~~bash
python interactive_app/app.py
~~~

Open:

~~~text
http://127.0.0.1:8901
~~~

To choose another port:

~~~bash
python interactive_app/app.py 9000
~~~

The application lets you:

1. choose one of the bundled `R` or `RH` models;
2. select a verified desired-return command and, for `RH`, its horizon;
3. start and step a rollout;
4. inspect observations, remaining commands, action probabilities, and valid actions;
5. move backward through the current rollout;
6. request a foil action and run CF-ZOO or C&W;
7. continue from the changed action to verify the realized outcome;
8. save a text report;
9. import or delete trusted custom model/front pairs.

The server is single-threaded because MuJoCo off-screen rendering contexts are thread-bound. It binds only to the loopback interface and is not intended to be exposed directly to a network.

## Root PCN workflow

### Train a new run

Both `--env` and `--run-name` are required:

~~~bash
python main_pcn.py --env dst --run-name dst_seed0
~~~

Override the environment's default absolute training budget when needed:

~~~bash
python main_pcn.py --env reward-line --run-name reward_line_smoke --total-steps 10000
~~~

A new run writes to `output_artifacts/<run-name>/`. Run names may contain letters, numbers, periods, underscores, and hyphens.

### Warm-start versus resume

Warm-starting loads model weights into a new run:

~~~bash
python main_pcn.py --env dst --run-name dst_warmstart --model PATH_TO_CHECKPOINT
~~~

Full resumption continues an existing run in place and restores its saved training state:

~~~bash
python main_pcn.py --resume-run dst_seed0 --total-steps 400000
~~~

For resumption, `--total-steps` is the new absolute target and must be larger than the saved step count. `--resume-run` cannot be combined with `--model`.

### Evaluate a run

~~~bash
python eval_pcn.py RUN_NAME
python eval_pcn.py RUN_NAME --checkpoint 10
python eval_pcn.py RUN_NAME --interactive
~~~

A checkpoint can be selected by index, filename, or full path. Without `--checkpoint`, the latest numbered checkpoint is used.

### Render logged policies

~~~bash
python render_pcn.py RUN_NAME --policy-index 0
python render_pcn.py RUN_NAME --checkpoint 10 --interactive --fps 4
~~~

WalkRoom training and evaluation are supported, but step-by-step WalkRoom rendering is not.

### Render an arbitrary command

~~~bash
python custom_run_pcn.py RUN_NAME
~~~

If the model argument is omitted, the script prompts for an available root run. It then prompts for the desired-return vector.

## Counterfactual workflows

### Interactive CF-ZOO

~~~bash
python interactive_pcn_zoo_cf.py
~~~

The prompt selects a run, checkpoint, logged front row, trajectory state, foil action, and search settings. Results are saved under:

~~~text
output_artifacts/<run>/counterfactuals/zoo_cf_interactive_<timestamp>/
~~~

### Interactive C&W

~~~bash
python interactive_pcn_cw.py
~~~

ART creates a local `.art/` configuration directory on import. That directory is ignored because it is machine-local runtime state.

### Batch CF-ZOO evaluation

Bounded example:

~~~bash
python evaluate_pcn_zoo_counterfactuals.py --runs Minecart --max-fronts 2 --max-timesteps 3 --foil-top-k 1 --max-cases 20
~~~

Without `--runs`, the command examines candidate run directories under `output_artifacts/`. Use `--max-cases` for a wiring check; omit limits only when a full research-scale evaluation is intended.

Outputs include case-level CSV data, failures, per-environment summaries, per-front summaries, configuration, and JSON summaries under:

~~~text
output_artifacts/pcn_zoo_cf_evaluations/<label>_<timestamp>/
~~~

### Diagnose failures

~~~bash
python diagnose_zoo_failures.py --eval-dir output_artifacts/pcn_zoo_cf_evaluations/EVAL_DIRECTORY
~~~

The diagnostic scans a command grid to distinguish optimizer-budget failures from cases where the requested foil is unavailable in the searched command region.

### Paired C&W-versus-ZOO benchmark

~~~bash
python evaluate_cw_vs_zoo_paper_experiments.py --runs Minecart --max-fronts 2 --max-timesteps 3 --foil-top-k 1 --max-benchmark-cases 20
~~~

The output contains long-form and paired case tables, summaries by method/environment/run, saved configuration, and selected landscapes.

### Representative command landscapes

~~~bash
python generate_showcase_landscapes.py --runs Minecart --max-candidates-per-run 12 --shortlist 2
~~~

This screens real policy states and command dimensions, then writes selected cases, numerical data, summaries, and decision plots.

## Goal-influence workflow

Run these commands from the repository root.

### Explain one logged policy

~~~bash
python goal_influence/goal_influence.py --run RUN_NAME --checkpoint 10 --front 0
~~~

### Explain all or a bounded subset of logged policies

~~~bash
python goal_influence/goal_influence.py --run RUN_NAME --front all
python goal_influence/goal_influence.py --run RUN_NAME --front all --max-fronts 5
~~~

### Add validation and state-basis aggregation

~~~bash
python goal_influence/goal_influence.py --run RUN_NAME --front 0 --validate --state-basis
~~~

`--validate` compares the influence scores with a dense command-box behavioral scan. `--state-basis` groups repeated visits to the same state and reports how influence varies across commands at that state.

Single-run outputs are written to:

~~~text
goal_influence/goal_output_artifacts/<label>_<timestamp>/
~~~

Typical outputs include:

- `influence_profile.csv` with per-state distributions, actions, residual commands, and four influence scores;
- `summary.json` with run, checkpoint, method, query, and validation metadata;
- a trajectory influence plot;
- optional `state_basis.csv` and state-basis statistics.

### Cross-environment evaluation

The evaluator has configured keys for `dst`, `bp`, `reward_line`, `three_tree`, `fourroom2`, `bb`, `rsg2`, `c2`, `MC3`, and `ft`. Each key maps to an expected trained run name in `output_artifacts/`.

~~~bash
python goal_influence/evaluate_goal_influence_envs.py --envs dst bp reward_line --out-label study_name
python goal_influence/aggregate_goal_influence_results.py --out-label study_name
~~~

The same `--out-label` must be passed to the evaluator and aggregator. Aggregated outputs include pooled state rows, one row per environment-and-score summary, faithfulness plots, AUROC plots, flip calibration, and state-basis fork separation.

The single-run goal-influence method is generic across compatible trained reward-only PCNs. The cross-environment command is narrower because its run names and evaluation budgets are explicitly configured in `ENV_SETTINGS`.

## Reward versus reward+horizon workflow

Run comparison commands from inside the self-contained comparison folder:

~~~bash
cd pcn_reward_vs_reward_horizon_comparison
~~~

### Train and compare both variants

~~~bash
python train_compare_pcn_variants.py --env dst --variant both --seed 0
~~~

The command creates three sibling directories:

~~~text
comparison_output_artifacts/<prefix>_reward/
comparison_output_artifacts/<prefix>_horizon/
comparison_output_artifacts/<prefix>_comparison/
~~~

Use `--run-prefix` to select the prefix. If omitted, a timestamped prefix is generated.


### Train one variant

~~~bash
python main_pcn.py --env dst --run-name dst_reward --variant reward
python main_pcn.py --env dst --run-name dst_horizon --variant horizon
~~~

### Resume, evaluate, or render a comparison run

~~~bash
python main_pcn.py --resume-run dst_horizon --total-steps 400000
python eval_pcn.py dst_horizon --seed 0
python render_pcn.py dst_horizon --policy-index 0
python custom_run_pcn.py dst_horizon
~~~

Variant type is recorded in run metadata and inferred during resume/evaluation. The comparison evaluator accepts an optional rollout seed; each command then uses `seed + command_index`.

## Supported environments

| Environment | Canonical CLI name | Common aliases | Root PCN | R/RH comparison | Bundled app pair |
| --- | --- | --- | :---: | :---: | :---: |
| Deep Sea Treasure | `dst` | `deep-sea-treasure` | Yes | Yes | Yes |
| Collect Two | `collect_two` | `collect-two`, `c2` | Yes | Yes | Yes |
| Branch Path | `branch-path` | `branch_path`, `bp` | Yes | Yes | Yes |
| Reward Line | `reward-line` | `reward_line`, `rl` | Yes | Yes | Yes |
| Three Tree | `three-tree` | `three_tree`, `tt` | Yes | Yes | Yes |
| Minecart | `minecart` | `mc` | Yes | Yes | Yes |
| Fruit Tree | `fruit-tree-v0` | `fruit-tree`, `ft` | Yes | Yes | Yes |
| Four Room | `four-room-v0` | `four-room`, `4room` | Yes | Yes | Yes |
| Resource Gathering | `resource-gathering-v0` | `resource-gathering`, `rsg` | Yes | Yes | Yes |
| Breakable Bottles | `breakable-bottles-v0` | `breakable-bottles`, `bb` | Yes | Yes | Yes |
| WalkRoom | `walkroom2` through `walkroom9` | - | Yes | Yes | 2 and 3 |
| MO Mountain Car Time-Speed | `mo-mountaincar-timespeed-v0` | `momc`, `mountaincar` | No | Yes | Yes |
| MO Lunar Lander | `mo-lunar-lander-v3` | `moll`, legacy `v2` alias | No | Yes | Yes |
| MO Reacher | `mo-reacher-v5` | `mor`, `reacher` | No | Yes | Yes |

Fruit Tree accepts `--fruit-tree-depth 5`, `6`, or `7`. The default is `6`.

## Interactive model bundle

Each of the 15 app environment directories contains:

~~~text
<environment>_R.pt
<environment>_RH.pt
<environment>_R_achievable_within_1pct.txt
<environment>_RH_achievable_within_1pct.txt
~~~

The verified front files separate the command used to query the policy from the achieved target return:

- `R` rows contain `n` command values followed by `n` achieved-return values;
- `RH` rows contain `n` command values, one positive integer horizon, and `n` achieved-return values.

Custom uploads must match the selected environment, variant, objective count, and expected model interface.

## Artifact layout

New root runs use:

~~~text
output_artifacts/
|-- <run-name>/
|   |-- checkpoints/
|   |-- evaluations/
|   |-- animation_render/
|   |-- custom_runs/
|   |-- counterfactuals/
|   |-- plots/
|   |-- log.h5
|   |-- run_config.json
|   +-- training_state.pt
|-- pcn_zoo_cf_evaluations/
|-- cw_vs_zoo_paper_experiments/
+-- showcase_landscapes/
~~~

Goal-influence studies use:

~~~text
goal_influence/goal_output_artifacts/
|-- <single-run-label>_<timestamp>/
+-- <cross-environment-label>/
    |-- per_env/
    |-- plots/
    |-- cross_env_summary.csv
    +-- states_all.csv
~~~

R/RH experiments use:

~~~text
pcn_reward_vs_reward_horizon_comparison/comparison_output_artifacts/
|-- <prefix>_reward/
|-- <prefix>_horizon/
+-- <prefix>_comparison/
~~~

## Complete CLI index

Run `--help` for current defaults and every optional budget flag.

| Working directory | Command | Purpose |
| --- | --- | --- |
| Root | `python main_pcn.py --help` | Train, warm-start, or resume a reward-conditioned PCN. |
| Root | `python eval_pcn.py --help` | Evaluate logged Pareto commands. |
| Root | `python render_pcn.py --help` | Render logged policy rollouts. |
| Root | `python custom_run_pcn.py --help` | Render a custom desired-return command. |
| Root | `python interactive_pcn_zoo_cf.py` | Run prompt-driven CF-ZOO. |
| Root | `python interactive_pcn_cw.py` | Run prompt-driven C&W. |
| Root | `python evaluate_pcn_zoo_counterfactuals.py --help` | Run a batch CF-ZOO study. |
| Root | `python diagnose_zoo_failures.py --help` | Diagnose unsuccessful ZOO cases on command grids. |
| Root | `python evaluate_cw_vs_zoo_paper_experiments.py --help` | Run paired C&W/ZOO experiments. |
| Root | `python generate_showcase_landscapes.py --help` | Generate selected command-space landscapes. |
| Root | `python goal_influence/goal_influence.py --help` | Explain one or more logged policies. |
| Root | `python -m goal_influence.goal_influence --help` | Equivalent package-style goal-influence invocation. |
| Root | `python goal_influence/evaluate_goal_influence_envs.py --help` | Run configured cross-environment influence evaluation. |
| Root | `python goal_influence/aggregate_goal_influence_results.py --help` | Aggregate a cross-environment influence study. |
| Root | `python interactive_app/app.py [PORT]` | Launch the local application. |
| Comparison | `python train_compare_pcn_variants.py --help` | Train and compare `R`/`RH` variants under a shared budget. |
| Comparison | `python main_pcn.py --help` | Train or resume one comparison variant. |
| Comparison | `python eval_pcn.py --help` | Evaluate a comparison run. |
| Comparison | `python render_pcn.py --help` | Render a comparison run. |
| Comparison | `python custom_run_pcn.py --help` | Run a custom `R` or `RH` command. |

## Operational checks

After installation:

~~~bash
python -m pip check
python main_pcn.py --help
python evaluate_pcn_zoo_counterfactuals.py --help
python goal_influence/goal_influence.py --help
~~~


## Additional use info

- PyTorch `.pt` files use Python serialization. Loading an untrusted checkpoint can execute code.
- The interactive server is for localhost use.
- C&W imports may create `.art/`; it is ignored as machine-local runtime state.

## Troubleshooting

### PyGMO cannot be installed with pip on Windows

Use the supplied Conda environment or install `pygmo=2.19.7` from `conda-forge` before running `pip install -r requirements.txt`.

### Box2D installation fails

Use the pinned Conda/pip setup. `swig` and `Box2D` are already included in the dependency files.

### MP4 export fails

Confirm FFmpeg is installed and visible:

~~~bash
ffmpeg -version
~~~

### MuJoCo rendering fails or produces no frames

Check the graphics driver and MuJoCo installation. Use the local app's serialized server process; do not replace it with a threaded server.

## Research code basis

This repository code builds on:

- Reymond, M., Bargiacchi, E., and Now&eacute;, A. (2022). *Pareto Conditioned Networks*. Proceedings of AAMAS 2022, 1110-1118. [Paper](https://www.ifaamas.org/Proceedings/aamas2022/pdfs/p1110.pdf)
- Chen, P.-Y. et al. (2017). *ZOO: Zeroth Order Optimization Based Black-box Attacks to Deep Neural Networks without Training Substitute*. ACM AISec.
