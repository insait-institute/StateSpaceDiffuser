# UniverseLab

Benchmarks and data processing for world models.

## Setup

Run the setup script from the repo root:

```bash
bash install.sh
```

CUDA is required. Ensure CUDA is installed and your environment exposes the correct CUDA paths (these may vary by machine).

## Benchmarking

### Structure

Benchmark runs are driven by `run_bench.py` and Hydra configs under `config/`.

- **Models**: Wrapper classes in `models/` (e.g., Mamba WM, SSW, Diamond).
- **Tasks**: Evaluation logic in `tasks/` (e.g., match ground truth, reverse action).
- **Datasets**: Data loaders in `data/` for benchmarking (separate from training loaders).
- **Metrics**: Implementations in `metrics/` used by tasks.

The flow is:

1) `run_bench.py` loads a Hydra config (defaults in `config/config.yaml`).
2) `tools/build_tools.py` builds the model, dataset, and task.
3) `bench/engine.py` iterates over data and executes the task.

### Models (API)

Models are `WorldModelBase` subclasses and must implement:

- `forward(observations, actions, extras, **kwargs) -> dict`
- `imagine(...)` if imagination is supported

Relevant paths:

- `bench/model.py`
- `models/*.py`

### Tasks (API)

Tasks are `TaskBase` subclasses and must implement:

- `execute(model, input_batch) -> dict[str, Any]`

Tasks read from the batch dict:

- `observations`
- `actions`
- `extras` (e.g., `frames_hq`)

Relevant paths:

- `bench/task.py`
- `tasks/*.py`

### Datasets (API)

Bench datasets return a dict with at least:

- `observations`
- `actions`
- `extras` (optional)

Relevant paths:

- `bench/dataset.py`
- `data/*.py`

### Metrics (API)

Metrics are used by tasks and should implement:

- `reset()`
- `update(prediction, target) -> float`

Relevant paths:

- `bench/metric.py`
- `metrics/*.py`

## Examples

Run a benchmark with an explicit checkpoint:

```bash
python run_bench.py \
  common.ckpt_load=/path/to/checkpoint.pt \
  common.ckpt_enable_manager=false \
  data=minigrid \
  model=mamba_wm \
  data.data_root_dpath=/path/to/datasets \
  task=match_ground_truth \
  bench.batch_size=2 \
  task.qualitative_dpath=outputs/gt_vis \
  model.enable_imagine=false
```

## Task Options

Each task exposes its own options via `task.*` overrides. Common ones include:

- `task.qualitative_dpath`: save qualitative outputs (supported by match_ground_truth, reverse_action)
- `task.action`: base action for reverse_action (dataset-specific enum name)
- `task.n_steps`: task-specific horizon (where applicable)

## Imagination

Imagination is controlled by the model config flag:

- `model.enable_imagine=true|false`

Tasks may override behavior (e.g., `reverse_action` always uses imagination), so set this explicitly when you want deterministic behavior across tasks.

## Checkpoint Loading

Checkpoint loading is controlled via `common.*` options:

- `common.ckpt_enable_manager=true|false`
  - When `true`, `common.ckpt_load` can be a tuple `(model_id, step)` or a path.
  - When `false`, `common.ckpt_load` is treated as a direct file path.
- `common.ckpt_dpath`: root directory for checkpoint runs (used by the manager).
- `common.ckpt_central_dpath`: optional secondary root for checkpoint syncing (created if set).
- `common.ckpt_load`: either a tuple `(id, step)` or a direct path.

Examples:

```bash
# Manager mode with (id, step)
python run_bench.py common.ckpt_enable_manager=true \
  common.ckpt_dpath=/path/to/checkpoints \
  common.ckpt_load="(239, 71000)"

# Direct path mode
python run_bench.py common.ckpt_enable_manager=false \
  common.ckpt_load=/path/to/checkpoint.pt
```
