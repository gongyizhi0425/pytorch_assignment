# Project Guidelines

## Architecture
- This workspace has four top-level directories:
  - `CNN/` — CIFAR-10 / ResNet-18 CNN pruning, quantization, search-space evaluation, and reporting. Contains `src/`, `configs/`, `data/`, and `results_*/`.
  - `LLM/` — LLM pruning (Wanda), AWQ, SmoothQuant, and attention analysis driven by YAML configs. Contains `src/`, `configs/`, and `results_*/`.
  - `tools/` — Standalone utility scripts (CSV-to-Excel converter, JSON rewriter, calibration viewer).
  - `basic/` — Environment verification and general model benchmarking (`hello_torch.py`, `MemoryMeaseurement.py`).
- Treat `CNN/src/pruning_common.py` as the primary shared CNN utility module for loaders, model construction, metrics, checkpoints, and CSV output. Prefer updating shared logic there instead of duplicating fixes across entry scripts.
- Treat `LLM/src/llm_prune/` as the reusable LLM core. Keep orchestration in `LLM/src/run_*.py` and reusable logic in the package modules.
- `CNN/src/pruning_experiment.py` contains older overlapping CNN logic. Do not update it unless the task explicitly targets the legacy path.
- Result directories under `CNN/results_*/` and `LLM/results_*/` are tracked experiment artifacts. Avoid editing generated outputs unless the task is specifically about results or analysis.

## Build And Validation
- Create a virtual environment and install dependencies with `python3 -m venv venv && source venv/bin/activate && pip install -U pip && pip install -r requirements.txt`.
- There is no formal `pytest` or `unittest` suite in the repo. Validate changes with targeted script checks and the smallest safe run that covers the changed path.
- Safe first checks from the repo root:
  - `python basic/hello_torch.py`
  - `python CNN/src/run_baseline.py --help`
  - `python LLM/src/run_llm_wanda.py --config LLM/configs/llm_wanda.yaml --help`
- Heavy experiment runs can download datasets or Hugging Face models and may require CUDA. Do not launch them by default unless the user asks for execution.

## Conventions
- Default working directory should be the repo root. Use repo-root-relative paths in commands.
- CNN training and pruning scripts in `CNN/src/` are GPU-only by design. `require_cuda()` in `CNN/src/pruning_common.py` enforces this. Do not add CPU fallbacks unless requested.
- LLM experiments are config-first. Prefer changing YAML files in `LLM/configs/` over hardcoding values in Python.
- Preserve result naming patterns based on `out_prefix` and keep CSV schemas stable. Downstream plotting and Excel scripts depend on those names and columns.
- Preserve the CIFAR-10-specific ResNet-18 stem used for 32x32 inputs unless the task explicitly changes model assumptions.
- Follow the existing style: type hints, dataclasses for structured outputs, small reusable helpers, and optional dependency guards where appropriate.

## Project-Specific Pitfalls
- Many scripts are expensive: CNN scripts may auto-download CIFAR-10, and LLM scripts may download large Hugging Face assets. Prefer config inspection, `--help`, or reduced sample counts before full runs.
- CNN scripts default to `--data-dir ./data`. When running from `CNN/src/` use `--data-dir ../data` to keep data in `CNN/data/`.
- `LLM/src/run_llm_wanda.py` resolves relative config paths for either repo-root or `LLM/` execution. Keep that behavior intact when touching config loading.
- Activation-pruning code relies on hooks tied to the current CNN model structure. If the model topology changes, review the hook logic before assuming it still applies.