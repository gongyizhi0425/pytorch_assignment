# PyTorch Demo (CNN Metrics + Pruning)

This repo contains two experiment tracks:

1) **CNN efficiency comparison** (peak memory + compute): ResNet-18 vs MobileNetV3
2) **CNN pruning on CIFAR-10**: baseline + weight pruning + activation-based pruning, with CSV/plots/Excel

It also contains a third track:

3) **LLM pruning (baseline / magnitude / WANDA)** on a HuggingFace causal LM, with perplexity + speed + sparsity metrics.

## 0) Environment

Recommended:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

Notes:
- The pruning scripts under `src/` are configured as **GPU-only** (`--device cuda/auto` + internal CUDA checks).
- The older all-in-one pruning script at repo root can run on CPU.

## 1) Quick GPU sanity check

```bash
python3 hello_torch.py
```

## 2) CNN efficiency comparison (peak metrics)

Script: `MemoryMeaseurement.py`

Example (CPU):

```bash
python3 MemoryMeaseurement.py --device cpu --models resnet18,mobilenet_v3_small --batch-sizes 1-16 --resolutions 224 --out-prefix cnn_compare
```

Outputs:
- `cnn_compare_static_metrics.csv`
- `cnn_compare_runtime_metrics.csv`
- `cnn_compare_*.png`

## 3) Pruning assignment (GPU-only pipeline)

All scripts are in `src/`. Run from the `src/` directory:

```bash
cd src
```

### 3.1 Baseline (produces checkpoint)

```bash
../venv/bin/python run_baseline.py \
  --device auto \
  --out-prefix hw90 \
  --epochs 0 \
  --resolution 32
```

Outputs:
- `hw90_baseline.csv`
- `hw90_baseline.pt`
- (if `--eval-every` enabled) `hw90_baseline_best.pt`

### 3.2 Weight pruning (fine / channel / N:M)

```bash
../venv/bin/python run_weight_pruning.py \
  --device auto \
  --ckpt hw90_baseline_best.pt \
  --out-prefix hw90_wt \
  --ratios 0.3,0.5,0.8 \
  --nm-patterns 1:4,2:4,3:4 \
  --channel-impl slim \
  --resolution 32 \
  --test-limit 1000
```

Output:
- `hw90_wt_weight.csv`

### 3.3 Activation-based channel pruning (mask or true slimming)

```bash
../venv/bin/python run_activation_pruning.py \
  --device auto \
  --ckpt hw90_baseline_best.pt \
  --out-prefix hw90_act_ft \
  --impl slim \
  --ratios 0.3,0.5,0.8 \
  --resolution 32 \
  --test-limit 1000
```

Output:
- `hw90_act_ft_activation.csv`

### 3.4 Merge to Excel + plots

```bash
../venv/bin/python plot_and_excel.py \
  --out-prefix hw90_combo \
  --baseline-prefix hw90 \
  --weight-prefix hw90_wt \
  --activation-prefix hw90_act_ft
```

Outputs:
- `hw90_combo_results.xlsx` (multi-sheet: baseline/fine/channel/nm/actchannel)
- `hw90_combo_*.png`

## 4) Dataset location

CIFAR-10 is downloaded automatically to:
- `./data/` (for repo-root scripts)
- `./src/data/` (for `src/` scripts when you keep default `--data-dir ./data` while running inside `src/`)

To unify dataset location, always pass an absolute or repo-root path, e.g. `--data-dir ../data` when running from `src/`.

## 5) LLM Wanda pruning assignment

Config file: `LLM_pruning/configs/llm_wanda.yaml`

Run:

```bash
./venv/bin/python LLM_pruning/src/run_llm_wanda.py --config LLM_pruning/configs/llm_wanda.yaml
```

Outputs:
- `LLM_pruning/results_llm/<run_name>_calibration.csv` (baseline + magnitude + wanda sweeps)
- `LLM_pruning/results_llm/<run_name>_calibration_shifted.csv` (domain-shifted calibration to induce Wanda degradation)
- `LLM_pruning/results_llm/*_ppl_vs_sparsity.png`

## 6) Attention Sink analysis (PG-19)

Config file: `LLM_pruning/configs/attention_sink.yaml`

Run (recommended to use the repo venv explicitly):

```bash
./venv/bin/python LLM_pruning/src/run_attention_sink.py \
  --config LLM_pruning/configs/attention_sink.yaml \
  --cache-dir /home/gyz/hf_cache \
  --local-files-only
```

Notes:
- PG-19 is loaded via streaming with `trust_remote_code=True` (see `src/attention_sink/pg19.py`).
- Perplexity is computed on a fixed suffix (last `analysis.eval_last_n` tokens) so interventions change only context while keeping the evaluated targets comparable.

Outputs (in `LLM_pruning/results_attention/`):
- `<run_name>_L{L}_curve.csv/.png`: attention-to-key-position curve for each length
- `<run_name>_sink_summary.csv`: top-k sink positions per length
- `<run_name>_ppl_interventions.csv`: per-excerpt PPL for each intervention
- `<run_name>_ppl_summary.csv`: mean/std + ΔPPL vs baseline
- `<run_name>_delta_ppl.png`: bar plot of ΔPPL vs baseline
