Do a test for environment:
python hello_torch.py 

Go to venv:
source venv/bin/activate

Do the baseline training:
python run_baseline.py --device cuda --amp --eval-every 1 --early-stop-acc 0.90 --out-prefix hw90

Do the weight pruning:
python run_weight_pruning.py --out-prefix hw90_wt --baseline-prefix hw90 --channel-impl mask --weight-sparsity 0.9  


Draw the plot & excel:
python plot_and_excel.py --out-prefix hw90_combo --baseline-prefix hw90 --weight-prefix hw90_wt --activation-prefix hw90_act_ft


## LLM Wanda Pruning (new)

This repo now also includes a runnable implementation for the in-class assignment:
baseline + magnitude pruning + WANDA pruning on a HuggingFace causal LM, with
perplexity + speed + sparsity metrics and PPL-vs-sparsity plots.

Install deps:

```bash
pip install -r ../requirements.txt
```

Run (from repo root):

```bash
python src/run_llm_wanda.py --config configs/llm_wanda.yaml
```

Outputs go to `results_llm/` by default.


## Attention Sink Analysis (new)

Key PG-19 download/streaming code (inside the script):

```python
from datasets import load_dataset
ds = load_dataset("pg19", split="test", streaming=True)
```

Run (from repo root):

```bash
python src/run_attention_sink.py --config configs/attention_sink.yaml --cache-dir /home/gyz/hf_cache --local-files-only
```

Outputs go to `results_attention/` by default.


# PyTorch Demo (CNN Metrics + Pruning)

This repo contains two experiment tracks:

1) **CNN efficiency comparison** (peak memory + compute): ResNet-18 vs MobileNetV3
2) **CNN pruning on CIFAR-10**: baseline + weight pruning + activation-based pruning, with CSV/plots/Excel

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