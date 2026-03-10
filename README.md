# PyTorch Experiments — CNN Pruning / Quantization & LLM Compression

> 基于 PyTorch 的深度学习模型压缩实验合集，涵盖 CNN 剪枝 / 量化和 LLM 压缩（剪枝、注意力分析、量化）。

## 目录结构总览

```
pytorch_demo/
├── README.md                 ← 本文件
├── requirements.txt          ← Python 依赖
├── .gitignore
│
├── src/                      ← CNN 实验源码（剪枝 + 量化）
│   ├── hello_torch.py            # 环境验证（PyTorch / CUDA / GPU 信息）
│   ├── MemoryMeaseurement.py     # CNN 效率对比（峰值内存 + 计算量）
│   ├── pruning_common.py         # 剪枝公共工具（数据加载、指标、模型工具）
│   ├── pruning_experiment.py     # 剪枝实验框架（fine/channel/N:M 剪枝引擎）
│   ├── run_baseline.py           # CIFAR-10 ResNet-18 基线训练
│   ├── run_weight_pruning.py     # 权重剪枝实验入口
│   ├── run_activation_pruning.py # 激活值剪枝实验入口
│   ├── run_quantization.py       # CNN K-means vs Linear 量化（PTQ）
│   ├── plot_and_excel.py         # 汇总 CSV → Excel + 对比图表
│   └── readme.md                 # CNN 运行命令速查
│
├── CNN_pruning/              ← CNN 实验配置 + 结果
│   ├── configs/
│   │   ├── test_config.yaml          # 剪枝测试配置
│   │   └── quantization_config.yaml  # K-means/Linear 量化配置
│   ├── results_root/             # 调试/sanity check 实验结果
│   ├── results_src/              # 正式剪枝实验结果（hw90 系列）
│   └── results_quantization/     # K-means vs Linear 量化结果
│
├── LLM_pruning/              ← LLM 实验源码 + 配置 + 结果
│   ├── configs/
│   │   ├── llm_wanda.yaml        # Wanda 剪枝配置
│   │   ├── attention_sink.yaml   # Attention Sink 分析配置
│   │   ├── smoothquant.yaml      # SmoothQuant W8A8 配置
│   │   └── awq.yaml              # AWQ W4A16 配置
│   ├── src/
│   │   ├── run_llm_wanda.py      # LLM 剪枝（Baseline + Magnitude + Wanda）
│   │   ├── run_attention_sink.py # Attention Sink 分析（PG-19）
│   │   ├── run_smoothquant.py    # SmoothQuant W8A8 量化
│   │   ├── run_awq.py            # AWQ W4A16 量化
│   │   ├── rerun_error_analysis.py # AWQ 误差分析独立重跑脚本
│   │   ├── attention_sink/       # Attention Sink 子模块
│   │   │   ├── attn_curve.py         # 注意力曲线提取
│   │   │   ├── pg19.py               # PG-19 数据集加载（流式）
│   │   │   └── ppl.py                # 困惑度计算
│   │   ├── llm_prune/            # LLM 剪枝子模块
│   │   │   ├── data.py               # 校准数据加载
│   │   │   ├── prune.py              # 剪枝算法（magnitude / Wanda）
│   │   │   ├── metrics.py            # PPL + 速度指标
│   │   │   ├── activation_stats.py   # 激活统计收集
│   │   │   ├── plot.py               # 绘图工具
│   │   │   └── utils.py              # 通用工具
│   │   └── llmwanda/             # 模型下载工具
│   │       ├── download_model.py     # HuggingFace 模型下载
│   │       └── README.md
│   ├── results_llm/              # Wanda 剪枝结果
│   ├── results_attention/        # Attention Sink 分析结果
│   ├── results_smoothquant/      # SmoothQuant 结果
│   └── results_awq/              # AWQ 结果（含量化模型权重）
│
├── tools/                    ← 独立辅助工具
│   ├── convert_csv_to_excel.py       # CSV → Excel 批量转换
│   ├── rewrite_llm_snapshot_json.py  # 实验快照 JSON 重写
│   └── show_calibration_examples.py  # 查看校准样本
│
├── data/                     ← 数据集（.gitignore 排除）
│   └── cifar-10-batches-py/      # CIFAR-10 解压数据
└── venv/                     ← Python 虚拟环境（.gitignore 排除）
```

---

## 实验一览

### Track A: CNN 效率与剪枝 / 量化 (CIFAR-10 / ResNet-18)

| 实验 | 脚本 | 说明 |
|------|------|------|
| 环境检查 | `src/hello_torch.py` | PyTorch 版本、CUDA、GPU 信息 |
| CNN 效率对比 | `src/MemoryMeaseurement.py` | ResNet-18 vs MobileNetV3 峰值内存/计算量/延迟 |
| 基线训练 | `src/run_baseline.py` | CIFAR-10 ResNet-18 训练至 90% acc |
| 权重剪枝 | `src/run_weight_pruning.py` | Fine-grained / Channel (Slim) / N:M 结构化剪枝 |
| 激活值剪枝 | `src/run_activation_pruning.py` | 基于激活统计的通道剪枝 + fine-tune |
| K-means/Linear 量化 | `src/run_quantization.py` | CNN 权重 PTQ：K-means 聚类 vs 线性量化 (4/8-bit) |
| 汇总绘图 | `src/plot_and_excel.py` | 合并所有 CSV → Excel + 对比图表 |

### Track B: LLM 压缩 (TinyLlama-1.1B / WikiText-2)

| 实验 | 脚本 | 说明 |
|------|------|------|
| Wanda 剪枝 | `LLM_pruning/src/run_llm_wanda.py` | Baseline + Magnitude + Wanda 非结构化剪枝 |
| Attention Sink | `LLM_pruning/src/run_attention_sink.py` | PG-19 长文本注意力模式分析 + sink token 干预 |
| SmoothQuant W8A8 | `LLM_pruning/src/run_smoothquant.py` | 逐通道平滑 + INT8 权重/激活量化 (α sweep) |
| AWQ W4A16 | `LLM_pruning/src/run_awq.py` | 激活感知 4-bit 权重量化 (group size sweep) |

---

## 环境准备

```bash
python3 -m venv venv
source venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

硬件要求：NVIDIA GPU (>=8 GB VRAM)，已在 RTX 4060 Laptop 上验证。

---

## 运行方法

### 0) 环境验证

```bash
python src/hello_torch.py
```

### 1) CNN 效率对比

```bash
python src/MemoryMeaseurement.py --device cpu \
  --models resnet18,mobilenet_v3_small \
  --batch-sizes 1-16 --resolutions 224 \
  --out-prefix CNN_pruning/results_root/cnn_compare
```

### 2) CNN 剪枝实验

```bash
cd src

# 基线训练
python run_baseline.py --device auto --out-prefix hw90 \
  --epochs 0 --resolution 32

# 权重剪枝
python run_weight_pruning.py --device auto \
  --ckpt hw90_baseline_best.pt --out-prefix hw90_wt \
  --ratios 0.3,0.5,0.8 --nm-patterns 1:4,2:4,3:4 \
  --channel-impl slim --resolution 32

# 激活值剪枝
python run_activation_pruning.py --device auto \
  --ckpt hw90_baseline_best.pt --out-prefix hw90_act_ft \
  --impl slim --ratios 0.3,0.5,0.8 --resolution 32

# 汇总
python plot_and_excel.py --out-prefix hw90_combo \
  --baseline-prefix hw90 --weight-prefix hw90_wt \
  --activation-prefix hw90_act_ft
```

### 3) CNN K-means vs Linear 量化

```bash
python src/run_quantization.py \
  --ckpt CNN_pruning/results_src/hw90_baseline_best.pt \
  --out-dir CNN_pruning/results_quantization
```

### 4) LLM Wanda 剪枝

```bash
python LLM_pruning/src/run_llm_wanda.py \
  --config LLM_pruning/configs/llm_wanda.yaml
```

### 5) Attention Sink 分析

```bash
python LLM_pruning/src/run_attention_sink.py \
  --config LLM_pruning/configs/attention_sink.yaml
```

### 6) SmoothQuant W8A8

```bash
python LLM_pruning/src/run_smoothquant.py \
  --config LLM_pruning/configs/smoothquant.yaml
```

### 7) AWQ W4A16

```bash
python LLM_pruning/src/run_awq.py \
  --config LLM_pruning/configs/awq.yaml
```

---

## 缓存与大文件说明

以下文件/目录被 `.gitignore` 排除，不会上传到 Git：

| 类型 | 路径 | 大小 | 说明 |
|------|------|------|------|
| 虚拟环境 | `venv/` | ~7 GB | Python 依赖，用 `pip install -r requirements.txt` 重建 |
| CIFAR-10 数据 | `data/`, `src/data/` | ~341 MB × 2 | 运行脚本时自动下载 |
| HF 缓存 | `~/.cache/huggingface/hub/` | ~16 GB | TinyLlama + Mistral + datasets，系统级缓存 |
| CNN checkpoint | `*.pt` | ~43 MB each | `run_baseline.py` 生成，可重新训练 |
| AWQ 模型权重 | `results_awq/awq_g*/model.safetensors` | ~2.1 GB × 3 | `run_awq.py` 生成，可重新量化 |
| Excel 文件 | `*.xlsx` | 各数 KB | 从 CSV 自动生成 |

CSV、PNG、YAML 配置、analysis.txt 等 **实验结果会保留在 Git 中**，方便复现和审查。

---

## 源码统计

| 模块 | 文件数 | 总行数 | 说明 |
|------|:------:|:------:|------|
| `src/` (CNN) | 9 | 4,377 | CNN 效率/剪枝/量化 |
| `LLM_pruning/src/` (主脚本) | 5 | 2,440 | LLM 剪枝 + 量化 |
| `LLM_pruning/src/` (子模块) | 10 | 834 | attention_sink / llm_prune / llmwanda |
| `tools/` | 3 | 285 | 辅助工具 |
| **合计** | **27** | **~8,250** | |
