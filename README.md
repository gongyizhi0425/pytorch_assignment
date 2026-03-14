# PyTorch Experiments — CNN Pruning / Quantization & LLM Compression

> 基于 PyTorch 的深度学习模型压缩实验合集，涵盖 CNN 剪枝 / 量化和 LLM 压缩（剪枝、注意力分析、量化）。

## 目录结构总览

```
pytorch_demo/
├── README.md                 ← 本文件
├── requirements.txt          ← Python 依赖
├── .github/
│   └── copilot-instructions.md  ← AI 助手工作区指令
│
├── CNN/                      ← CNN 实验（CIFAR-10 / ResNet-18）
│   ├── src/                      # 全部 CNN 源码
│   │   ├── pruning_common.py         # 剪枝公共工具（数据加载、指标、模型）
│   │   ├── pruning_experiment.py     # 旧版剪枝实验框架（备用）
│   │   ├── run_baseline.py           # 基线训练
│   │   ├── run_weight_pruning.py     # 权重剪枝入口
│   │   ├── run_activation_pruning.py # 激活值剪枝入口
│   │   ├── run_quantization.py       # K-means vs Linear 量化（PTQ）
│   │   ├── run_search_space_eval.py  # 搜索空间质量评估
│   │   ├── run_ss3_supplement.py     # SS3 补充实验
│   │   ├── analyze_search_space.py   # 搜索空间后处理分析
│   │   ├── nas_trial.py             # MobileNetV2 NAS 试验
│   │   ├── plot_and_excel.py         # 汇总 CSV → Excel + 对比图表
│   │   └── readme.md                # CNN 运行命令速查
│   ├── configs/                  # YAML 配置
│   │   ├── test_config.yaml
│   │   ├── quantization_config.yaml
│   │   └── search_space_config.yaml
│   ├── data/                     # CIFAR-10 数据集（.gitignore 排除）
│   ├── results_root/             # 调试 / sanity check 结果
│   ├── results_src/              # 正式剪枝实验结果（hw90 系列）
│   ├── results_quantization/     # K-means vs Linear 量化结果
│   └── results_search_space/     # 搜索空间评估结果
│
├── LLM/                      ← LLM 实验（TinyLlama-1.1B）
│   ├── src/                      # LLM 源码
│   │   ├── run_llm_wanda.py          # Wanda / Magnitude 剪枝
│   │   ├── run_attention_sink.py     # Attention Sink 分析（PG-19）
│   │   ├── run_smoothquant.py        # SmoothQuant W8A8
│   │   ├── run_awq.py               # AWQ W4A16
│   │   ├── rerun_error_analysis.py   # AWQ 误差分析重跑
│   │   ├── llm_prune/               # 剪枝核心子模块
│   │   ├── attention_sink/           # 注意力分析子模块
│   │   └── llmwanda/                # 模型下载工具
│   ├── configs/                  # YAML 配置
│   │   ├── llm_wanda.yaml
│   │   ├── attention_sink.yaml
│   │   ├── smoothquant.yaml
│   │   └── awq.yaml
│   ├── results_llm/              # Wanda 剪枝结果
│   ├── results_attention/        # Attention Sink 分析结果
│   ├── results_smoothquant/      # SmoothQuant 结果
│   └── results_awq/              # AWQ 结果
│
├── tools/                    ← 独立辅助工具
│   ├── convert_csv_to_excel.py
│   ├── rewrite_llm_snapshot_json.py
│   └── show_calibration_examples.py
│
├── basic/                    ← 环境验证与通用基准测试
│   ├── hello_torch.py            # PyTorch 版本、CUDA、GPU 信息
│   └── MemoryMeaseurement.py     # ResNet-18 vs MobileNetV3 效率对比
│
└── venv/                     ← Python 虚拟环境（.gitignore 排除）
```

---

## 实验一览

### Track A: CNN 剪枝 / 量化 (CIFAR-10 / ResNet-18)

| 实验 | 脚本 | 说明 |
|------|------|------|
| 基线训练 | `CNN/src/run_baseline.py` | CIFAR-10 ResNet-18 训练至 90% acc |
| 权重剪枝 | `CNN/src/run_weight_pruning.py` | Fine-grained / Channel (Slim) / N:M 结构化剪枝 |
| 激活值剪枝 | `CNN/src/run_activation_pruning.py` | 基于激活统计的通道剪枝 + fine-tune |
| K-means/Linear 量化 | `CNN/src/run_quantization.py` | CNN 权重 PTQ：K-means 聚类 vs 线性量化 (4/8-bit) |
| 搜索空间评估 | `CNN/src/run_search_space_eval.py` | MobileNetV2 架构搜索空间质量分析 |
| 汇总绘图 | `CNN/src/plot_and_excel.py` | 合并所有 CSV → Excel + 对比图表 |

### Track B: LLM 压缩 (TinyLlama-1.1B / WikiText-2)

| 实验 | 脚本 | 说明 |
|------|------|------|
| Wanda 剪枝 | `LLM/src/run_llm_wanda.py` | Baseline + Magnitude + Wanda 非结构化剪枝 |
| Attention Sink | `LLM/src/run_attention_sink.py` | PG-19 长文本注意力模式分析 + sink token 干预 |
| SmoothQuant W8A8 | `LLM/src/run_smoothquant.py` | 逐通道平滑 + INT8 权重/激活量化 (α sweep) |
| AWQ W4A16 | `LLM/src/run_awq.py` | 激活感知 4-bit 权重量化 (group size sweep) |

### 辅助工具

| 脚本 | 说明 |
|------|------|
| `basic/hello_torch.py` | 环境检查：PyTorch 版本、CUDA、GPU 信息 |
| `basic/MemoryMeaseurement.py` | CNN 效率对比：峰值内存 / 计算量 / 延迟 |
| `tools/convert_csv_to_excel.py` | CSV → Excel 批量转换 |

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
python basic/hello_torch.py
```

### 1) CNN 效率对比

```bash
python basic/MemoryMeaseurement.py --device cpu \
  --models resnet18,mobilenet_v3_small \
  --batch-sizes 1-16 --resolutions 224 \
  --out-prefix CNN/results_root/cnn_compare
```

### 2) CNN 剪枝实验

```bash
cd CNN

# 基线训练
python src/run_baseline.py --device auto --out-prefix hw90 \
  --data-dir ./data --epochs 0 --resolution 32

# 权重剪枝
python src/run_weight_pruning.py --device auto \
  --ckpt hw90_baseline_best.pt --out-prefix hw90_wt \
  --data-dir ./data --ratios 0.3,0.5,0.8 --nm-patterns 1:4,2:4,3:4 \
  --channel-impl slim --resolution 32

# 激活值剪枝
python src/run_activation_pruning.py --device auto \
  --ckpt hw90_baseline_best.pt --out-prefix hw90_act_ft \
  --data-dir ./data --impl slim --ratios 0.3,0.5,0.8 --resolution 32

# 汇总
python src/plot_and_excel.py --out-prefix hw90_combo \
  --baseline-prefix hw90 --weight-prefix hw90_wt \
  --activation-prefix hw90_act_ft
```

### 3) CNN K-means vs Linear 量化

```bash
python CNN/src/run_quantization.py \
  --ckpt CNN/results_src/hw90_baseline_best.pt \
  --out-dir CNN/results_quantization
```

### 4) LLM Wanda 剪枝

```bash
python LLM/src/run_llm_wanda.py \
  --config LLM/configs/llm_wanda.yaml
```

### 5) Attention Sink 分析

```bash
python LLM/src/run_attention_sink.py \
  --config LLM/configs/attention_sink.yaml
```

### 6) SmoothQuant W8A8

```bash
python LLM/src/run_smoothquant.py \
  --config LLM/configs/smoothquant.yaml
```

### 7) AWQ W4A16

```bash
python LLM/src/run_awq.py \
  --config LLM/configs/awq.yaml
```

---

## 缓存与大文件说明

以下文件/目录被 `.gitignore` 排除，不会上传到 Git：

| 类型 | 路径 | 说明 |
|------|------|------|
| 虚拟环境 | `venv/` | 用 `pip install -r requirements.txt` 重建 |
| CIFAR-10 数据 | `CNN/data/` | 运行脚本时自动下载 |
| HF 缓存 | `~/.cache/huggingface/hub/` | TinyLlama + datasets，系统级缓存 |
| CNN checkpoint | `*.pt` | `run_baseline.py` 生成，可重新训练 |
| AWQ 模型权重 | `LLM/results_awq/awq_g*/model.safetensors` | `run_awq.py` 生成 |
| Excel 文件 | `*.xlsx` | 从 CSV 自动生成 |

CSV、PNG、YAML 配置、analysis.txt 等**实验结果会保留在 Git 中**，方便复现和审查。
