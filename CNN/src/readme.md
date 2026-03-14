# CNN 实验运行速查

所有 CNN 命令建议从 `CNN/` 目录执行，数据统一放在 `CNN/data/`。

## 环境验证（从仓库根）

```bash
python basic/hello_torch.py
```

## 基线训练

```bash
cd CNN
python src/run_baseline.py --device cuda --amp --eval-every 1 \
  --early-stop-acc 0.90 --data-dir ./data --out-prefix hw90
```

## 权重剪枝

```bash
python src/run_weight_pruning.py --device auto \
  --ckpt hw90_baseline_best.pt --out-prefix hw90_wt \
  --data-dir ./data --ratios 0.3,0.5,0.8 --nm-patterns 1:4,2:4,3:4 \
  --channel-impl slim --resolution 32
```

## 激活值剪枝

```bash
python src/run_activation_pruning.py --device auto \
  --ckpt hw90_baseline_best.pt --out-prefix hw90_act_ft \
  --data-dir ./data --impl slim --ratios 0.3,0.5,0.8 --resolution 32
```

## 汇总绘图

```bash
python src/plot_and_excel.py --out-prefix hw90_combo \
  --baseline-prefix hw90 --weight-prefix hw90_wt \
  --activation-prefix hw90_act_ft
```

## K-means / Linear 量化

```bash
python src/run_quantization.py \
  --ckpt results_src/hw90_baseline_best.pt \
  --out-dir results_quantization
```

## 数据集

CIFAR-10 自动下载到 `--data-dir` 指定的路径（默认 `./data`）。
从 `CNN/` 运行时默认即为 `CNN/data/`。