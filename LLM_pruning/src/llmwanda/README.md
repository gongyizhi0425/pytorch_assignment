# LLM Wanda Pruning (作业入口)

主可执行脚本在：`src/run_llm_wanda.py`

从仓库根目录运行：

```bash
python src/run_llm_wanda.py --config configs/llm_wanda.yaml
```

该脚本会：
- 加载 HuggingFace CausalLM（默认 gpt2）
- 跑 baseline
- 跑 magnitude pruning（|w|）
- 跑 WANDA pruning（|w| * activation RMS）
- 输出 PPL/速度/稀疏度指标到 results_llm/

如需更换模型：修改 configs/llm_wanda.yaml 里的 model.name。
