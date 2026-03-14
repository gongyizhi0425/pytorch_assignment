# 仓库重构对话记录

**日期**: 2026-03-14  
**Agent**: GitHub Copilot (Claude Opus 4.6)  
**任务**: 初始化工作区指令 + 仓库目录重构

---

## 用户请求

### 第一轮：初始化工作区指令（init.prompt.md）

按照 VS Code Copilot 的 `init.prompt.md` 流程，为仓库生成 `.github/copilot-instructions.md` 文件。

**执行过程**：
1. 搜索仓库中已有的约定文件（README.md、AGENTS.md 等），确认没有现成指令文件
2. 用只读子代理分别扫描 CNN 和 LLM 代码，提取关键约定和坑点
3. 读取 `copilot-skill:/agent-customization/references/workspace-instructions.md` 获取模板
4. 根据模板和扫描结果生成 `.github/copilot-instructions.md`

**生成的指令文件包含四个章节**：
- Architecture：双轨结构、共享模块边界
- Build And Validation：安全验证命令
- Conventions：GPU-only、config-first、CSV 结构稳定等
- Project-Specific Pitfalls：自动下载、数据路径、hook 依赖等

---

### 第二轮：仓库重构 + 知识问答

用户提出两个问题：

#### 问题 1：指令文件存储位置与聊天记录

**回答**：
- `.github/copilot-instructions.md` 是仓库中的普通文件，`git push` 后即上云
- 聊天记录存在本地 VS Code 的 `globalStorage/` 中，不会同步到 GitHub
- 重装 Ubuntu 后聊天记录会丢失（除非手动备份 globalStorage）
- 不能直接把聊天记录上传到 GitHub，只能手动复制保存

#### 问题 2：重构目录结构

用户要求将仓库从"源码/结果分离"改为"按实验类型分类"的四目录结构。

---

## 重构详情

### 旧结构

```
pytorch_demo/
├── src/              ← CNN + basic 脚本混放
├── CNN_pruning/      ← CNN 配置 + 结果（无源码）
├── LLM_pruning/      ← LLM 全套（源码+配置+结果）
├── tools/
└── data/             ← CIFAR-10
```

### 新结构

```
pytorch_demo/
├── CNN/              ← CNN 全套（源码+配置+数据+结果）
│   ├── src/          (12 个 .py + readme.md)
│   ├── configs/
│   ├── data/         (CIFAR-10, gitignore)
│   └── results_*/
├── LLM/              ← LLM 全套
│   ├── src/          (5 个 run_*.py + 3 子模块)
│   ├── configs/
│   └── results_*/
├── tools/            ← 辅助工具（不变）
├── basic/            ← 环境验证 + 通用基准
│   ├── hello_torch.py
│   └── MemoryMeaseurement.py
└── .github/
    └── copilot-instructions.md
```

### 文件分类依据

| 目标目录    | 文件                                                                                                                                                                                                                  | 理由                                       |
| ----------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------ |
| `CNN/src/`  | pruning_common, pruning_experiment, run_baseline, run_weight_pruning, run_activation_pruning, run_quantization, run_search_space_eval, run_ss3_supplement, analyze_search_space, nas_trial, plot_and_excel, readme.md | 全部绑定 CIFAR-10 ResNet-18 或 MobileNetV2 |
| `basic/`    | hello_torch.py                                                                                                                                                                                                        | 纯环境检测，不属于任何实验                 |
| `basic/`    | MemoryMeaseurement.py                                                                                                                                                                                                 | 通用模型基准测试（非 CIFAR-10 专用）       |
| `CNN/data/` | cifar-10-batches-py/, cifar-10-python.tar.gz                                                                                                                                                                          | CIFAR-10 数据归属 CNN                      |

### 执行的操作

1. **`git mv CNN_pruning CNN`** — 重命名，保留 git 历史
2. **`git mv LLM_pruning LLM`** — 同上
3. **`git mv src/*.py CNN/src/`** — 移动 CNN 追踪文件（8个）
4. **`mv src/*.py CNN/src/`** — 移动未追踪文件（4个：analyze_search_space, run_search_space_eval, run_ss3_supplement, nas_trial）
5. **`git mv src/hello_torch.py basic/`** — 环境验证脚本
6. **`git mv src/MemoryMeaseurement.py basic/`** — 基准测试脚本
7. **`mv data/ CNN/data/`** — 数据迁移
8. 清理空的 `src/`、`data/` 残余目录

### 更新的路径引用

| 文件                               | 修改内容                                                                  |
| ---------------------------------- | ------------------------------------------------------------------------- |
| `CNN/src/analyze_search_space.py`  | `../CNN_pruning/results_search_space/` → `../results_search_space/`       |
| `CNN/src/run_ss3_supplement.py`    | `../CNN_pruning/results_search_space/` → `../results_search_space/`       |
| `CNN/src/run_quantization.py`      | docstring 中 `src/run_quantization.py` → `CNN/src/run_quantization.py` 等 |
| `CNN/src/run_search_space_eval.py` | docstring 中 config 路径                                                  |
| `.gitignore`                       | `data/` + `src/data/` → `CNN/data/`                                       |
| `.github/copilot-instructions.md`  | 全部路径对齐新结构                                                        |
| `README.md`                        | 完全重写，匹配新目录树和运行命令                                          |
| `CNN/src/readme.md`                | 从混合 CNN+LLM 内容精简为纯 CNN 速查                                      |

### 验证结果

| 检查项                                                                       | 结果                               |
| ---------------------------------------------------------------------------- | ---------------------------------- |
| `python basic/hello_torch.py`                                                | ✅ PyTorch 2.7.1+cu118, RTX 4060    |
| `import pruning_common` (from CNN/src/)                                      | ✅ OK                               |
| `python CNN/src/run_baseline.py --help`                                      | ✅ 参数列表正常                     |
| `python LLM/src/run_llm_wanda.py --config LLM/configs/llm_wanda.yaml --help` | ✅ 参数列表正常                     |
| `git status`                                                                 | ✅ 全部显示为 R（rename），历史完整 |

### Git Commit

```
refactor: reorganize repo into CNN/, LLM/, tools/, basic/

- Rename CNN_pruning/ -> CNN/, LLM_pruning/ -> LLM/
- Move CNN source files from src/ -> CNN/src/
- Move hello_torch.py, MemoryMeaseurement.py -> basic/
- Move CIFAR-10 data -> CNN/data/
- Update hardcoded paths in analyze_search_space, run_ss3_supplement,
  run_quantization, run_search_space_eval
- Update .gitignore for new data path (CNN/data/)
- Rewrite README.md and CNN/src/readme.md for new structure
- Add .github/copilot-instructions.md
- Add .docs/ai-logs/ conversation exports
```

---

## 后续建议（来自第一轮 init 提示）

1. **`/create-instruction cnn.instructions.md`** — 仅对 `CNN/**` 生效，补充 GPU-only、out_prefix 规则
2. **`/create-instruction llm.instructions.md`** — 仅对 `LLM/**` 生效，强调 YAML-first、HF cache
3. **`/create-prompt experiment-sanity.prompt.md`** — 统一"轻量验证"模板
4. **`/create-hook confirm-heavy-runs`** — 拦截可能下载大模型或启动训练的命令
