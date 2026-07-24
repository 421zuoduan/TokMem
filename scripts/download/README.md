# Download And Prepare

这个目录只放训练前的数据和模型准备脚本。

## 环境

所有命令都在 `tokmem` conda 环境下执行：

```bash
source /data/ruochen/anaconda/etc/profile.d/conda.sh
conda activate tokmem
cd /data/ruochen/tokmem
```

## 1. 下载 1B 模型

```bash
bash scripts/download/download_1b_model.sh
```

输出目录：

```bash
compositional/models/Llama-3.2-1B-Instruct
```

## 2. 下载常用本地模型

使用 hf-mirror 下载 Qwen2.5 0.5B、Llama 3.2 1B、Llama 3.2 3B、Llama 3.1 8B：

```bash
bash scripts/download/download_hf_mirror_models.sh
```

输出目录：

```bash
models/Qwen2.5-0.5B-Instruct
models/Llama-3.2-1B-Instruct
models/Llama-3.2-3B-Instruct
models/Llama-3.1-8B-Instruct
```

只验证下载链路能开始时运行：

```bash
bash scripts/download/download_hf_mirror_models.sh --verify-only
```

脚本默认使用 `Qwen/Qwen2.5-0.5B-Instruct` 和公开可访问的 `unsloth/*Llama*` 仓库，下载后保存成本仓库训练脚本使用的本地目录名。

## 3. 准备 compositional 数据集

```bash
bash scripts/download/prepare_xlam_dataset.sh
```

输出目录：

```bash
compositional/data
```

主要会生成：

```bash
compositional/data/training/function_calling_train_tools1-50_4calls.json
compositional/data/test/function_calling_test_tools1-50_4calls.json
compositional/data/training/function_calling_train_tools51-100_4calls.json
compositional/data/test/function_calling_test_tools51-100_4calls.json
compositional/data/tool_descriptions_tools1-50.json
compositional/data/tool_descriptions_tools51-100.json
```

## 4. 准备完成后开始训练

`compositional` 主实验直接使用已有脚本：

```bash
cd compositional
bash run_n_rounds_main.sh
```

如果只想自己手动跑入口脚本，也可以从：

```bash
python main_sequential.py
```

## 5. 下载 Toolathlon 和 MCP-Atlas

下载 Toolathlon-Verified 的 108 个公开任务、配套输入与评测资源，以及 MCP-Atlas
的 500 条公开任务：

```bash
python scripts/download/download_tool_use_benchmarks.py
```

默认输出：

```text
datasets/toolathlon/tasks/finalpool/
datasets/toolathlon/DOWNLOAD_INFO.json
datasets/mcp-atlas/MCP-Atlas.parquet
datasets/mcp-atlas/DOWNLOAD_INFO.json
```

下载器固定 Toolathlon 和 MCP-Atlas 的官方发布提交，支持断点续传，并分别用
Git blob SHA-1 和官方 SHA-256 校验文件。若只下载其中一个 benchmark：

```bash
python scripts/download/download_tool_use_benchmarks.py --benchmark toolathlon
python scripts/download/download_tool_use_benchmarks.py --benchmark mcp-atlas
```

如果当前机器无法连接 `raw.githubusercontent.com`，可以指定 GitHub 文件镜像；
下载器仍会用官方文件树中的 Git blob SHA-1 校验每个文件：

```bash
python scripts/download/download_tool_use_benchmarks.py \
  --benchmark toolathlon \
  --github-raw-prefix https://ghfast.top/https://raw.githubusercontent.com
```

开始。
