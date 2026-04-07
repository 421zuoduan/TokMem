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

## 2. 准备 compositional 数据集

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

## 3. 准备完成后开始训练

`compositional` 主实验直接使用已有脚本：

```bash
cd compositional
bash run_n_rounds_main.sh
```

如果只想自己手动跑入口脚本，也可以从：

```bash
python main_sequential.py
```

开始。
