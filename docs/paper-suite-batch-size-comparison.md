# Paper Suite Batch Size 对比表

日期：2026-04-25

来源脚本：

- `scripts/atomic/run_paper_atomic_suite.sh`
- `scripts/compositional/run_paper_compositional_suite.sh`

本文只汇总 suite 脚本当前实际传给训练或评测入口的 batch size。`train bs` 对应训练参数，`eval bs` 对应验证或测试 dataloader 参数，`test bs` 对应纯评测入口参数。

## Atomic Suite

固定设置：

- 数据：`700 tasks / train 500 / val 10 / test 50 / seed 42`
- 模型：`qwen0_5b`、`llama3b`、`llama8b`
- 方法：`base`、`rag`、`lora`、`tokmem`、`tokmem_logit_bias`
- 试验次数：每个 model/method `3` 次
- 训练类方法共享：`epochs=1`、`lr=5e-3`
- `rag` 共享：`retrieval_top_k=3`

| 方法 | CLI 参数含义 | `qwen0_5b` | `llama3b` | `llama8b` |
| --- | --- | ---: | ---: | ---: |
| `base` | `test_batch_size` | `512` | `256` | `128` |
| `rag` | `test_batch_size` | `256` | `128` | `64` |
| `lora` | `batch_size` | `8` | `2` | `2` |
| `lora` | `gradient_accumulation_steps` | `1` | `2` | `2` |
| `lora` | `val_batch_size` / `test_batch_size` | `32` | `16` | `8` |
| `tokmem` | `batch_size` | `16` | `8` | `4` |
| `tokmem` | `gradient_accumulation_steps` | `1` | `1` | `2` |
| `tokmem` | `val_batch_size` / `test_batch_size` | `64` | `32` | `16` |
| `tokmem` | effective train batch | `16` | `8` | `8` |
| `tokmem_logit_bias` | `batch_size` | `16` | `8` | `4` |
| `tokmem_logit_bias` | `gradient_accumulation_steps` | `1` | `1` | `2` |
| `tokmem_logit_bias` | `val_batch_size` / `test_batch_size` | `64` | `32` | `16` |
| `tokmem_logit_bias` | effective train batch | `16` | `8` | `8` |

按实验方法展开：

| 方法 | 模型 | train bs | eval bs | test bs | 备注 |
| --- | --- | ---: | ---: | ---: | --- |
| `base` | `qwen0_5b` | - | - | `512` | 纯测试 |
| `base` | `llama3b` | - | - | `256` | 纯测试 |
| `base` | `llama8b` | - | - | `128` | 纯测试 |
| `rag` | `qwen0_5b` | - | - | `256` | `retrieval_top_k=3` |
| `rag` | `llama3b` | - | - | `128` | `retrieval_top_k=3` |
| `rag` | `llama8b` | - | - | `64` | `retrieval_top_k=3` |
| `lora` | `qwen0_5b` | `8` | `32` | `32` | `gradient_accumulation_steps=1` |
| `lora` | `llama3b` | `2` | `16` | `16` | `gradient_accumulation_steps=2` |
| `lora` | `llama8b` | `2` | `8` | `8` | `gradient_accumulation_steps=2` |
| `tokmem` | `qwen0_5b` | `16` | `64` | `64` | TokMem family, `gradient_accumulation_steps=1`, effective train bs `16` |
| `tokmem` | `llama3b` | `8` | `32` | `32` | TokMem family, `gradient_accumulation_steps=1`, effective train bs `8` |
| `tokmem` | `llama8b` | `4` | `16` | `16` | TokMem family, `gradient_accumulation_steps=2`, effective train bs `8` |
| `tokmem_logit_bias` | `qwen0_5b` | `16` | `64` | `64` | `use_logit_bias`, `gradient_accumulation_steps=1`, effective train bs `16` |
| `tokmem_logit_bias` | `llama3b` | `8` | `32` | `32` | `use_logit_bias`, `gradient_accumulation_steps=1`, effective train bs `8` |
| `tokmem_logit_bias` | `llama8b` | `4` | `16` | `16` | `use_logit_bias`, `gradient_accumulation_steps=2`, effective train bs `8` |

## Compositional Suite

固定设置：

- 数据：`tools 51-100 / 4 calls`、`tools 51-100 / 10 calls`
- 模型：`llama1b`、`llama3b`、`llama8b`
- 方法：`icl`、`rag`、`lora`、`tokmem`、`tokmem_eoc`、`tokmem_eoc_logit_bias`、`tokmem_eoc_replace_head`、`adap_tokmem`、`adap_tokmem_eoc`、`adap_tokmem_eoc_logit_bias`、`adap_tokmem_eoc_replace_head`
- 试验次数：每个 call-scope/model/method `5` 次
- TokMem 训练轮次：`51-100:1`
- LoRA 训练轮次：`51-100:3`
- Adap 训练轮次：`1-50:1,51-100:3`
- TokMem/Adap 共享：`epochs=3`、`lr=5e-3`
- LoRA 共享：`lr=5e-5`
- `rag` 共享：`retrieval_k=5`
- `max_length`：`4calls=512`，`10calls=1024`

`4calls` 覆盖全部 compositional 方法；`10calls` stress-test 覆盖 `tokmem`、`tokmem_eoc_logit_bias`、`tokmem_eoc_replace_head`、`adap_tokmem`、`adap_tokmem_eoc_logit_bias`、`adap_tokmem_eoc_replace_head`。

| 方法族 | CLI 参数含义 | `llama1b` | `llama3b` | `llama8b` |
| --- | --- | ---: | ---: | ---: |
| `icl` | `batch_size` | `64` | `32` | `24` |
| `rag` | `batch_size` | `256` | `192` | `128` |
| `lora` | `batch_size` | `16` | `8` | `4` |
| `lora` | `eval_batch_size` | `128` | `96` | `32` |
| TokMem family | `batch_size` | `24` | `16` | `8` |
| TokMem family | `eval_batch_size` | `256` | `192` | `64` |
| Adap TokMem family | `batch_size_per_round` | `16,24` | `8,16` | `4,8` |
| Adap TokMem family | `eval_batch_size` | `256` | `192` | `64` |

10-call stress-test 的 batch 设置：

| 方法族 | CLI 参数含义 | `llama1b` | `llama3b` | `llama8b` |
| --- | --- | ---: | ---: | ---: |
| TokMem family 10calls | `batch_size` | `4` | `2` | `1` |
| TokMem family 10calls | `eval_batch_size` | `16` | `8` | `4` |
| Adap TokMem family 10calls | `batch_size_per_round` | `16,4` | `8,2` | `4,1` |
| Adap TokMem family 10calls | `eval_batch_size` | `16` | `8` | `4` |

按实验方法展开：

| 方法 | 模型 | train bs | train bs per round | eval/test bs | 备注 |
| --- | --- | ---: | --- | ---: | --- |
| `icl` | `llama1b` | - | - | `64` | 纯测试 |
| `icl` | `llama3b` | - | - | `32` | 纯测试 |
| `icl` | `llama8b` | - | - | `24` | 纯测试 |
| `rag` | `llama1b` | - | - | `256` | `retrieval_k=5` |
| `rag` | `llama3b` | - | - | `192` | `retrieval_k=5` |
| `rag` | `llama8b` | - | - | `128` | `retrieval_k=5` |
| `lora` | `llama1b` | `16` | - | `128` | `training_rounds=51-100:3` |
| `lora` | `llama3b` | `8` | - | `96` | `training_rounds=51-100:3` |
| `lora` | `llama8b` | `4` | - | `32` | `training_rounds=51-100:3` |
| `tokmem` | `llama1b` | `24` | - | `256` | `training_rounds=51-100:1` |
| `tokmem` | `llama3b` | `16` | - | `192` | `training_rounds=51-100:1` |
| `tokmem` | `llama8b` | `8` | - | `64` | `training_rounds=51-100:1` |
| `tokmem_eoc` | `llama1b` | `24` | - | `256` | `use_eoc` |
| `tokmem_eoc` | `llama3b` | `16` | - | `192` | `use_eoc` |
| `tokmem_eoc` | `llama8b` | `8` | - | `64` | `use_eoc` |
| `tokmem_eoc_logit_bias` | `llama1b` | `24` | - | `256` | `use_eoc`、`use_logit_bias` |
| `tokmem_eoc_logit_bias` | `llama3b` | `16` | - | `192` | `use_eoc`、`use_logit_bias` |
| `tokmem_eoc_logit_bias` | `llama8b` | `8` | - | `64` | `use_eoc`、`use_logit_bias` |
| `tokmem_eoc_replace_head` | `llama1b` | `24` | - | `256` | `use_eoc`、`use_tool_head_replacement` |
| `tokmem_eoc_replace_head` | `llama3b` | `16` | - | `192` | `use_eoc`、`use_tool_head_replacement` |
| `tokmem_eoc_replace_head` | `llama8b` | `8` | - | `64` | `use_eoc`、`use_tool_head_replacement` |
| `adap_tokmem` | `llama1b` | - | `16,24` | `256` | `1-50:1,51-100:3` |
| `adap_tokmem` | `llama3b` | - | `8,16` | `192` | `1-50:1,51-100:3` |
| `adap_tokmem` | `llama8b` | - | `4,8` | `64` | `1-50:1,51-100:3` |
| `adap_tokmem_eoc` | `llama1b` | - | `16,24` | `256` | `use_eoc` |
| `adap_tokmem_eoc` | `llama3b` | - | `8,16` | `192` | `use_eoc` |
| `adap_tokmem_eoc` | `llama8b` | - | `4,8` | `64` | `use_eoc` |
| `adap_tokmem_eoc_logit_bias` | `llama1b` | - | `16,24` | `256` | `use_eoc`、`use_logit_bias` |
| `adap_tokmem_eoc_logit_bias` | `llama3b` | - | `8,16` | `192` | `use_eoc`、`use_logit_bias` |
| `adap_tokmem_eoc_logit_bias` | `llama8b` | - | `4,8` | `64` | `use_eoc`、`use_logit_bias` |
| `adap_tokmem_eoc_replace_head` | `llama1b` | - | `16,24` | `256` | `use_eoc`、`use_tool_head_replacement` |
| `adap_tokmem_eoc_replace_head` | `llama3b` | - | `8,16` | `192` | `use_eoc`、`use_tool_head_replacement` |
| `adap_tokmem_eoc_replace_head` | `llama8b` | - | `4,8` | `64` | `use_eoc`、`use_tool_head_replacement` |

Adap 的 `batch_size_per_round` 顺序对应 `training_rounds=1-50:1,51-100:3`，因此 `16,24` 表示 `1-50` 使用 `16`，`51-100` 使用 `24`。

10-call stress-test 按实验方法展开：

| 方法 | 模型 | train bs | train bs per round | eval/test bs | 备注 |
| --- | --- | ---: | --- | ---: | --- |
| `tokmem` | `llama1b` | `4` | - | `16` | `tools 51-100 / 10 calls` |
| `tokmem` | `llama3b` | `2` | - | `8` | `tools 51-100 / 10 calls` |
| `tokmem` | `llama8b` | `1` | - | `4` | `tools 51-100 / 10 calls` |
| `tokmem_eoc_logit_bias` | `llama1b` | `4` | - | `16` | `use_eoc`、`use_logit_bias`、`10 calls` |
| `tokmem_eoc_logit_bias` | `llama3b` | `2` | - | `8` | `use_eoc`、`use_logit_bias`、`10 calls` |
| `tokmem_eoc_logit_bias` | `llama8b` | `1` | - | `4` | `use_eoc`、`use_logit_bias`、`10 calls` |
| `tokmem_eoc_replace_head` | `llama1b` | `4` | - | `16` | `use_eoc`、`use_tool_head_replacement`、`10 calls` |
| `tokmem_eoc_replace_head` | `llama3b` | `2` | - | `8` | `use_eoc`、`use_tool_head_replacement`、`10 calls` |
| `tokmem_eoc_replace_head` | `llama8b` | `1` | - | `4` | `use_eoc`、`use_tool_head_replacement`、`10 calls` |
| `adap_tokmem` | `llama1b` | - | `16,4` | `16` | `1-50:1,51-100:3`，calls per round=`4,10` |
| `adap_tokmem` | `llama3b` | - | `8,2` | `8` | `1-50:1,51-100:3`，calls per round=`4,10` |
| `adap_tokmem` | `llama8b` | - | `4,1` | `4` | `1-50:1,51-100:3`，calls per round=`4,10` |
| `adap_tokmem_eoc_logit_bias` | `llama1b` | - | `16,4` | `16` | `use_eoc`、`use_logit_bias`、calls per round=`4,10` |
| `adap_tokmem_eoc_logit_bias` | `llama3b` | - | `8,2` | `8` | `use_eoc`、`use_logit_bias`、calls per round=`4,10` |
| `adap_tokmem_eoc_logit_bias` | `llama8b` | - | `4,1` | `4` | `use_eoc`、`use_logit_bias`、calls per round=`4,10` |
| `adap_tokmem_eoc_replace_head` | `llama1b` | - | `16,4` | `16` | `use_eoc`、`use_tool_head_replacement`、calls per round=`4,10` |
| `adap_tokmem_eoc_replace_head` | `llama3b` | - | `8,2` | `8` | `use_eoc`、`use_tool_head_replacement`、calls per round=`4,10` |
| `adap_tokmem_eoc_replace_head` | `llama8b` | - | `4,1` | `4` | `use_eoc`、`use_tool_head_replacement`、calls per round=`4,10` |
