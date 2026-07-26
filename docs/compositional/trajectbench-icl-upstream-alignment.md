# TRAJECT-Bench ICL 与 TokMem upstream 对齐

> 2026-07-26 更新：TRAJECT-Bench launcher 的默认协议为
> `tokmem_icl`，用于最终汇报并严格复现 TokMem upstream 行为。修正
> argument-key proxy bug 的唯一参数 schema 推理仍由
> `PROTOCOL=compositional_icl` 保留为诊断协议，已有归档与协议名含义不变。

## 对齐目标

TRAJECT-Bench 保留的 `tokmem_icl` 协议以 TokMem 官方仓库 commit
`0cf99764817afd98afc41ca8fed880d508efc647` 为行为规范。它用于复现公开代码，
不修复公开实现的 ICL Tool F1 问题。更强且指标语义正确的显式工具名版本保留为
独立的 `explicit_tool_calls` 协议。

## Upstream 行为

- Prompt 使用硬编码 Llama chat tokens，包含全部可获得的工具文档、arg-only
  输出要求和静态 JSON 格式示例。
- 工具文档使用 `json.dumps(..., indent=2)`；非 ASCII 字符被转义。
- 每个调用输出一个 arguments JSON object，每行一个。
- 推理使用 `max_new_tokens=256`、`temperature=0.6`、`top_p=0.9`、
  `do_sample=False`、`pad_token_id=eos_token_id` 和 `use_cache=True`。
- Parser 逐行尝试 `json.loads`；非法行静默跳过。空响应返回空列表，不记
  parse error。只有完全无法解析的非空响应才保存 raw-text parse-error sentinel。
- `_infer_tool_from_call` 虽然存在，但 prediction/evaluation 路径从不调用。
- Tool F1 直接将 predicted/gold argument JSON 传入 `extract_tool_names`。单键
  arguments 的参数名被当作工具名，多键 arguments 通常提取不到名字。

最后一项是公开 upstream 的指标 bug，但严格复现协议必须保留。因此
`tokmem_icl` 的 Tool F1 是 argument-key proxy，不是真实工具选择 F1。因为响应
不包含可恢复的真实工具序列，Routing Accuracy、Transition Error 和工具感知的
Rouge-L 均报告为 N/A。

## TRAJECT 数据适配

TRAJECT-Bench 的工具目录格式不同于 xLAM，adapter 只负责转换为 upstream 的
`{name, description, parameters, type}` 形状，并按公共 catalog 的首次发现顺序
保留工具。固定 split 有 53 个候选工具，其中 2 个不在公共 catalog；对齐
upstream extractor 的缺文档行为，这两个工具只记录缺失并从 prompt 省略，不再
生成 fallback 描述。

## 独立 Review 与验证

两份独立代码 review 均确认此前实现是“upstream arg-only prompt + corrected
schema metric”，不是严格复现。修正后的回归测试覆盖：

- prompt byte equality 与 Unicode escaping；
- 单行/pretty JSON array；
- 合法与非法混合行、全非法响应和空响应；
- 不运行 schema inference；
- 单键/多键 argument-key proxy Tool F1；
- catalog 顺序、缺文档和工具文档字段。

验证命令：

```bash
source /home/shilong/anaconda3/etc/profile.d/conda.sh
conda activate tokmem
python compositional_trajectbench/test_trajectbench_base.py
```

最初严格 upstream 对齐的 17 项测试通过；加入 `compositional_icl` 后，
当前多协议测试总数为 22 项，全部通过。另用 Llama-1B 完成 2 条真实 GPU
smoke test，确认 run config 记录 upstream commit/metric bug，prediction 不生成
schema-derived tools，Routing/Transition 输出为 N/A。

## 结果口径

- `tokmem_icl`：当前默认和最终汇报口径；忠实 upstream 复现，只将
  Tool/Argument F1 与论文旧 ICL 口径比较。
- `compositional_icl`：诊断协议；使用相同 arg-only 生成，并通过唯一参数
  schema 恢复工具。Tool/Routing/Transition 指标是 schema-inferred 近似值。
- `explicit_tool_calls`：显式输出 `[{tool_name, arguments}, ...]`，使用真实工具
  指标，作为更强的现代 Direct Prompting baseline。
- 2026-07-26 早先标为 `trajectbench_tokmem_aligned_icl_*` 的完整 1B/8B 运行使用
  unique-schema corrected metric，现视为 preliminary corrected runs，不能作为
  upstream-aligned 结果。

## 最终汇报的 TokMem upstream ICL 结果

2026-07-26 使用固定 `7be133de0424` split 完成 1B/8B 各 90 条重跑。两组均使用
batch size 4、`max_new_tokens=256`、bf16、greedy decoding 和 seed 42；prompt
平均 10310.52 tokens，53 个候选工具中 51 个有公共 catalog 文档。为与 TokMem
原仓库 baseline 对齐，rebuttal 最终采用这组严格复现结果：

| 模型 | Tool F1* | Argument F1 | Argument P/R | Parse Error Example Rate |
| --- | ---: | ---: | ---: | ---: |
| Llama-3.2-1B-Instruct | 0.0556 | 0.0000 | 0.0000 / 0.0000 | 0.8444 |
| Llama-3.1-8B-Instruct | 0.5785 | 0.2227 | 0.2292 / 0.2181 | 0.0222 |

\* Tool F1 严格复现 upstream 的 argument-key proxy bug，不是真实工具选择 F1。
两组的 Routing Accuracy、Transition Error 和工具感知 Rouge-L 均为 N/A。
完整 predictions、配置、日志、split 和代码快照归档于：

- `results/compositional/trajectbench_upstream_icl_llama1b_min12_seed42_20260726/`
- `results/compositional/trajectbench_upstream_icl_llama8b_min12_seed42_20260726/`

## Compositional schema-inferred ICL 诊断结果

`compositional_icl` 使用相同 arg-only prompt、parser 和 generation，但在
解析后调用当前 `compositional/icl_baseline.py` 的唯一参数 schema 推理，并以
恢复出的工具序列计算 Tool F1、Routing、Rouge-L 和 Transition Error。2026-07-26
在同一固定 split 上完成 1B/8B 各 90 条全量重跑；这两组结果只作为修正
upstream Tool F1 bug 后的诊断，不进入 rebuttal 最终 ICL 汇总：

| 模型 | Tool F1 | Argument F1 | Routing | Rouge-L | Transition Error | Unresolved calls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Llama-3.2-1B-Instruct | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 85 / 108 |
| Llama-3.1-8B-Instruct | 0.5752 | 0.2227 | 0.0444 | 0.3280 | 0.7897 | 191 / 593 |

两组 raw responses/function calls 与严格 upstream 运行逐条完全一致，因此
Argument F1 完全相同。Tool 指标变化只来自 evaluator：upstream 使用错误的
argument-key proxy，当前 compositional 只在参数 schema 唯一时恢复工具。
8B 的 `0.5752` 与 upstream proxy `0.5785` 数值接近是巧合，不能直接比较。

完整归档：

- `results/compositional/trajectbench_compositional_icl_llama1b_min12_seed42_20260726/`
- `results/compositional/trajectbench_compositional_icl_llama8b_min12_seed42_20260726/`
