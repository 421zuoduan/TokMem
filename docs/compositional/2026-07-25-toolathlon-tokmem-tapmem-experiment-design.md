# TokMem 与 TapMem 在 Toolathlon-Verified 上的实验设计

> 日期：2026-07-25
>
> 状态：实验协议，尚未产生新实验结果
>
> 评测范围：预先固定的 `local-office-10` 自定义子集
>
> 相关背景：[Toolathlon / MCP-Atlas 验证计划](./2026-07-25-toolathlon-mcp-atlas-tapmem-validation-plan.md)

## 1. 实验结论和固定决策

本实验采用“见过工具、没有见过测试任务”的设置：

1. 训练阶段只使用根据工具文档自行生成的任务、文件和轨迹。
2. 所有训练轨迹都必须在本地真实工具环境中执行，并通过合成任务自己的 evaluator。
3. Toolathlon 的正式题目、初始工作区、ground truth、官方轨迹和 evaluator 不进入训练。
4. 一条成功轨迹拆成多个带历史 observation 的下一步决策样本。
5. TokMem、EOC-only 和 TapMem 使用完全相同的 episode、split、上下文和候选工具。
6. 正式推理时，每轮只生成并执行一个工具调用，执行结果回填后再生成下一步。
7. Toolathlon 的主指标是官方 evaluator 对最终环境状态给出的 Pass，不是轨迹匹配。
8. Tool F1 和 Arguments F1 只作为合成 held-out 数据上的诊断指标。
9. 正式结果必须称为 `Toolathlon-Verified local-office-10` 子集结果，不能称为完整 Toolathlon 成绩。
10. 学生模型不接收自然语言工具文档，因此这是 `no-tool-doc memory-token interface` 的自定义接入；即使使用原始任务和 evaluator，也不能直接与官方 leaderboard 数字等价比较。

整体流程如下：

```text
固定工具版本和 memory-token 映射
→ 生成独立任务、工作区、oracle 和 evaluator
→ teacher agent 在本地真实执行
→ evaluator 验证并筛选规范成功轨迹
→ 按决策点拆成 observation-conditioned 样本
→ 训练 TokMem、EOC-only、TapMem
→ 冻结模型
→ 在每题全新 Toolathlon 容器中闭环执行
→ 官方 evaluator 检查最终状态
```

## 2. 研究问题

主研究问题是：

> 在工具接口已知、测试任务和测试状态未见的条件下，TapMem 是否比 TokMem 更能根据用户问题和真实工具 observation 选择下一工具、生成有效参数，并最终完成长流程任务？

对应假设为：

- **H1：最终成功率。** TapMem 在 `local-office-10` 上的最终 Pass 高于 TokMem。
- **H2：工具路由。** TapMem 在合成 held-out 决策点上的 next-tool 指标高于 TokMem。
- **H3：长流程稳定性。** 随调用步数增加，TapMem 的错误累积速度低于 TokMem。
- **H4：干扰工具鲁棒性。** 增加真实可执行但任务不需要的工具后，TapMem 的性能下降小于 TokMem。

不能仅凭 Tool F1 上升就接受 H1。只有最终环境状态成功率提高，才能声称完整任务能力提高。

## 3. 方法和对照组定义

本实验保持仓库中的术语：

| 方法 | Adaptation | EOC | TCRA / logit bias |
|---|---:|---:|---:|
| TokMem | 否 | 否 | 否 |
| EOC-only | 否 | 是 | 否 |
| TapMem | 否 | 是 | 是 |

其中：

- TokMem 是论文中的 plain method，不能为了方便解析而偷偷加入 EOC。
- EOC-only 是消融实验，不应写成 TokMem。
- TapMem 的 TCRA 就是本仓库的 logit-bias 机制。
- 三种方法使用相同冻结 backbone，不增加 LoRA、planner、retriever 或测试时 adaptation。
- 正式推理必须关闭 ground-truth tool forcing。

推荐报告三组核心结果：

```text
TokMem
TokMem + EOC（EOC-only）
TapMem（EOC + TCRA）
```

如果资源很紧，最低限度比较 TokMem 与 TapMem；但只有加入 EOC-only，才能判断收益来自边界标记还是 TCRA。

## 4. 冻结的 Toolathlon 测试范围

### 4.1 local-office-10

正式测试任务在训练前固定为：

```text
arrange-workspace
courses-ta-hws
detect-revised-terms
excel-data-transformation
excel-market-research
imagenet
paper-checker
privacy-desensitization
reimbursement-form-filler
university-course-selection
```

这 10 题的 MCP server 并集只有：

```text
filesystem
terminal
excel
pdf-tools
```

这里的四项是 server，不是四个 callable tools；每个 server 实际暴露多少个函数，必须以固定版本环境运行时的 `tools/list` 结果为准。

它们不需要 Canvas、Notion、邮箱、WooCommerce、Snowflake、Kubernetes 或第三方测试账号，适合先验证本地文件、表格和 PDF 场景。

本文中的 `official-menu` 只表示保留原始 task config 所确定的可用工具集合，不表示模型输入接口与官方 agent 完全相同。学生侧移除了工具文本说明和 schema，改为 memory token 与 available-tool mask，因此结果应完整写作：

```text
Toolathlon-Verified local-office-10
(no-tool-doc, memory-token interface)
```

### 4.2 每题环境和评分特点

| 任务 | MCP server | preprocess | 关键隐藏信息 | evaluator 主要检查 |
|---|---|---:|---|---|
| `arrange-workspace` | filesystem、terminal、pdf-tools、excel | 是 | 当前文件名、原位置和分类 | 目录和文件路径集合精确匹配 |
| `courses-ta-hws` | terminal、excel、filesystem | 是 | 大量代码文件和学院/学号表 | 文件数量、命名、学院、学号、语言和内容 hash |
| `detect-revised-terms` | filesystem、pdf-tools | 否 | 判决和新法 PDF 的完整条文 | CSV 行及法条文本的规范化精确匹配 |
| `excel-data-transformation` | excel、filesystem、terminal | 否 | 源表布局、示例格式和数据 | 输出工作簿结构、文本和数值 |
| `excel-market-research` | excel、filesystem、terminal | 否 | Methodology 映射、权重和源数据 | ground-truth 工作簿单元格 |
| `imagenet` | filesystem、pdf-tools | 否 | 论文指标和 `format.tex` | `survey.tex` 规范化全文匹配 |
| `paper-checker` | filesystem、terminal | 是 | tex/bib 中错误位置和正确引用 | 所有 tex/bib 与 GT 的逐行比较 |
| `privacy-desensitization` | filesystem、terminal | 是 | 27 个文档内容及字段上下文 | 文件名、数量和去空白后的内容 |
| `reimbursement-form-filler` | filesystem、excel、pdf-tools、terminal | 是 | 票据类型、月份、金额和模板 | 40 个非空行及对应单元格内容 |
| `university-course-selection` | filesystem、pdf-tools、terminal、excel | 否 | 课表候选、字段和四种组合 | 输出文件数量及课程逐字段内容 |

这 10 题都不能仅根据题面静态生成完整参数轨迹。后续调用参数依赖文件枚举、PDF/Excel 内容以及前面工具返回的 observation。

### 4.3 子集的局限

`local-office-10` 只代表 Toolathlon 的本地 Office/文件类任务，不能代表完整 benchmark 的：

- 远程 SaaS 写入；
- 邮件和账号状态；
- 浏览器交互；
- 数据库；
- Kubernetes；
- 多应用一致性。

因此论文中应把它写成低部署成本的预先固定子集，而不是 Toolathlon 全集。

## 5. 环境架构

### 5.1 共享镜像，逐 episode 隔离

不需要为 10 道题构建 10 个不同镜像。正确结构是：

```text
宿主机：
  tokmem conda + TokMem/TapMem + GPU

同一个 Toolathlon 基础镜像：
  filesystem / terminal / excel / pdf-tools
  preprocess
  MCP gateway
  evaluator

每个 episode：
  从同一镜像启动新容器
  → 复制新的 task bundle 和 initial workspace
  → 运行 preprocess
  → 执行 agent
  → 运行 evaluator
  → 保存产物
  → 销毁容器
```

TokMem/TapMem 通过 Toolathlon decoupled runner 暴露的 MCP gateway 与容器通信。这样可以：

- 避免 Toolathlon Python 3.12 与 `tokmem` 环境 Python 3.10 冲突；
- 防止 `terminal` 或 `python_execute` 污染宿主机；
- 保证 TokMem、TapMem、不同 seed 都从同一初始状态开始；
- 防止一个任务删除或移动的文件影响下一个任务。

### 5.2 当前本地快照状态

当前 `datasets/toolathlon` 固定于：

```text
commit: 2aed2468858f15818acafa178518390cc4b0f5cb
benchmark: Toolathlon-Verified
tasks: 108
```

当前目录主要包含任务数据和部分顶层文件，尚缺完整运行所需的：

```text
scripts/
configs/
utils/
global_preparation/
deployment/
```

正式实现前必须补齐同一 commit 的完整仓库，并拉取或构建匹配的 prepared image。不能混用不同 commit 的题目、preprocess、MCP 配置和 evaluator。

### 5.3 Ground truth 隔离

正式执行时：

- agent 的 filesystem 根只能指向 agent workspace；
- groundtruth workspace 和 evaluator 源码不能挂载到模型可读路径；
- teacher 和被测模型均不能调用 terminal 读取 evaluator；
- evaluator 只在 agent 结束后运行；
- 并行任务必须使用独立 groundtruth 副本，因为部分 evaluator 会创建临时文件。

## 6. 工具清单和 memory-token 映射

### 6.1 从真实环境发现工具

不能仅根据文档手写工具列表。完整 decoupled runner 启动后，直接对 raw SSE
gateway 执行 MCP `tools/list`，保存：

```json
{
  "origin": "gateway_mcp",
  "wire_name": "excel-read_sheet",
  "source_server": null,
  "backend_name": null,
  "description": "...",
  "input_schema": {},
  "server_version": "...",
  "schema_hash": "...",
  "memory_token": "<reserved_special_token_...>"
}
```

该版本 gateway 内部知道 `source_server` 和 `backend_name`，但标准 `tools/list`
只返回分配后的 exposed name，不返回结构化 provenance。该 wire name 可能含连字符
和冲突后缀，不能靠拆字符串恢复来源，也不能添加 host 模型层的 `gw-` 前缀。因此
未扩展 gateway 元数据接口时，稳定 ID 使用：

```text
toolathlon-gateway::opaque_wire_name::schema_hash
```

manifest 必须显式保存 `stable_id -> wire_name`，调用时原样传给 raw
`tools/call`。如果以后增加只读 metadata endpoint，则可改用
`source_server::backend_name::schema_hash`，但仍要单独保存 wire name。工具版本、
wire name 或 schema 变化后必须重新生成 manifest 和训练数据。

### 6.2 memory-token 不能动态换义

同一个 checkpoint 中：

- 一个 memory token 永远对应同一个稳定工具 ID；
- 不同题目不能把同一 token 临时映射成不同工具；
- benchmark namespace 的差异由 adapter 做确定性转换；
- schema 不同的同名工具必须使用不同 token。

TapMem 当前最多使用 248 个 reserved token，其中 EOC 占一个，因此最多有 247 个工具 token。正式生成数据前必须确认：

```text
全部 MCP callable tools
+ 需要由模型选择的 local tools
+ EOC
≤ reserved-token 容量
```

如果超出容量，应在看到正式结果前固定扩容方案，且 TokMem、EOC-only、TapMem 使用相同扩容代码。

### 6.3 local tools 的处理

Toolathlon 题目还会列出：

```text
claim_done
python_execute
handle_overlong_tool_outputs
manage_context
history
```

冻结 revision 的官方 default decoupled runner 并不会完整保留这些 group：

- `claim_done` 由 gateway 无条件注册为 raw `local-claim_done`；
- `manage_context`、`history`、`handle_overlong_tool_outputs` 在 decoupled host
  中被过滤，不进入模型菜单；
- `python_execute` 在题目声明需要时仍是 host-local、会改变 workspace 的模型动作；
- Claude SDK decoupled 条件只保留 gateway tools，连 host-local
  `python_execute` 也不提供。

主结果预注册为 **`toolathlon_default_decoupled`**：gateway 全量（含
`claim_done`）加题目实际需要的 `python_execute`。manifest 把 gateway MCP 和
host-local 分层记录；runtime drift check 只检查 gateway 子集。
`python_execute` 必须绑定本次 workspace、限制 120 秒，并在强制 bubblewrap jail
中执行：只读挂载 Python runtime、可写挂载本次 workspace、不挂载源码 checkout、
trusted bundle 或 GT，并关闭网络。jail 失败时禁止退回宿主裸跑。禁用它的
gateway-only 条件只作受控分析，两者不能混在同一列。

full TaskAgent 的 context/history/overlong 工具依赖 runner 内部状态、history writer
和自动 overlong producer；本实验不会只抄 schema 做一个假实现，也不把它们加入
decoupled 主条件。

### 6.4 当前可用工具 mask

学生模型不读取工具说明，但 adapter 知道当前任务允许哪些稳定工具 ID，并在每次工具选择边界：

- 只允许当前 available-tool mask 中的 memory token；
- 允许正常的参数文本 token；
- TapMem 同时允许 EOC；
- 禁止模型调用未提供的工具；
- TokMem、EOC-only、TapMem 使用完全相同的 mask。

该 mask 必须同时作用于：

- backbone 输出的 tool-token logits；
- TCRA/tool-prior head 的 logits 和 softmax 归一化；
- routing 辅助 loss 的候选类别；
- 推理时的最终工具选择。

不能只屏蔽 backbone logits，却让 TCRA 把概率质量分配给当前题目不可用的工具。

## 7. 合成训练数据

### 7.1 数据边界

训练生成器可以读取：

- 固定版本的工具名、说明和 input schema；
- 上游工具公开文档；
- 我们自己生成的虚构文件和实体；
- 自己的 oracle 和 evaluator 模板。

训练生成器不能读取：

- 10 道正式题的 `task.md`；
- 正式 initial workspace；
- 正式 groundtruth workspace；
- 正式 evaluator；
- Toolathlon 官方或历史轨迹；
- 从正式任务运行中得到的 observation。

主实验因此属于：

```text
known-tool / unseen-task / unseen-state
```

训练过程知道四个 server 的全局工具 manifest，但不知道某个正式测试题实际需要哪些工具，也不使用正式轨迹中的“实际调用工具集合”。

官方 public evaluation service 只负责运行已有正式题目，不能作为上传和执行任意合成训练任务的平台。本实验的合成任务和 teacher 轨迹均在自托管的本地工具容器中执行。

### 7.2 不再从原题生成改写题

不采用：

```text
正式题目
→ 改实体、改数字、改文件名
→ 作为训练变体
```

即使文字变化，这仍然泄漏正式任务模板和解法。

替代方案是从四个 server 的全局 manifest 出发，生成独立任务族，并用 coverage matrix 保证训练覆盖。这样不需要知道原题的 gold 轨迹，也能保证模型学过测试范围内的工具接口。

### 7.3 合成任务组成

每个合成任务必须包含：

```text
task_id
template_id
asset_seed
instruction
available_tools
intended_required_tools
distractor_tools
initial_workspace
oracle_final_state
evaluator
generation_provenance
```

其中：

- `intended_required_tools` 只提供给生成器和数据审计，不提供给 solver；
- `available_tools` 至少包含 intended tools 和 3 个真实可执行干扰工具；
- 干扰工具不能只存在于文档中，必须通过当前环境的 `tools/list` 和 smoke call；
- evaluator 优先使用确定性程序，不使用 LLM judge；
- oracle 只用于生成期望最终状态，不暴露给 teacher 或学生模型。

### 7.4 任务族

建议至少覆盖：

| 任务族 | 示例能力 |
|---|---|
| 文件发现与整理 | 递归枚举、按内容/后缀分类、移动、重命名、删除 |
| 多文件筛选 | 根据文件名和表格映射筛选目标文件 |
| Excel reshape | 宽表转长表、列映射、排序、缺失值处理 |
| Excel aggregation | join、分组、增长率、加权和、格式约束 |
| PDF 信息抽取 | 搜索、读页、抽取字段、跨 PDF 对齐 |
| PDF 到表格 | 多票据抽取、筛选、汇总、写模板 |
| LaTeX/BibTeX 修复 | 查找 broken cite/ref、跨文件 label 对齐 |
| 文本脱敏 | 多格式正则与上下文敏感替换 |
| 文档综述 | 从多份 PDF 取指标并写 LaTeX/CSV |
| 约束组合 | 从大型表格或 PDF 筛选候选并输出多方案 |

任务中的文件、论文、人物、课程、票据和数值必须是新生成的虚构数据，不复用 Toolathlon 实体。

### 7.5 数量和覆盖

分三档验收：

| 阶段 | 每个目标工具的成功实际调用 | 用途 |
|---|---:|---|
| smoke | 至少 3 次 | 只检查工具映射和数据管线 |
| pilot | 至少 10 次 | 检查是否有可学习信号 |
| formal | 建议 20–50 次 | 正式比较 |

“出现”必须指工具在接受的轨迹中真实执行成功，不能只是在候选菜单中作为干扰项。

正式数据还应满足：

- 每个目标工具至少覆盖 5 种不同参数形态；
- 每个工具尽量出现在至少 3 个任务模板中；
- 同一模板不能通过简单改数字贡献大部分样本；
- 同一工具既作为目标出现，也在其他任务中作为干扰项出现；
- 2–4、5–8、9–15 步任务都有覆盖；
- 每个 episode 的所有调用具有真实 observation。

长轨迹拆出的 step 更多，不能因此在 loss 中自动获得更大权重。正式 loader 应按 episode 或目标工具重采样，使少量长轨迹不会支配训练；三种方法必须使用相同采样权重。

### 7.6 多 session 生成策略

不能让所有数据来自完全相同的 prompt。至少使用三个独立 generator session：

#### Generator A：environment-first

先生成可验证状态和 oracle，再写题目。

```text
You are generating an original local tool-use training task.
You may use only the supplied tool manifest and a new temporary workspace.
Do not imitate or refer to any benchmark prompt, entity, file, or answer.

First create:
1. a deterministic initial workspace,
2. an oracle final state,
3. a deterministic evaluator.

Then write a natural user instruction whose solution requires the selected
target tools. Add at least three executable distractor tools to the available
menu. Return the task specification as JSON. Do not generate a trajectory.
```

#### Generator B：tool-chain-first

先构造 observation-dependent 工具依赖链，再生成资产。

```text
Design a new multi-step task around the supplied MCP schemas.
At least one later tool argument must depend on an earlier real observation.
Create original files and values so the task can be solved entirely inside
the temporary workspace. Avoid benchmark-like names and scenarios.

Specify the intended tool dependency graph, generate the input assets,
derive the expected final state programmatically, and write a deterministic
checker. Include executable tools that are irrelevant to the solution.
```

#### Generator C：distractor-first

重点生成容易混淆但不会改变正确答案的干扰项。

```text
Create an original task in which several available tools have overlapping
descriptions or plausible names, but only a subset is appropriate.
All offered tools must exist in the provided manifest.
The task must remain objectively checkable from a generated local workspace.
Do not reuse any benchmark task structure. Produce the workspace recipe,
instruction, intended tools, distractors, oracle result, and evaluator.
```

不同 session 应使用不同随机 seed、文件生成器和任务族配额。数据记录中保存：

```text
generator_model
generator_prompt_version
session_id
seed
tool_manifest_hash
```

### 7.7 独立 verifier

任务生成后，由没有参与生成的 verifier 检查：

```text
You are auditing a synthetic tool-use task.

Reject the task if:
- it resembles any protected benchmark prompt or asset,
- the answer is under-specified,
- an offered tool does not exist in the manifest,
- the intended solution needs an unavailable external service,
- the evaluator can pass without completing the user request,
- the evaluator depends on nondeterministic text judgment,
- the oracle reads protected benchmark data,
- the task can only be solved by knowing hidden generator metadata.

Report PASS/FAIL and concrete reasons. Do not repair the task.
```

程序检查至少包括：

- JSON schema；
- 所有文件可生成和解析；
- evaluator 在 oracle final state 上通过；
- evaluator 在未修改 initial state 上失败；
- 至少一个关键错误状态会失败；
- prompt 与正式测试题做 n-gram 和 embedding 去重；
- 没有绝对机器路径、密钥或真实个人信息。

使用正式题面做相似度检查只用于拒绝近重复样本，不能把正式题面内容反馈给 generator 进行改写。

## 8. GT 轨迹生成

### 8.1 两层 GT

一个合成任务需要两种 ground truth：

| GT | 内容 | 用途 |
|---|---|---|
| final-state GT | 期望文件、内容或环境状态 | 合成 evaluator 判断是否完成 |
| canonical trajectory GT | 一条真实执行成功的工具轨迹 | SFT 的下一步动作监督 |

canonical trajectory 不是唯一正确解，只是从多个合法解中选出的规范示范。

### 8.2 teacher agent

每个接受的合成任务在全新容器中运行 teacher agent。teacher 可以看到：

- 用户题目；
- 当前 workspace；
- 完整候选工具文档和 schema；
- 每次真实工具 observation。

teacher 不能看到：

- intended-required-tools 标签；
- oracle；
- evaluator 源码；
- final-state GT；
- 其他 solver 的轨迹。

推荐 teacher prompt：

```text
You are solving a local tool-use task in a fresh sandbox.
Use only the tools currently provided to you.
Do not inspect evaluator or ground-truth paths.
Call one tool at a time and wait for its real observation before deciding
the next action. Verify important writes with read-after-write when useful.
Finish only when the user-visible final state is complete.
Do not replace the whole workflow with a large Python or shell program.
```

teacher 可以使用 GPT-5.6 或其他足够强的模型，但必须固定模型版本、解码设置和 prompt 版本。

### 8.3 多候选轨迹和 canonical 选择

每个任务最多生成若干条独立候选轨迹，全部从干净 initial state 开始。候选必须：

1. 环境初始化成功；
2. 每个调用真实执行；
3. 最终 evaluator Pass；
4. 没有 schema-invalid 调用；
5. 没有读取 GT/evaluator；
6. 没有越过 workspace；
7. 没有过长万能脚本。

如果有多条成功轨迹，按以下顺序选 canonical：

1. 无执行错误；
2. 所有关键写入由 observation 支撑；
3. 没有明显无关调用；
4. 调用数量合理；
5. 参数和路径使用方式具有代表性。

不要直接手工删除轨迹中间步骤，因为后续 observation 可能依赖被删除调用。需要修改时，应从全新状态重新运行 teacher。

### 8.4 错误和恢复数据

最终成功但中途包含错误的轨迹不能直接全量监督，否则会把失败调用当成 GT。

主训练集只使用 clean canonical trajectory。可选 recovery 数据集只构造：

```text
题目 + 已发生错误调用 + 错误 observation
→ 正确恢复动作
```

导致错误的动作本身不计算监督 loss。是否加入 recovery 数据必须作为独立消融，不能只给 TapMem。

### 8.5 原始 episode 格式

建议保存不可逆修改前的 raw episode：

```json
{
  "episode_id": "fs_sort_000123",
  "task_id": "synthetic_fs_sort_000123",
  "template_id": "fs_sort_by_manifest_rule_v2",
  "instruction": "Organize ...",
  "available_tool_ids": [
    "filesystem::list_directory::<hash>",
    "filesystem::move_file::<hash>",
    "excel::read_sheet::<hash>"
  ],
  "workspace_template": "tasks/synthetic_fs_sort_000123/initial_workspace",
  "messages": [
    {
      "role": "assistant",
      "tool_id": "filesystem::list_directory::<hash>",
      "arguments": {"path": "<WORKSPACE>"}
    },
    {
      "role": "tool",
      "tool_id": "filesystem::list_directory::<hash>",
      "observation": "[...]",
      "success": true
    }
  ],
  "final_state_hash": "...",
  "evaluator": {
    "passed": true,
    "version": "..."
  },
  "teacher": {
    "model": "...",
    "prompt_version": "...",
    "run_id": "..."
  },
  "tool_manifest_hash": "..."
}
```

所有绝对 workspace 路径在进入训练集前规范化为 `<WORKSPACE>`，推理时由 adapter 做确定性替换。

## 9. 从 episode 拆成逐步训练样本

### 9.1 定义

一条轨迹写成：

```text
q, a1, o1, a2, o2, ..., aN, oN
```

其中：

- `q` 是用户问题；
- `at` 是第 `t` 次工具调用；
- `ot` 是该调用的真实 observation。

第 `t` 个训练样本为：

```text
input_t  = system + q + (a1, o1) + ... + (a[t-1], o[t-1])
target_t = at
```

模型只能看到过去，不得看到未来 observation、final-state GT 或 evaluator。

### 9.2 示例

原始成功轨迹：

```text
list_directory
→ observation: ["sales.xlsx"]
read_sheet
→ observation: [100, 200, 300]
write_file
→ observation: write success
claim_done
```

转换为：

```text
样本 1：题目
        → list_directory(arguments)

样本 2：题目 + list_directory + 文件列表 observation
        → read_sheet(arguments)

样本 3：题目 + 前两次调用 + Excel observation
        → write_file(arguments)

样本 4：题目 + 完整执行前缀 + write success
        → claim_done(arguments)
```

### 9.3 建议的 step JSON

```json
{
  "sample_id": "fs_sort_000123_step_02",
  "episode_id": "fs_sort_000123",
  "step_index": 2,
  "system_context": {
    "workspace": "<WORKSPACE>"
  },
  "user_input": "Organize ...",
  "available_tool_ids": ["..."],
  "history": [
    {
      "tool_id": "...",
      "arguments": {},
      "observation": "...",
      "success": true
    }
  ],
  "target_tool_id": "...",
  "target_arguments": {},
  "is_terminal": false
}
```

### 9.4 训练 target

TokMem：

```text
<MEM_target_tool> canonical_json(arguments) <native response end>
```

EOC-only 和 TapMem：

```text
<MEM_target_tool> canonical_json(arguments) <EOC> <native response end>
```

JSON 统一使用：

- UTF-8；
- key 排序；
- 固定紧凑 separators；
- 明确区分数值、字符串、布尔值和 null；
- 训练和推理使用相同 canonicalization。

### 9.5 loss mask

以下 token 只作为上下文，label 设为 `-100`：

- system prompt；
- 用户题目；
- 过去的 tool call；
- 过去的 arguments；
- tool observation；
- available-tool metadata。

只监督当前 target：

```text
memory token
+ arguments
+ EOC（仅 EOC-only/TapMem）
+ response end
```

### 9.6 split 规则

必须在拆 step 之前按 episode 和任务族切分：

```text
train / validation / synthetic_test
```

禁止随机打散 step 后切分，否则同一任务前几个步骤可能在训练集、后几个步骤在测试集。

分组键至少包括：

```text
template_id
asset_seed_family
semantic_signature
```

同一模板的近重复变体全部进入同一 split。正式 Toolathlon 10 题不属于这三个合成 split。

### 9.7 observation 长度

Toolathlon 中 PDF 和目录 observation 可能很长。所有方法使用同一上下文策略：

- 重要短 observation 原样保留；
- 超长结果保存为 artifact，并在上下文中保留稳定引用；
- 如需截断，固定保留 head、tail 和总长度信息；
- 不允许只为 TapMem 提供更长上下文；
- 不允许截掉 target memory token、arguments 或 EOC。

当前 dataset 默认 `max_length=1024`，正式实验前必须用真实轨迹统计长度，并预先固定更合适的上下文长度或 artifact 策略。

## 10. 对当前代码的必要修改

### 10.1 当前开环限制

当前 `compositional/dataset.py` 的训练格式是：

```text
user
→ tool token 1 + function call 1
→ tool token 2 + function call 2
→ ...
```

它不包含 tool observation。当前 `generate_complete_sequence` 也会从用户输入一次生成完整序列。

因此当前代码只能评估开环轨迹模仿，不能直接完成 Toolathlon。

### 10.2 需要新增的能力

实现阶段至少需要：

1. `episode -> step` 转换器；
2. observation-aware dataset loader；
3. stable tool manifest 和 token 映射；
4. per-task available-tool mask；
5. 单次调用生成和停止逻辑；
6. memory token 到 MCP tool call 的 adapter；
7. observation 回填和上下文管理；
8. Toolathlon decoupled runner 接口；
9. 每 episode reset；
10. 最终 evaluator 和轨迹日志收集。

### 10.3 TapMem 决策边界

当前 TCRA 训练会收集：

- assistant 起始位置的第一个工具目标；
- EOC 后紧邻的下一个工具目标。

在 Toolathlon 闭环中，EOC 后必须先执行工具并插入 observation，下一工具不会直接紧邻 EOC。因此本实验需要把：

```text
题目 + 历史 + 最新 observation
→ 下一工具
```

的 assistant-start 位置显式标记为新的决策边界，并在这些位置训练、应用 TCRA。

逐步样本下：

- EOC 负责标记本次工具参数 span 的结束；
- TCRA 在每次新的 assistant-start 边界选择工具；
- 下一决策使用真实 observation 后的 hidden state；
- 不把 EOC 后的 EOT 错当成下一工具监督。

这是为了接入交互式 benchmark 的必要适配。论文中应明确它主要验证 observation-conditioned boundary routing，不应表述成“模型在不执行工具时连续预测了整个工具链”。

## 11. 训练协议

### 11.1 公平性约束

三种方法固定相同：

- backbone checkpoint；
- tokenizer；
- tool manifest；
- train/validation/synthetic-test episode；
- step 展开规则；
- batch size；
- optimizer；
- tool embedding learning rate；
- epoch/step 数；
- context length；
- observation 截断；
- available-tool mask；
- checkpoint 选择规则；
- 训练随机 seed 列表。

只允许方法定义中的 EOC 和 TCRA 不同。

### 11.2 输入中删除工具文档

学生模型训练和推理时不接收自然语言工具说明或 JSON schema。它只通过：

- memory token；
- 训练例子；
- 当前可用 token mask；
- 调用后的 observation；

学习工具能力和参数模式。

teacher 生成 GT 时可以看到工具文档，因为 teacher 的职责是产生正确调用；这不等于学生模型在测试时能看到文档。

### 11.3 推荐训练阶段

#### Stage A：smoke

- 每工具至少 3 个 accepted episode；
- 只跑一个 seed；
- 检查 loss、memory-token 选择、JSON 生成和 checkpoint I/O；
- 不报告为正式结果。

#### Stage B：pilot

- 每工具至少 10 个 accepted episode；
- 加入 2–8 步任务；
- 在 synthetic held-out 环境中真实 rollout；
- 检查 TokMem/TapMem 是否有方向性差异。

#### Stage C：formal

- 每工具建议 20–50 个 accepted episode；
- 使用预先固定的 3 个训练 seed；
- 加入 9–15 步任务和干扰工具；
- checkpoint 只按 synthetic validation 选择；
- 冻结后才运行 Toolathlon。

### 11.4 checkpoint 选择

不能根据 10 道正式题的分数选择 checkpoint、超参数、prompt 或数据配比。

checkpoint 选择只使用：

- synthetic validation loss；
- next-tool 指标；
- function-call exact match；
- schema-valid rate；
- synthetic validation rollout Pass。

正式 Toolathlon 运行一次后，不得回到训练阶段针对失败题生成相似数据。若这样做，后续结果必须标为 test-aware adaptation，不能和主结果合并。

## 12. 闭环推理协议

### 12.1 每次正式运行

```text
1. 从同一 prepared image 启动新容器
2. 复制该题 initial workspace
3. 运行该题 preprocess
4. 启动 MCP gateway
5. 读取官方 task 和 system prompt
6. 构造当前 available-tool mask
7. 调用冻结的 TokMem/EOC-only/TapMem
8. 解析并执行一个工具调用
9. 回填真实 observation
10. 重复 7–9
11. claim_done / EOS / 达到限制
12. 运行原始 evaluator
13. 保存完整轨迹、最终 workspace 和 eval 结果
14. 销毁容器
```

模型输入不包含工具说明、GT 或 evaluator。

### 12.2 单轮生成

TokMem：

```text
context
→ memory token
→ JSON arguments
→ native response end
```

TapMem：

```text
context
→ memory token
→ JSON arguments
→ EOC
→ 立即停止本轮生成
```

TapMem 生成 EOC 后，adapter 必须先执行工具并回填 observation，不能继续开环生成下一 memory token。

TokMem 没有 EOC，adapter 应在检测到第一个完整、schema 可解析的 JSON arguments 对象和本轮 response end 后停止；不能让 TokMem 在工具执行前继续生成第二个工具调用。解析和 stopping criteria 必须在三种方法正式测试前冻结。

### 12.3 adapter 允许和禁止的修正

允许：

- memory token 到稳定工具 ID 的确定性映射；
- benchmark namespace 转换；
- `<WORKSPACE>` 到本次绝对 workspace 的替换；
- JSON canonicalization 中无语义的空白处理。

禁止：

- 根据 GT 自动选择工具；
- 自动补写缺失业务参数；
- 模糊匹配到另一个工具；
- 用 oracle 修正文件名或单元格；
- 在模型失败后偷偷重试另一个工具；
- 根据 evaluator 错误修改模型输出。

### 12.4 错误处理

所有方法使用相同策略：

- JSON 解析失败：作为一次失败调用记录，并向模型返回固定格式错误 observation；
- schema 校验失败：同上；
- 调用未提供工具：同上；
- 工具执行异常：原始错误作为 observation；
- 每次错误都计入轮数；
- 达到预先固定的连续错误或总轮数上限后终止；
- 超时和异常退出均计为 Fail。

不静默丢弃失败任务。

### 12.5 终止

优先使用 `claim_done`。如果官方 scaffold 允许普通回复结束，则：

- 将 EOS 视为模型主动结束；
- 仍然运行 evaluator；
- 不因为没有 `claim_done` 自动判成功；
- 最终状态不正确即 Fail。

## 13. 评估指标

### 13.1 Toolathlon 主指标

主指标只使用官方 evaluator：

- 每次运行的 `Pass`；
- 10 题的平均成功率；
- 每题成功/失败；
- 平均工具轮数；
- 超时率。

固定 revision 的 `container_eval` 还会接收 host-agent exit code，因此官方 Pass
不是一个绕过 agent 协议的“纯环境状态”分数。本实验保留这一行为：成功修改环境但
host loop 异常退出的运行不能改写成官方 Pass。若 evaluator 原始输出允许分离
final-state 检查，可额外报告 `environment_effect_pass` 作为研究性指标，但必须与
官方 Pass 分列，不能替代主指标。

正式资源允许时，每个 checkpoint 对每题运行三次并报告：

- `Pass@1`：所有运行的平均成功率；
- `Pass@3`：每题三次中至少成功一次的比例；
- `Pass³`：每题三次全部成功的比例。

同时报告 3 个训练 seed 的均值、标准差和逐题配对结果。

10 题、3 个训练 seed 和每题 3 次 rollout 产生的是重复测量，不是 90 个独立测试任务。统计分析应以 task 为配对单位，并同时公开逐题、逐 seed、逐 rollout 原始结果。

### 13.2 轨迹诊断指标

只在 synthetic held-out 或 teacher-forced observation 状态上报告：

- next-tool accuracy；
- Tool precision / recall / F1；
- function-call exact match；
- 当前仓库定义的 Arguments F1；
- JSON parse rate；
- schema-valid rate；
- execution success rate；
- error-recovery success。

当前仓库的 Arguments F1 是规范化后的完整 function-call 集合 F1，不是参数字段级 F1。如果新增字段级指标，应命名为 `argument_field_f1`，避免混淆。

### 13.3 为什么原题不使用 Tool F1 作为主指标

Toolathlon 没有唯一 gold trajectory。同一个最终状态可能通过：

- 不同工具；
- 不同调用顺序；
- 不同文件读写粒度；
- terminal 或 filesystem 的等价操作。

因此，把模型调用与一条 canonical trajectory 对比会惩罚其他正确路径。原题只用最终 evaluator 判定完成度，调用指标只用于错误分析。

### 13.4 干扰工具实验

分开报告：

1. **Official-menu。** 使用原始 `task_config` 产生的候选工具。
2. **Distractor stress。** 在官方菜单外加入预先固定的、真实可执行且 schema 稳定的额外工具。

stress 条件记录：

- 新增工具数量；
- 新增 server 数量；
- 与正确工具的名称/说明相似度；
- 工具选择准确率和最终 Pass 的变化。

不能把 stress 条件的结果当成官方菜单结果。

## 14. 消融实验

优先级从高到低：

1. TokMem vs. EOC-only vs. TapMem；
2. 有无真实 observation 上下文；
3. 官方菜单 vs. 额外干扰工具；
4. clean canonical only vs. 加入 recovery 样本；
5. 禁用 `python_execute` vs. workspace-only sandbox；
6. 每工具 3、10、20–50 条成功 episode；
7. 单步任务 vs. 长流程任务；
8. docs-only silver trajectory vs. 真实执行 trajectory。

docs-only 数据如果使用，必须单独标注。主结论应来自真实执行验证的数据。

## 15. Python/terminal 万能脚本问题

除 `imagenet` 外，多数选定任务提供 `python_execute`，其中八题还提供 terminal。模型可能用一个大脚本在工具内部自行：

```text
枚举文件
→ 读取内容
→ 计算
→ 写结果
```

这种调用仍然需要真实环境，但它把 observation-dependent 决策隐藏进工具内部，会使实验主要测代码生成，而不是多轮工具路由。

因此：

- official-menu 主结果保留原配置并如实记录；
- 主条件把 `python_execute` 限制在 workspace-only、无网络 jail 内；
- 受控分析完全禁用 `python_execute`；
- teacher 主训练轨迹拒绝大段万能脚本；
- terminal 本身作为目标工具时，使用有限、可审计的命令；
- 分别报告 native multi-tool 和 sandboxed-Python 结果。

## 16. 数据质量门槛

### 16.1 task gate

- instruction 自洽；
- 输入文件齐全；
- oracle 可重复；
- evaluator 确定性；
- initial state 必须失败；
- oracle state 必须通过；
- 不依赖公网实时数据；
- 不含测试实体。

### 16.2 trajectory gate

- preprocess 成功；
- 所有调用来自 available menu；
- 参数通过 schema；
- observation 来自真实执行；
- 无 GT/evaluator 访问；
- final evaluator Pass；
- 无未解释的外部状态；
- 无错误调用作为正标签。

### 16.3 dataset gate

- 按 episode/template 分 split；
- 没有跨 split 近重复；
- 每个目标工具在 `train` split 中至少有 3 个不同的真实成功 episode；validation
  和 synthetic test 的调用只报告、不计入该门槛；
- 路径已规范化；
- tool manifest hash 一致；
- target 未被截断；
- TokMem、EOC-only、TapMem 使用同一 sample ID 集合。

### 16.4 evaluation gate

- 每次运行 fresh workspace；
- 方法顺序随机或轮换；
- 同一超时和最大轮数；
- evaluator 未修改；
- 失败和超时全部入表；
- 正式测试后不再调参。

## 17. 实现目录与 artifact 布局

当前实现使用独立顶层轨道 `compositional_toolathlon/`；合成数据仍与源码分开保存：

```text
compositional_toolathlon/data/generated/
├── manifests/
│   ├── tool_manifest.json
│   ├── tool_manifest.sha256
│   └── excluded_tools.json
├── tasks/
│   ├── train/
│   ├── validation/
│   └── synthetic_test/
├── episodes/
│   ├── candidates/
│   ├── accepted/
│   └── rejected/
├── steps/
│   ├── train.json
│   ├── validation.json
│   └── synthetic_test.json
├── prompts/
│   ├── generators/
│   ├── teacher/
│   └── verifier/
├── provenance/
│   ├── generation_runs.jsonl
│   └── leakage_audit.json
└── README.md

compositional_toolathlon/
├── configs/
├── prompts/
├── scripts/
├── tests/
├── environment.py
├── manifest.py
├── context.py
├── episode_to_steps.py
├── dataset.py
├── masked_routing.py
├── training.py
├── main_train.py
├── decode_one_call.py
├── rollout.py
├── generate_tasks.py
└── teacher.py

results/toolathlon_<run_name>/
├── configs/
├── checkpoints/
├── synthetic_metrics.json
├── official_rollouts/
├── eval_stats.json
├── tool_manifest.json
└── run_summary.md
```

`data/generated/`、`artifacts/` 和 `runs/` 是运行时目录，默认不提交；成功实验再归档到
`results/toolathlon_<run_name>/`。

## 18. 分阶段执行计划

### Phase 0：环境

1. 补齐固定 commit 的 Toolathlon runner。
2. 拉取 prepared image。
3. 检查 Docker daemon 权限。
4. 启动四个 MCP server。
5. 使用 `tools/list` 生成 manifest。
6. 用无副作用调用确认 schema 和行为。

建议先用：

- `excel-data-transformation` 覆盖 excel、filesystem、terminal；
- `imagenet` 覆盖 pdf-tools、filesystem；
- 一个有 preprocess 的任务覆盖 archive 初始化。

### Phase 1：smoke 数据

1. 每工具生成至少 3 个成功 episode。
2. 执行 generator、teacher、verifier 全链路。
3. 人工抽查至少 20–50 条轨迹。
4. 运行 episode-to-step 转换。
5. 检查 observation 和 loss mask。

这一步只证明管线工作，不报告方法结论。

### Phase 2：pilot

1. 每工具至少 10 个成功 episode。
2. 训练一个 seed 的 TokMem、EOC-only、TapMem。
3. 在 synthetic validation 做 teacher-forced 和 rollout。
4. 在 2–3 道 Toolathlon 题上只检查工程链路。

这些题的结果不能用于调超参数后再作为正式结果。

### Phase 3：formal 数据和训练

1. 冻结 generator、数据配比、split 和 manifest。
2. 每工具扩到建议的 20–50 个 accepted episode。
3. 运行泄漏审计。
4. 训练预先固定的 3 个 seed。
5. 只按 synthetic validation 选择 checkpoint。
6. 冻结模型和推理配置。

### Phase 4：正式 Toolathlon

1. 逐 checkpoint、逐题、逐 trial 从新容器开始。
2. 保存完整 trajectory 和最终 workspace。
3. 运行原始 evaluator。
4. 汇总 Pass、Pass@3、Pass³、轮数和失败类型。
5. 先完成 official-menu，再运行 distractor stress。
6. 不根据失败题返回修改训练数据。

## 19. 失败类型

至少标注：

```text
wrong_tool
tool_not_available
invalid_json
schema_invalid
bad_argument_value
path_error
execution_error
observation_misread
planning_error
write_not_verified
premature_done
max_turns
timeout
evaluator_mismatch
environment_failure
```

分析时区分：

- 模型错误；
- adapter 错误；
- MCP server 错误；
- preprocess/evaluator 错误；
- 环境基础设施错误。

环境失败不能静默删除。应单独列出，并按预注册规则决定是否重跑。

## 20. 成功标准

在以下条件同时满足时，才能认为实验支持 TapMem：

1. TapMem 的 Toolathlon 最终 Pass 高于 TokMem；
2. EOC-only 不能解释全部收益，或论文明确把收益归因于 EOC；
3. 多个训练 seed 的方向基本一致；
4. TapMem 没有使用更多训练数据、工具文档或更长上下文；
5. 收益不是由 ground-truth tool forcing、万能 Python 脚本或更少干扰工具造成；
6. synthetic held-out 的 next-tool 或 schema-valid 指标能解释部分最终收益；
7. 所有正式题都从干净环境运行，失败任务没有被丢弃。

如果 TapMem 只提高 synthetic Tool F1，而 Toolathlon Pass 没有提高，结论只能是：

> TapMem 改善了该设置下的工具路由或轨迹模仿，但没有证明完整任务成功率提高。

## 21. 论文报告口径

推荐写法：

> We evaluate TokMem and TapMem on a preregistered ten-task local subset of
> Toolathlon-Verified. Both methods are trained on the same independently
> generated tool-use episodes. The training tasks, assets, states, and
> trajectories do not overlap with Toolathlon tasks. Every training
> trajectory is executed against the pinned MCP environment and retained only
> after a deterministic task-specific evaluator passes. During training and
> inference, the student model does not receive natural-language tool
> descriptions. At test time, the frozen model interacts with a fresh task
> container one tool call at a time, conditions each decision on actual tool
> observations, and is scored only by the original final-state evaluator.

同时必须声明：

- 这是 `local-office-10` 自定义子集，不是 Toolathlon 108 题全集；
- 模型输入移除了官方通常提供的工具文本说明，因此不与 leaderboard 直接可比；
- 这是 known-tool、unseen-task 设置，不是 unseen-tool；
- 工具 manifest 和测试子集在看到结果前固定；
- 官方轨迹未用于训练；
- Verified 轨迹的 do-not-train 政策得到遵守；
- 如果训练生成器预先知道四个 server，应明确写出；
- official-menu 与受控/干扰条件分开报告。

## 22. 开始实现前的检查清单

- [ ] 完整 Toolathlon runner 与任务数据固定到同一 commit
- [ ] prepared image digest 已保存
- [ ] 四个 MCP server 能启动
- [ ] local tools 处理策略已固定
- [ ] 工具 manifest 和 schema hash 已生成
- [ ] 工具数量不超过 token 容量，或扩容方案已固定
- [ ] 10 个测试 task ID 已冻结
- [ ] generator 无法读取 Toolathlon task bundle
- [ ] teacher 无法读取 oracle、GT 和 evaluator
- [ ] synthetic evaluator 有正例和负例测试
- [ ] 每次 trajectory 从 fresh workspace 开始
- [ ] canonical trajectory 选择规则已固定
- [ ] episode split 在 step 展开前完成
- [ ] observation span 不计算 loss
- [ ] TCRA assistant-start 决策边界已实现并测试
- [ ] TokMem 没有 EOC 和 logit bias
- [ ] EOC-only 没有 TCRA
- [ ] TapMem 使用 EOC 和 TCRA
- [ ] 三种方法 sample ID、上下文和 mask 完全一致
- [ ] `--use_ground_truth_tools` 在正式推理中关闭
- [ ] adapter 不进行语义参数修复
- [ ] `python_execute`/terminal 条件已预注册
- [ ] 正式 evaluator 未修改
- [ ] 超时、失败和环境错误记录规则已固定
- [ ] checkpoint 不根据正式测试分数选择
- [ ] 正式测试后不再针对失败题补训练数据
