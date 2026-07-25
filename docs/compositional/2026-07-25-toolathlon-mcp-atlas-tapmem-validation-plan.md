# TapMem 在 Toolathlon 和 MCP-Atlas 上的验证方案

> 调研日期：2026-07-25
> 这是一份实验设计，不包含尚未运行的新结果。

## 1. 先说结论

TapMem 可以在这两个 benchmark 上测试，也能在完全不使用两套 benchmark 题目和轨迹的情况下准备训练数据。

关键是把两套 benchmark 当成“工具目录”，而不是训练集：

- 从官方配置中找到它们实际使用的 MCP server、上游仓库和启动命令；
- 在一个全新的目录、数据库或测试账号中启动相同工具；
- 自己生成新的文件、实体和任务，真正调用工具得到轨迹；
- 训练时只使用这些独立生成的轨迹；
- 最后把冻结的模型放回 Toolathlon 或 MCP-Atlas，完成从未见过的正式题目。

正确的做法是：

1. 先固定要测试的题目和这些题目会提供的全部候选工具。
2. 从官方 server 配置找到相同的开源工具，并固定版本。
3. 优先从有明确许可证的外部 MCP 数据中筛选同源样本；找不到时，自己生成新的短任务。
4. 在与评测环境隔离的干净环境中重新执行，只保留实际成功且检查通过的记录。
5. TokMem、EOC-only 和 TapMem 使用完全相同的练习训练。
6. 训练完成后冻结模型，再进入 benchmark 完成从未见过的题目。
7. 最终结果只看 benchmark 的官方评分。

这个实验验证的是：

> 模型以前学过怎样使用这些工具，但从未见过测试题、测试实体和测试环境状态。TapMem 能否比 TokMem 更稳定地完成新的长流程任务？

它不是“从未见过工具也能直接使用”的实验。

实际执行时建议先做 MCP-Atlas，再做 Toolathlon：

- MCP-Atlas 的公开工具去重后是 220 个，能放进当前最多 247 个工具记忆的实现。
- Toolathlon 涉及 604 个 MCP 工具和一些本地工具，全集超过当前容量；先跑小规模子集更现实。

## 2. 两个 benchmark 到底是什么

| 问题 | Toolathlon | MCP-Atlas |
|---|---|---|
| 一道题长什么样 | 一个完整任务目录，包含题目、工具配置、初始环境和隐藏评分程序 | Parquet 文件中的一行，包含题目、候选工具、答案要点和参考轨迹 |
| 模型在测试时能看到 | 题目、当前可用工具、每次工具执行结果 | `PROMPT`、`ENABLED_TOOLS`、每次工具执行结果 |
| 模型不能看到 | 正确答案、最终正确状态、评分代码、官方轨迹 | `GTFA_CLAIMS`、`TRAJECTORY` |
| 模型最后要做什么 | 真正修改应用、文件或服务状态，并声明任务完成 | 调用工具收集信息，最后给出文字答案 |
| 官方怎么判分 | 检查最终环境状态是否正确 | 检查最终答案覆盖了多少条标准答案要点 |
| 外部可以完整运行多少题 | 108 题 | public 500 题；private 500 不公开 |
| 当前 TapMem 能否直接覆盖全部工具 | 不能，工具数超过 247 | 能，公开集共 220 个工具 |

## 3. Toolathlon

### 3.1 一道题的数据结构

Toolathlon-Verified 共有 108 道公开评测题，分为 Research、Campus、Finance、Tech、Business/Office、Daily 和 E-commerce 七类。

一道题不是简单的“问题—答案”，而是一个任务包，通常包含：

```text
docs/task.md              用户要完成的任务
system prompt             agent 的通用规则
task_config.json          这道题需要启动哪些 MCP server 和本地工具
preprocess/               测试前如何初始化环境
initial_workspace/        初始文件和状态
groundtruth_workspace/    隐藏的正确状态
evaluation/               检查最终结果的程序
```

整个 benchmark 涉及：

- 32 个 MCP server；
- 604 个 MCP 工具；
- 7 组本地辅助工具，共 16 个；
- 每题平均向模型提供 69.9 个候选工具，最少 28 个，最多 128 个；
- 72/108 道题需要先初始化应用状态。

候选工具中有很多并不是完成任务所必需的。模型不仅要会调用工具，还要从几十个工具中选出正确的几个。

### 3.2 题目和“回答”是什么样的

Toolathlon 中的“回答”通常不是一段可以和标准答案逐字比较的文字，而是三部分：

1. 模型调用工具完成任务的过程；
2. 最后留在邮箱、Canvas、Notion、Google Sheets、Snowflake、WooCommerce、
   Kubernetes 或本地文件中的真实结果；
3. 一句简短的完成声明，通常通过 `claim_done` 结束任务。

真正决定得分的是第二部分。官方不要求模型复现某条固定轨迹；只要最终状态满足
evaluator，不同的工具顺序也可以成功。少数检索类任务还会要求模型在最后返回标题、
URL 等文字内容。

按官方完整任务索引逐项统计，108 道题覆盖七个方面：

| 类别 | 题数 | 主要内容 |
|---|---:|---|
| Research & Academic | 15 | 论文检索、LaTeX/BibTeX、学术主页、文献和会议信息整理 |
| Campus & Study | 18 | 选课、考试安排、Canvas 作业和测验、教学与招生事务 |
| Daily & Entertainment | 17 | 路线和行程、饮食、音乐视频、体育数据、个人事务 |
| Finance & Market | 10 | 股票和市场数据、量化分析、回测、财务表格 |
| Office & Business | 18 | 报销、发票、HR、审计、数据清洗、隐私脱敏和邮件通知 |
| Shopping & E-commerce | 11 | 商品筛选、库存同步、召回、客户邮件和在线表单 |
| Tech & Dev | 19 | GitHub、Hugging Face、W&B、数据处理和 Kubernetes 运维 |

下面是七个代表例子。这里写的是题意和正确结果的形状，不公开隐藏的
ground truth 数值。

#### 例 1：查找 Alita 论文

题目要求在截止日期之前发表的论文中，找到标题含 “Alita” 的 agentic reasoning
论文最新 arXiv 版本。

模型需要搜索 arXiv 和代码仓库、核对版本，然后：

- 把论文保存为 `alita_2505.20286.pdf`；
- 返回论文标题；
- 返回 arXiv abstract 页面 URL；
- 返回代码仓库 URL。

题面要求的最终文字是：

```text
title: Alita: Generalist Agent Enabling Scalable Agentic Reasoning with Minimal Predefinition and Maximal Self-Evolution
arxiv_abs_url: https://arxiv.org/abs/2505.20286
code_url: https://github.com/CharlesQ9/Alita
```

这个例子也说明必须区分“题面要求”和“当前评分脚本实际检查什么”：固定到 Verified
快照 `d57361c0` 时，evaluator 实际只检查下载的 PDF 是否是 arXiv 最新版本，没有把
最后这三行文字纳入 Pass/Fail。

#### 例 2：批改 Canvas Python 作业

题目要求模型从邮箱找到每位学生最新的 Homework 2 附件，下载 Python 文件，按照
本地作业要求运行检查，然后在 Canvas 中打分：通过为 10 分，否则为 0 分；重复提交
只看最新版。

一个典型过程是：

```text
搜索邮件
→ 找到每位学生的最新附件
→ 下载到 workspace
→ 用 terminal 执行
→ 根据运行结果更新 Canvas 分数
→ claim_done
```

这里的“答案”不是回复一张成绩表，而是 Canvas 中每个学生的分数真的被更新正确。

#### 例 3：分析国际米兰欧冠决赛

题目要求从 UEFA 官方数据整理国际米兰 2023 和 2025 年欧冠决赛表现，严格填写
Google Sheets 中的三个工作表，并计算 `2025 - 2023` 的差值。

正确结果包括：

- 两个年份的统计表填写完整；
- `StatsDifference` 中的差值正确；
- 缺失值按要求标记；
- 本地 `sheet_url.txt` 保存完成后表格的只读链接。

#### 例 4：分析 NVIDIA 机构持仓

题目要求整理 2023 Q1 到 2024 Q4 八个季度的 NVIDIA 机构持仓，回溯调整拆股影响，
忽略期权，只统计普通持股。

模型需要检索数据、计算并填写给定 Excel 模板，最终把文件改名为
`results.xlsx`。评分重点是单元格内容、百分比精度、普通股筛选和文件名，而不是模型
在对话里复述了多少分析。

#### 例 5：审核差旅报销

题目要求检查 workspace 中每笔报销是否有完整发票、申报金额是否和发票一致。

模型需要根据不同情况采取不同动作：

- 材料不完整或金额不符：给员工发邮件并抄送经理；
- 材料有效：把报销记录写入 Snowflake；
- 超过限额：在数据库中标记 `flag = 1`，同时发送超额通知邮件。

评分器会同时检查邮件收件人、主题、数据库行和异常标记。

#### 例 6：处理滞销商品

题目要求找出入库超过 90 天、近 30 天销量少于 10 件的 WooCommerce 商品，把它们
移到 `Outlet/Clearance` 分类，并用给定模板给订阅用户发邮件。

正确结果是商品分类真的被修改，而且邮件中的商品顺序满足“先按入库时间、同一天再按
折扣率”的规则。只在最终回答里列出商品名而不修改网店，得不到分。

#### 例 7：部署 Kubernetes PR 预览环境

题目要求把 GitHub 仓库 `SimpleShopping` 的 `feature/pr-123` 分支部署到
Kubernetes，按照 `preview.yaml` 创建 ConfigMap，让服务长期可从
`localhost:30123` 访问，运行该分支测试并填写测试报告。

正确结果包括：

- Kubernetes 资源处于正确状态；
- 页面能够访问；
- 测试确实运行；
- workspace 根目录存在完成的 `filled-test-results-report.md`。

这类题的过程可能是：

```text
读取 GitHub 分支和配置
→ 创建或修改 Kubernetes 资源
→ 检查 Pod、Service 和端口
→ 访问页面
→ 执行测试
→ 写测试报告
→ claim_done
```

这些例子说明 Toolathlon 同时考察“选对工具”“参数写对”“根据 observation
继续处理”和“把最终环境真正改对”。这正是 TapMem 可以验证 procedural memory
是否有效的地方。

### 3.3 一道题怎么运行

一次测试大致是：

```text
初始化邮箱、网店、文件、表格或其他应用状态
→ 给模型题目和当前可用工具
→ 模型选择工具并生成参数
→ 官方环境真正执行工具
→ 把执行结果或错误返回模型
→ 模型继续选择下一步
→ 模型调用 claim_done
→ 官方程序检查最终应用状态
```

因此 Toolathlon 主要考察：

- 长流程规划；
- 跨应用协作；
- 根据工具返回结果调整下一步；
- 出错后的恢复；
- 是否真的把环境改成了目标状态。

它不要求模型复现某一条固定工具轨迹。只要最终状态正确，走不同路径也可以成功。

### 3.4 官方指标

官方对同一个冻结模型的每道题运行三次：

- `Pass@1`：三次成功率的平均值；
- `Pass@3`：三次中至少成功一次的题目比例；
- `Pass³`：三次都成功的题目比例；
- 另外报告平均执行轮数。

如果只运行一次，就只能报告这一次的成功率，不能写 Pass@3 或 Pass³。

### 3.5 旧版轨迹什么时候发布，结构是什么

这里的“旧版”指
[Toolathlon-Trajectories](https://huggingface.co/datasets/hkust-nlp/Toolathlon-Trajectories/blob/main/README.md)，不是后来的 `Toolathlon-Verified_Trajectories`。

发布时间应分开看：

| 时间 | 发生了什么 |
|---|---|
| 2025-10-20 | Hugging Face 仓库创建 |
| 2025-10-22 | 第一批真实 `.jsonl` 轨迹文件上传；这是最接近“数据开始公开”的日期 |
| 2025-10-23 | 完整 README 和数据说明加入仓库 |
| 2025-10-29 | Toolathlon 论文公开 |
| 2025-12-05 | 又增加 Claude 4.5 Opus、DeepSeek 3.2 Thinking、Gemini 3 Pro Preview 和 GPT-5.1，每个模型各 3 次运行 |

因此，简短写法是“旧轨迹数据于 2025 年 10 月下旬发布，并在 2025 年 12 月补充模型”。不能把 12 月 5 日的最后数据更新时间说成首次发布日期。

仓库不是按 train/test split 组织，而是“一个模型的一次完整 benchmark 运行对应一个文件”：

```text
Toolathlon-Trajectories/
  README.md
  gpt-5-high_1.jsonl
  gpt-5-high_2.jsonl
  gpt-5-high_3.jsonl
  claude-4.5-sonnet-0929_1.jsonl
  ...
```

文件名格式是：

```text
{model_name}_{run_number}.jsonl
```

官方 README 写的是 17 个模型、每个模型 3 次运行，共 51 个 JSONL
文件；每个文件约有 108 行，总计超过 5,000 条任务执行记录。但这个数字本身漏掉了
`minimax-m2_1/2/3.jsonl`。逐项查看
[2025-10-27 的冻结文件清单](https://huggingface.co/datasets/hkust-nlp/Toolathlon-Trajectories/tree/b53e3f808707dbe90cd9d7216a1d425b61466a26)
可以数到 18 个模型、54 个运行文件。2025-12-05 又加入 4 个模型的 12
个文件，所以当前仓库是 22 个模型、66 个运行文件，而不是按照旧 README
推出来的 21/63。实验中应固定 commit 并保存实际文件清单，不能只引用 README
中的汇总数字。

每个 JSONL 文件中，一行是一道题的一次完整运行。逻辑结构如下；为了让 Hugging Face viewer 展示，多个复杂字段实际保存为 JSON 字符串，下载后需要再做一次 `json.loads`：

```json
{
  "modelname_run": "claude-4-sonnet-0514_1",
  "task_name": "find-alita-paper",
  "task_status": {
    "preprocess": "done",
    "running": "done",
    "evaluation": true
  },
  "config": {},
  "request_id": "...",
  "initial_run_time": "2025-10-19T04:00:40",
  "completion_time": "2025-10-19T04:02:00",
  "tool_calls": [],
  "messages": [],
  "key_stats": {},
  "agent_cost": {}
}
```

主要字段含义：

| 字段 | 实际内容 |
|---|---|
| `modelname_run` | 模型和第几次独立运行 |
| `task_name` | 题目 ID；同一道题会在不同模型、不同 run 文件中重复出现 |
| `task_status` | 环境初始化是否完成、agent 是否正常结束、官方 evaluator 是否通过 |
| `config` | `task_str`、所需 MCP server、本地工具、system prompt、workspace、步数上限和 evaluator 命令等 |
| 顶层 `tool_calls` | 这道题提供的工具定义，含工具名、说明和参数 schema；名字容易误解，它主要不是实际执行序列 |
| `messages` | 完整对话；包含 user、assistant 和 tool 消息 |
| `key_stats` | 轮数、实际工具调用数、token、耗时和截断次数等 |
| `agent_cost` | API 请求数、输入输出 token 和估算费用 |
| 时间与请求字段 | `request_id`、开始时间和结束时间 |

它是“运行日志集合”，不是一套可以单独启动的任务包。JSONL 里的 `config` 会引用 `tasks/finalpool/...`、initial workspace 和 evaluator 命令，但对应文件和评分代码仍在 Toolathlon GitHub 仓库及运行环境中；只下载旧轨迹库不能重新执行这些题目。

真正执行过的工具调用在 `messages` 中：

```text
assistant message
  └─ tool_calls:
       id
       function.name
       function.arguments

tool message
  ├─ tool_call_id
  └─ content，也就是 observation
```

所以，把旧轨迹转成 TapMem 数据时，应从 `messages` 配对“assistant 的调用”和“紧随其后的 tool observation”。不能直接把顶层 `tool_calls` 当成 gold 调用序列；它更接近当前题目的候选工具菜单。

数据同时保留成功和失败运行。若只想学习成功过程，至少过滤：

```text
task_status.preprocess == "done"
task_status.running == "done"
task_status.evaluation == true
```

但即使最终成功，中间也可能有报错调用和恢复过程；是全部保留还是只保留成功 step，需要在数据转换规则中单独说明。当前仓库的文件内容需要先同意分享联系信息后才能下载，下载时还应检查页面展示的访问条件。

### 3.6 新版和旧版到底差在哪里

这里的两版不是两套完全不同的题：

- 旧版是 2025 年 10 月公开、后来进入 ICLR 2026 论文实验和旧排行榜的
  Toolathlon；
- 新版是 2026 年 6 月 30 日发布的 `Toolathlon-Verified`，作者把它称为当前
  benchmark 的最终校验版。

两版都是同一组 108 个任务，任务没有增加、删除或改名。新版也不是从旧版挑出来的
一个小子集。真正变化的是题面、初始环境、ground truth、评分程序和运行基础设施。

官方给出的修改规模是：

| 修改位置 | 涉及多少道题 |
|---|---:|
| 至少有一处实际修改的任务包 | 83/108 |
| `evaluation/` | 76 |
| `preprocess/` | 28 |
| `groundtruth_workspace/` | 19 |
| `initial_workspace/` | 14 |
| 模型能够看到的题面 | 14 |

这些数字有重叠。例如，同一道题可能同时修改题面、标准答案和评分程序。

这不是简单改错别字。旧版中存在几类会影响分数的问题：

- 题面要求四项，但 evaluator 实际检查五项，正确完成也可能被判错；
- evaluator 的程序错误可能漏掉真正的字段错误，让错误结果通过；
- 上一次运行留下的邮件、附件或应用状态可能污染下一次评分；
- 随机初始化没有固定种子；
- Notion、Canvas、WooCommerce 等服务写入成功后不能立刻读到，评分程序可能把服务
  延迟误判成模型失败；
- 时间敏感的数据和题目要求会过期。

Verified 对这些问题做了对应修复，包括对齐题面、ground truth 和 evaluator，清理
历史状态，固定随机种子，把 evaluator 和 ground truth 从 agent 可见环境中移走，
加入服务就绪检查和有上限的重试，并改善不同模型 provider 和 agent scaffold
之间的调用接口。

轨迹的发布方式也不同：

| | 旧版轨迹 | Verified 轨迹 |
|---|---|---|
| 主要组织方式 | 一个模型的一次运行对应一个 JSONL | 一个模型的三次运行打成一个归档包 |
| 内容 | 每题的扁平运行记录 | 完整运行目录、任务产物和 `eval_stats.json` |
| 使用政策 | 数据卡标注 CC BY 4.0 | 明确标注 `do-not-train`，只允许评测复现、审计、错误分析和污染研究 |
| 排行榜位置 | 官方历史归档 | 当前官方排行榜 |

官方已经明确说明，Verified 开启了新的分数序列。两版虽然题目 ID 相同，但题面、预期
输出、evaluator、初始状态和运行栈发生了变化，所以旧版和新版分数不能直接比较。

### 3.7 哪一版更受认可，旧版有没有正式发表

“认可度”需要分成两件事。

从同行评审和历史积累看，旧版并不是一个没人认可的临时数据集。论文
*The Tool Decathlon: Benchmarking Language Agents for Diverse, Realistic, and
Long-Horizon Task Execution* 已正式发表在 **ICLR 2026 主会，Poster**。ICLR
论文中的实验和最初积累的模型结果对应的是原始版本。旧排行榜在归档时已有 51
个模型，因此它的历史结果更多。

从现在开展新实验的评测可信度看，`Toolathlon-Verified` 更合适。作者将它列为
current benchmark 和 verified final release，当前官方排行榜只把 Verified
作为正式分数序列，旧版排行榜已经标为 archived snapshot。Verified 修复的也不是
无关紧要的显示问题，而是会产生误判、漏判和跨运行污染的测量问题。

因此本文的口径应是：

- TapMem 主结果使用 Toolathlon-Verified；
- 旧版最多作为补充实验，用于和 ICLR 论文或历史模型结果对照；
- 报旧版结果时明确写 `original/legacy Toolathlon`，固定代码 commit 和任务快照；
- 不把两代分数放在同一列中当作可直接比较的结果；
- 不用旧轨迹训练后再把 Verified 当成干净测试。两版仍是同一组 108 个任务，
  任务文本和解法存在直接重叠。

Verified 于 2026 年 6 月 30 日发布，晚于 ICLR 2026 接收和会议，目前没有一篇
单独介绍 Verified 的新会议或期刊论文。它是同一个 Toolathlon 项目的会后修订版本，
继承并继续引用同一篇 ICLR 论文，而不是另一个独立发表的 benchmark。

### 3.8 为什么本实验不拿官方轨迹训练

这里需要区分两个官方轨迹仓库。

新的
[Toolathlon-Verified_Trajectories](https://huggingface.co/datasets/hkust-nlp/Toolathlon-Verified_Trajectories/blob/main/README.md)
有明确的访问政策：该仓库中的文件、文本以及改写版本不能用于预训练、SFT、RL、蒸馏、生成训练数据，也不能做成专门回答 benchmark 的检索库。这个限制没有给学术研究设置例外。

旧的
[Toolathlon-Trajectories](https://huggingface.co/datasets/hkust-nlp/Toolathlon-Trajectories/blob/main/README.md)
目前标注为 CC-BY-4.0，公开数据卡中没有“仅限科研”或“禁止训练”的条款。CC BY 4.0 允许复制和改编，也允许商业用途，条件是正确署名、链接许可证并说明做过哪些修改。因此，就公开数据卡给出的版权许可而言，旧轨迹可以用于科研训练，也不只限于非商业训练。不能笼统地说“Toolathlon 所有轨迹都在许可上禁止训练”。

这里的“可以训练”只回答许可证问题，不代表训练后的模型还能参加干净的 Toolathlon 测试。旧库约有 108 道题，记录中包含 `task_str`、完整消息、工具调用和执行结果。[旧数据预览](https://huggingface.co/datasets/hkust-nlp/Toolathlon-Trajectories/viewer)可以确认 `find-alita-paper`、`privacy-desensitization`、`gdp-cr5-analysis`、`git-repo` 和 `inter-final-performance-analysis` 等任务 ID；它们仍然存在于[当前 `tasks/finalpool`](https://github.com/hkust-nlp/Toolathlon/tree/main/tasks/finalpool)，其中 `gdp-cr5-analysis` 等任务文本可以直接对上。官方把当前版本称为经过 prompt、ground truth 和 evaluator 复核后的 Toolathlon-Verified 最终版，而不是一套与旧版无关的新测试集。

官方 Verified 发布说明已经明确：两版使用同一组 108 个任务，没有增加、删除或重命名
任务。因此，用旧轨迹训练 TapMem，再测试当前 Toolathlon-Verified，会产生直接的任务级
测试污染，不能作为“未见任务”或官方干净 benchmark 结果。

目前没有找到官方声明说新 Verified 仓库的禁训政策会追溯覆盖旧仓库，旧数据卡也仍然是 CC-BY-4.0。因此不能把“评测会污染”误写成“官方明令禁止训练旧库”。不过，删掉题目文本、改写问题或只抽取工具调用，都不能消除已经看过测试解法这一事实。数据中还含模型输出和外部工具返回内容，商业使用前应另外检查第三方权利。这不是法律意见。

所以本文的实验原则是：

- 不使用任何 Toolathlon 测试题轨迹训练 TapMem；
- 新的 Verified 轨迹遵守其明确的禁训政策；
- 旧轨迹可以用于评测完成后的错误分析；
- 若用旧轨迹训练，只在 MCP-Atlas 或其他经过任务、实体和文本去重的外部 benchmark 上评测；
- 若用旧轨迹构造 Toolathlon 内部的 train/test split，必须按 `task_name` 分组切分，并将结果称为 Toolathlon-derived 自定义实验，不能报告成官方 Toolathlon 成绩；
- TapMem 的训练练习由我们根据工具接口另外编写并执行验证。

自定义 split 不能按轨迹行随机切。旧库中同一道题有多个模型和多次运行，同一 `task_name` 的所有轨迹必须全部进入训练侧或测试侧；最好再按语义模板分组，避免只替换实体名称的近重复任务跨到两侧。训练前还要把旧轨迹的工具名和 schema 与当前运行环境重新对齐，并重放能够重放的调用。

### 3.9 需要什么环境

完整自托管需要 Linux、Docker/Podman、Python 3.12、Node.js 22、网络访问，以及部分服务的测试账号或 API key。Google、GitHub、Notion、Snowflake、Canvas、邮箱、WooCommerce 和 Kubernetes 等任务还需要额外配置。

有两种比较省事的接入方式：

- 使用官方评测服务，把模型请求转到我们的 TapMem 模型服务；
- 使用官方 decoupled runner：Toolathlon 容器负责环境和评分，我们只在宿主机运行 TapMem。

官方单题最多 100 轮，超时上限 5,400 秒。完整环境的准备成本远大于模型训练本身。

## 4. MCP-Atlas

### 4.1 一道题的数据结构

MCP-Atlas 完整 benchmark 有 1,000 道题：

- 500 public；
- 500 private。

外部研究者只能独立运行 public 500。Hugging Face 页面把这 500 行显示为 `train`，只是文件的 split 名，不代表官方提供了训练集。

公开 Parquet 每行有五个字段：

| 字段 | 内容 | 测试时给不给模型 |
|---|---|---|
| `TASK` | 题目 ID | 只用于记录结果 |
| `ENABLED_TOOLS` | 这道题提供的全部候选工具，包括干扰工具 | 给 |
| `PROMPT` | 用户问题 | 给 |
| `GTFA_CLAIMS` | 最终答案必须覆盖的标准要点 | 不给，只用于评分 |
| `TRAJECTORY` | 专家完成这道题的参考执行过程 | 不给，只用于事后分析 |

工具的完整参数格式不在 Parquet 行中，需要从固定版本的 MCP 容器读取。

公开数据覆盖 36 个 MCP server、220 个工具。论文统计中：

- 每题提供 6–37 个候选工具，平均 15.2 个；
- 真正需要的工具平均约 4.1 个；
- 98.6% 的题需要跨两个或更多 server；
- 专家参考轨迹平均约 9.8 次工具调用。

### 4.2 一道题怎么运行

一次测试大致是：

```text
启动固定版本的 MCP 容器
→ 给模型问题和候选工具
→ 模型选择工具并生成参数
→ 容器执行工具并返回结果
→ 模型继续检索、计算或读取其他数据源
→ 模型整理证据并输出最终答案
→ 官方 judge 检查答案要点
```

MCP-Atlas 主要考察：

- 能否从多个 server 中选对工具；
- 参数是否正确；
- 能否把前一步结果用于下一步；
- 能否把多次工具结果整理成完整答案。

### 4.3 官方指标

每道题有若干条标准答案要点。judge 对每条要点评分：

- 完全满足：1；
- 部分满足：0.5；
- 未满足：0。

所有要点取平均，得到这道题的“答案要点覆盖率”。覆盖率达到 0.75 就算通过，主指标是 `Pass Rate@0.75`。

参考轨迹不参与最终判分。模型没有走专家路径，只要答案有充分证据并覆盖要点，仍然可以得满分。

### 4.4 需要什么环境

官方实现包括：

- Docker 中的 MCP 环境；
- TypeScript agent harness；
- Python 运行脚本和评分脚本；
- 一个提供 TapMem 推理的模型 API；
- 一个负责答案判分的 judge 模型 API。

主机至少需要 Docker、Python 3.10、Node/npm 和 8–10 GB Docker 内存。本地运行 TapMem 还需要 GPU。

官方目前列出 20 个不需要 API key 的 server，适合先做小规模实验。但“不需要 key”不等于完全离线，arXiv、PubMed、Wikipedia、天气等工具仍然依赖网络。

论文报告不同模型平均每题约 27–194 秒。public 500 的完整运行时间会随模型速度、并发、限流和重试明显变化。

## 5. 不用 benchmark 数据，训练数据从哪里来

### 5.1 先把实验边界说清楚

可以使用相同的工具，不能使用相同的测试经验。

这项实验最准确的名字是“见过工具、没见过任务”：模型在训练阶段学过这些 MCP 工具的参数和常见操作流程，但没有看过 Toolathlon 或 MCP-Atlas 的测试题、参考轨迹、答案和初始状态。这样才能检验 TapMem 能不能把独立经历压进 tokenized procedural memory，再迁移到新任务。

| 可以作为训练依据 | 不能进入训练流程 |
|---|---|
| server 名称、开源仓库、安装命令和固定版本 | Toolathlon 的 `task.md` 和 MCP-Atlas 的 `PROMPT` |
| 运行时公开的工具说明和 `inputSchema` | 官方参考轨迹和模型运行 dump |
| 上游项目的公开文档和示例，前提是许可证允许 | `GTFA_CLAIMS`、ground-truth workspace 和评分程序 |
| 我们自己创建的目录、数据库、账号和任务 | benchmark 的 initial workspace、preprocess 和 data exports |
| 在干净环境中重新执行得到的 observation | 进入评测容器后利用其现成状态“另造题目” |

训练程序最好与 benchmark 文件物理隔离。它只读一份工具清单，不能挂载 Toolathlon 的 `tasks/finalpool`，也不能访问 MCP-Atlas 的 Parquet、`data_exports` 或容器内置样例文件。正式评测时再由另一套进程读取测试题，并且禁止在线更新 memory。

工具范围也有两种定法：

- 最严格、最推荐：训练程序只读全局 server 配置和 `tools/list`，对事先选定的 server 族生成练习；它不知道哪些工具会出现在某一道测试题里。
- 更省数据：先读每道题的 `task_config` 或 `ENABLED_TOOLS`，只训练这些候选工具。这没有使用答案，但属于“已知测试工具集合”的 transductive 设置，论文中必须说明。

为了避免审稿人把“根据测试菜单定制训练”也看作泄漏，主实验优先采用第一种。每道题的候选菜单只在冻结模型正式测试时出现。

### 5.2 去哪里找到它们实际使用的工具

两套 benchmark 都公开了 server 配置，所以不需要从测试轨迹中猜工具。

#### Toolathlon

[Toolathlon 的 MCP 配置目录](https://github.com/hkust-nlp/Toolathlon/tree/main/configs/mcp_servers)
给出了每个 server 的上游来源、启动命令、参数和所需环境变量。例如：

| server | 怎样在独立训练环境中启动 | 训练状态从哪里来 |
|---|---|---|
| filesystem | 启动官方 filesystem MCP，并把根目录指向新建临时目录 | 随机生成文件和子目录 |
| git | 启动配置指向的 Git MCP，并指向新建仓库 | 随机生成 commit、branch 和修改 |
| memory | 启动 memory MCP，并使用新的 memory 文件 | 随机生成虚构实体和关系 |
| terminal | 启动 CLI MCP，只允许访问隔离目录和白名单命令 | 自己生成脚本和输入文件 |
| Excel、Word、PPT、PDF | 按官方配置启动对应 server | 自己生成表格、文档和模板 |

标准调用流程就是：

```text
按配置启动 server
→ MCP initialize
→ tools/list 读取工具名、说明和参数格式
→ tools/call 执行我们自己的任务
```

训练数据采集器可以直接使用官方 MCP Python SDK。应固定稳定的 v1 SDK 和每个 server 的 commit、包版本或容器摘要；如果原配置使用没有版本号的 `npx -y`，还要记录本次真正安装下来的版本，不能以后再次运行时悄悄升级。

[Toolathlon-GYM](https://github.com/eigent-ai/toolathlon_gym)
是一个更省安装工作的选择。它提供 25 个与 Toolathlon 接口体系对应的本地 server，状态存在本地 PostgreSQL 中，不需要真实邮箱、网店、Canvas 或 Snowflake 账号。逐项对照当前配置，这 25 个 server 名称都能在 Toolathlon 的 32 个 server 中找到；将本地 Git server 补进去后，可以低成本覆盖 26 个。

```text
12306、arxiv-latex、arxiv_local、canvas、emails、excel、
filesystem、google_calendar、google_forms、google_sheet、howtocook、
memory、notion、fetch、pdf-tools、playwright_with_chunk、pptx、
scholarly、snowflake、terminal、woocommerce、word、yahoo-finance、
youtube、youtube-transcript
```

它暂时没有覆盖 Toolathlon 的 GitHub、Hugging Face、W&B、Kubernetes、Google BigQuery、Google Maps 和 git。git 可以直接在本地补；其余工具按是否有账号和部署时间决定要不要进入第一轮实验。

“名称覆盖”不等于“二进制和行为完全相同”。先比较两边 `tools/list` 的 schema hash；相同的可以直接采集训练数据，不同的只能借 Gym 生成任务模板，最终仍要在 Toolathlon 配置指向的 server 上重放。

这里推荐只借用 Toolathlon-GYM 的 server、mock 服务和启动脚本，不读取它的 503 个任务、初始文件和答案。数据库类任务另建训练数据库，并填入随机生成的虚构记录，不依赖 `tasks/finalpool` 的 preprocess。虽然该项目是 Apache-2.0，任务也不是 Toolathlon-Verified 的 108 道题，但它明确说明自己沿用了 Toolathlon 的数据管线和任务格式。只使用环境、重新生成任务，给审稿人的数据隔离解释最干净。

#### MCP-Atlas

[MCP-Atlas 的 server 模板](https://github.com/scaleapi/mcp-atlas/blob/main/services/agent-environment/src/agent_environment/mcp_server_template.json)
列出了 36 个开源且固定版本的 server。调研时官方镜像是
`ghcr.io/scaleapi/mcp-atlas:1.2.7`，20 个不需要 key 的 server 会默认启动。可以直接从运行环境读取 enabled servers 和工具 schema，也可以根据模板把目标 server 单独装进一个干净的训练镜像。

20 个 no-key server 是：

```text
arxiv、calculator、cli-mcp-server、clinicaltrialsgov-mcp-server、
context7、ddg-search、desktop-commander、fetch、filesystem、git、
mcp-code-executor、mcp-server-code-runner、memory、met-museum、
open-library、osm-mcp-server、pubmed、weather、whois、wikipedia
```

其中第一轮最适合的是 filesystem、git、memory、calculator、CLI、code executor、code runner 和 desktop commander，因为它们可以在本地生成状态。arXiv、PubMed、Wikipedia、Open Library、天气和搜索等也不需要账号，但依赖公网，返回结果会随时间变化。训练记录必须保存抓取日期和原始 observation。

不要直接把官方镜像里的 `/data/Barber Shop.csv` 等样例文件拿来生成训练记录，也不要导入 benchmark 的 `data_exports`。最稳妥的方式是复用相同的 server 包和版本，另建一个空训练镜像；如果为了省事复用官方镜像，至少覆盖 `/data`、memory 和 Git 目录，使它们只包含我们随机生成的内容。

### 5.3 有没有现成的外部训练数据

有，但没有一个外部数据集能直接覆盖两套 benchmark 的全部工具。可用资源按推荐顺序如下。

| 来源 | 能提供什么 | 推荐用法 | 主要问题 |
|---|---|---|---|
| [Toucan-1.5M](https://huggingface.co/datasets/Agent-Ark/Toucan-1.5M) | Apache-2.0；约 165 万条真实 MCP 执行轨迹，metadata 中有 server 仓库地址和许可证 | 先下载较小的 SFT 子集，按仓库地址、工具名和 schema 筛选，再在目标版本上重放 | 社区 server 会失效；数据卡只过滤“所有调用都失败”的轨迹，不能直接相信每一步 |
| [MCPToolBench++](https://github.com/mcp-tool-bench/MCPToolBenchPP) | MIT；包含独立任务、完整 schema 和标准调用序列；明确覆盖官方 filesystem 和 Google Maps server | 把任务和调用当种子，在目标 server 版本和全新状态中重跑 | 许多 observation 不完整，启动配置常未锁版本 |
| [MCP-Universe](https://github.com/SalesforceAIResearch/MCP-Universe) | Apache-2.0；可启动、编排和采集多种 MCP 工具 | 借用其数据生成和轨迹采集框架，替换为 benchmark 的固定 server 版本 | 它不是可直接用于全部目标工具的现成轨迹集 |
| [MCPMark](https://github.com/eval-sys/mcpmark) | Apache-2.0；有 Filesystem、Notion、GitHub、Postgres 等独立任务和 verifier | 借任务模板和 verifier 思路，重新生成状态并重跑 | Notion、GitHub 等实现和版本未必与目标一致 |
| MCP-Flow | 有较多 MCP 调用样本 | 暂不使用 | 仓库和数据页没有找到明确的数据集许可证 |
| ToolACE、xLAM、ToolBench | 大量普通 function-calling 数据 | 只做通用格式预热 | 不是目标 MCP 的原始接口，不能证明学到了这些工具的 procedural memory |

Toucan 最值得先查。它的 metadata 中包含 `repository_url`，已经可以找到与目标 benchmark 同源的 server，例如 Toolathlon 使用的 Office Word server 和 MCP-Atlas 使用的 Met Museum server。但“同源”仍然不保证版本和 schema 相同，必须重放。

现成数据的使用规则是：

1. 先在目标运行环境执行 `tools/list`。
2. 用“上游仓库地址 + 固定版本 + 原始工具名 + 规范化后的 `inputSchema`”连接外部数据。
3. 只有完全匹配的调用才能直接作为候选；只是显示名称相似的不算。
4. 所有候选调用都在干净的目标 server 上重新执行。
5. 只保存实际成功、状态变化正确且 verifier 通过的记录。

因此，外部数据的价值主要是省掉“想任务和参数”的时间，最终 observation 仍然应该由我们自己的固定环境产生。

数据集许可证只解决“这份数据能不能用”，不自动覆盖每个上游 server、远程 API 和返回内容的许可证。需要对进入最终训练集的 server 逐一记录许可证；涉及商业 SaaS、网页内容或账号数据时，还要遵守对应服务条款。短实验优先用本地合成状态，可以避开大部分问题。

### 5.4 怎样判断是不是同一个工具

工具叫同一个名字并不够。两个 `fetch` 都能访问网页，参数字段可能完全不同；两个 Notion server 即使来自同一个仓库，不同版本也可能改过分页和返回格式。

每次实验先生成一份 `tool_manifest.jsonl`，至少记录：

```json
{
  "benchmark": "toolathlon",
  "server": "filesystem",
  "upstream_repo": "...",
  "server_version": "...",
  "tool_name": "read_text_file",
  "description": "...",
  "input_schema": {},
  "schema_hash": "...",
  "runtime_image": "...",
  "needs_key": false
}
```

`schema_hash` 对排序后的原始工具名和 JSON `inputSchema` 计算。只有下面四项都通过，两个环境才能共用同一个 memory token：

1. 来自同一个上游 server；
2. server 版本相同；
3. 原始工具名和 `inputSchema` 完全相同；
4. 一组无副作用的 smoke call 在两个环境中都成功，返回语义一致。

如果只有 benchmark 加的命名空间不同，例如
`filesystem_read_text_file` 和 `read_text_file`，可以由推理转换层去掉前缀。只要参数 schema 或行为不同，就分配两个 memory token，不能强行合并。

还要注意 MCP-Atlas 的版本边界：公开 Parquet 是 220 个工具，而当前官方仓库列出的运行环境已经有 307 个。应逐题确认 `ENABLED_TOOLS` 都能在本次冻结的镜像中找到。缺失时换兼容镜像或删除整道题并公布规则，不能只从候选菜单中删掉缺失工具。

### 5.5 找不到现成数据时，怎样自己造

最稳妥的办法不是“让大模型先随便写一道题，再看能不能跑通”，而是反过来做：

```text
先随机生成一个全新环境
→ 从 tools/list 采样合法工具和参数
→ 真正执行一次或多次调用
→ 组合已经成功的调用形成依赖链
→ 最后把这条已知可执行的过程改写成自然语言任务
```

例如：

- filesystem：生成随机目录和文件，练习查找、读取、移动和改写，用文件 hash 检查；
- git：新建临时仓库和若干 commit，练习 status、diff、log 和 branch，用 commit ID 检查；
- memory：创建虚构人物和组织关系，练习写入和检索，用导出的 graph JSON 检查；
- Office：生成随机 CSV，再转成 Excel、Word 或 PDF，用解析器检查 sheet、单元格和段落；
- 数据库和 SaaS：在专用测试账号或本地 mock 数据库中创建虚构对象，完成查询、更新和跨工具同步，再 read-after-write；
- arXiv、Wikipedia 等只读工具：采样与测试题无关的查询，保存原始返回并检查必需字段非空。

每条原始 episode 建议保存：

```json
{
  "source": "self_generated",
  "seed": 1234,
  "instruction": "一个与 benchmark 无关的新任务",
  "candidate_tools": [],
  "steps": [
    {
      "tool_name": "read_text_file",
      "arguments": {},
      "observation": "...",
      "verified": true
    }
  ],
  "state_before": {},
  "state_after": {},
  "verifier": {"name": "...", "passed": true},
  "tool_manifest_hash": "..."
}
```

只保留满足以下条件的记录：

- 参数通过 JSON schema 检查；
- 每一次工具调用都真实发生；
- observation 不是空的伪结果；
- 最终状态通过程序检查；
- 不含密钥、个人信息和 benchmark 实体；
- 每条 episode 结束后环境可以 reset；
- 与其他训练任务去重。

实用起点是每个工具做 20–50 个成功练习，混合单步任务和 2–4 步依赖任务。这个数量不是理论最优，只是当前仓库 `max_samples_per_tool=50` 下便于快速比较的规模。

### 5.6 怎样转成当前 TapMem 的训练格式

当前 `compositional/dataset.py` 接受：

```json
{
  "user_input": "任务和已有上下文",
  "tools": ["下一步工具"],
  "function_calls": ["{\"arg\": \"value\"}"]
}
```

最少改代码的做法是把一条真实轨迹拆成多个“看到当前状态后，预测下一步”的样本。假设轨迹是 A、B、C：

```text
样本 1：任务 → A 的参数
样本 2：任务 + A 的调用和 observation → B 的参数
样本 3：任务 + A/B 的调用和 observation → C 的参数
```

这样不改变 tokenized procedural memory，也不改变 TokMem、EOC 或 TCRA，只增加一个 episode-to-prefix 转换脚本。它足以先验证工具选择和参数生成。

更完整的做法是给 dataloader 增加“工具 observation span”：用户文本和 observation 只作为上下文，不计算 loss；每个 assistant span 仍然只监督
`memory token + JSON 参数 + EOC`。这样 TapMem 能在训练时真正看到前一步返回结果，再决定下一步工具。这个改动只涉及数据表示和 loss mask，不增加 planner、LoRA 或新的模型模块，更适合用来支持长流程结论。

如果 rebuttal 时间很紧，可以先用 prefix 样本完成第一轮结果；论文中应把它叫作“逐步工具决策设置”。如果要声称 TapMem 改善了 observation 驱动的长流程执行，最好补上 observation-aware loader。

### 5.7 两套 benchmark 的公共工具只用于省成本

Toolathlon 和 MCP-Atlas 确实有同源工具，最实用的是 filesystem、git、memory、arXiv、CLI、Google Maps 和 Notion。前五类中的大部分不需要账号，适合共用任务模板。

但公共工具不是训练方案的前提。正式做某个 benchmark 时，应为选定题目提供的全部候选工具准备 memory token 和独立练习。只训练公共工具会遇到一个问题：两套 benchmark 都会在候选菜单中加入许多其他干扰工具。

因此分两层准备：

1. 公共层：先做 filesystem、git、memory 和 CLI，检查数据采集、token 映射和 observation 回传。
2. benchmark 层：固定正式子集后，再为该子集完整候选菜单中的其余工具生成练习。

如果只保留公共工具、删掉原题的其他候选工具，结果只能叫“受控工具子集”，不能叫官方 benchmark 子集成绩。若保留原始候选菜单和官方评分器，才能报告对应 benchmark 子集结果。

### 5.8 三种方法学什么

| 方法 | 训练目标 |
|---|---|
| TokMem | 生成“工具 memory token + 工具参数” |
| EOC-only | 在 TokMem 基础上，用 EOC 标记一次工具调用结束 |
| TapMem | 使用 EOC，并用 TCRA 调整下一次应该选择哪个 memory token |

本仓库的定义保持不变：

- TokMem：不开 adaptation，不用 EOC，不用 logit bias；
- TapMem：不开 adaptation，使用 EOC 和 logit bias；TCRA 就是 logit-bias 机制。

TokMem、EOC-only 和 TapMem 必须使用同一批独立训练 episode。不增加 LoRA、额外 planner、专用检索器或 ground-truth tool forcing，否则无法判断收益是不是来自 TapMem。

## 6. TapMem 怎么在 benchmark 中推理

### 6.1 当前代码还缺什么

当前 `compositional/` 实验只让模型一次性生成工具和参数，不会真的执行工具，也不会把工具结果放回模型。因此不能直接得到 Toolathlon 成功率或 MCP-Atlas 答案要点覆盖率。

需要增加一个很薄的格式转换层：

```text
TapMem 输出 memory token 和 JSON 参数
→ 转成官方环境接受的 tool call
→ 官方环境真正执行
→ 把 observation 或错误放回上下文
→ TapMem 再决定下一步
```

这个转换层不负责规划，也不修改模型选择，只负责把两边的格式接起来。

### 6.2 每轮怎么生成

为了尽量保持现有方法，每一轮只执行一个工具：

```text
当前题目、工具列表、历史和最新 observation
→ memory token
→ JSON 参数
→ EOC（只有 EOC-only/TapMem）
→ 停止本轮生成并执行工具
```

TapMem 生成第一个 EOC 后必须立即停止，不能在工具尚未执行时继续生成下一个 memory token。工具结果返回后，再开始下一轮。

这种接法验证的是：模型收到工具结果后，下一步能不能选对。它不能单独证明 EOC 在连续生成过程中的转移作用，论文里应如实说明。

### 6.3 每道题只能选择当前提供的工具

每道题都只允许选择 `ENABLED_TOOLS` 或 `task_config` 给出的工具。未提供的 memory token 在这一题中不可选，但普通文本仍然可以正常生成。

TokMem、EOC-only 和 TapMem 必须使用完全相同的候选工具列表。不能只保留正确工具、删除干扰工具。

### 6.4 结束和评分

- MCP-Atlas：模型停止调用工具后输出最终文字答案，交给官方答案评分脚本。
- Toolathlon：模型完成所有操作后调用 `claim_done`，交给该题自己的最终状态检查程序。

参考轨迹只用于测试完成后的错误分析，不能拿来强制模型走固定路径。

## 7. 全集为什么有的能跑、有的暂时不能

当前模型只有 248 个 reserved token 位置。TapMem 还需要一个 EOC，所以公平比较时最多放 247 个工具。

### MCP-Atlas

public 500 的全部候选工具去重后是 220 个：

```text
220 tools + 1 EOC ≤ 248
```

因此容量上可以直接建立一套覆盖 public 500 的固定工具 memory tokens。

### Toolathlon

Toolathlon 有 604 个 MCP 工具，还要加实际使用的本地工具：

```text
全部工具 + 1 EOC > 248
```

有两种选择：

1. 先选一个工具并集不超过 247 的固定任务子集；
2. 给 tokenizer 增加更多 synthetic memory tokens，并 resize embedding。

第二种只是扩大 memory token 数量，backbone 仍然冻结，TokMem、EOC-only、TapMem 使用同一套扩容代码，因此不改变方法核心。

不能在不同题目中临时把同一个 memory token 改成不同工具，再把结果合成一个“Toolathlon 全集成绩”。那样 memory token 的含义不再稳定。

## 8. 建议跑哪些子集

所有子集都必须在看到 TapMem 结果前固定，并公布 task ID。这里的子集是本项目自定义实验，不是官方 split。

### 8.1 MCP-Atlas

| 阶段 | 规模 | 用途 | 结果怎么称呼 |
|---|---:|---|---|
| 链路检查 | 2–5 题 | 检查参数、工具执行、答案和评分脚本 | 工程检查，不是结果 |
| 无密钥小实验 | 24 题 | 最快比较 TokMem 与 TapMem | MCP-Atlas public no-key 子集 |
| 扩展实验 | 50 题 | 增加工具和 server 覆盖 | MCP-Atlas public 50 题子集 |
| public 全集 | 500 题 | 最强公开结果 | MCP-Atlas public 500，不是完整 1,000 |

对本地固定版本的 public 数据做只读检查后：

- 有 30 题的全部候选工具都来自 20 个 no-key server；
- 这 30 题的候选工具去重后共有 89 个；
- 可以在其中事先固定 24 题做第一轮正式比较。

筛选时必须检查一题的全部候选工具，包括干扰工具。不能因为某个干扰工具需要 key，就把它从菜单中删掉。

如果要扩到 50 题，可以再配置 MongoDB、Twelve Data 和一个测试 GitHub 账号，增加数据库、金融和远程代码工具。

### 8.2 Toolathlon

第一步先跑两题：

- `find-alita-paper`
- `git-bug-hunt`

它们只用于确认：

- TapMem 能收到题目和工具；
- 参数能被官方环境执行；
- observation 能返回；
- `claim_done` 和 evaluator 能工作。

低部署成本方案是 `local-office-10`。这 10 题只依赖 filesystem、terminal、excel 和 pdf-tools 这组 MCP server：

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

它适合验证文件、表格和文档类的有状态操作，但不能代表整个 Toolathlon。

如果希望覆盖七类任务，可以每类固定一道：

| 类别 | 任务 |
|---|---|
| Research | `find-alita-paper` |
| Campus | `canvas-homework-grader-python` |
| Finance | `ab-testing` |
| Tech | `git-bug-hunt` |
| Business/Office | `invoice-org` |
| Daily | `cooking-guidance` |
| E-commerce | `inventory-sync` |

这组任务覆盖更广，但需要 Canvas、email、Google Cloud 和 WooCommerce 等环境。运行前必须检查七题全部候选工具去重后是否超过 247。

后续可扩成：

- 14 题：每类 2 题；
- 21 题：每类 3 题；
- 108 题：先完成 memory-token 扩容，再跑完整评测。

## 9. 最快而且可信的实验路线

### 第一步：当天能完成的工具和数据检查

先不下载两套 benchmark 的轨迹，只做下面四件事：

1. 从 Toolathlon 配置和 MCP-Atlas server 模板生成 `tool_manifest.jsonl`。
2. 启动 filesystem、git、memory、CLI 和一到两个 Office server。
3. 对 `tools/list` 的结果计算 schema hash。
4. 从 Toucan 和 MCPToolBench++ 中筛选同源调用；没有现成数据的工具各自动生成 20 个练习并重跑。

先人工检查 20–50 条 episode，确认没有 benchmark 文件、实体或状态混进来。随后用 2–5 道题做工程检查：

- memory token 能否正确转成工具名；
- JSON 参数是否有效；
- 工具是否真的执行；
- observation 是否进入下一轮；
- 最终答案或状态是否能被官方评分程序读取。

这一步只证明链路通了，不能证明 TapMem 有效。

### 第二步：最短的正式比较

MCP-Atlas 和 Toolathlon 分别训练自己的 tool-token 表和 checkpoint，不要求一个 checkpoint 同时容纳两个 benchmark 的所有工具。

MCP-Atlas：

1. 从 30 个完整候选菜单都不需要 key 的题目中，事先选 8–12 道，使候选工具并集尽量小。
2. 仍然保留每道题原始的全部 `ENABLED_TOOLS`，包括干扰工具。
3. 每个候选工具收集 20–50 个独立成功练习；优先复用 Toucan 和 MCPToolBench++ 的任务种子并重放。
4. TokMem、EOC-only 和 TapMem 用同一批数据各训练一个随机种子。
5. 比较官方答案要点覆盖率、Pass Rate@0.75、选错工具率和参数错误率。

Toolathlon：

1. 先做 `local-office-10`，因为 filesystem、terminal、Excel 和 PDF 工具都能在本地独立造数据。
2. 用 Toolathlon-GYM 或官方 server 配置启动工具，但训练阶段只加载随机生成的目录、CSV、Excel 和 PDF。
3. 保留原任务的工具菜单、初始状态和官方 evaluator，只在正式测试阶段让冻结模型看到它们。
4. 每题先运行一次比较趋势；有正向结果后再按官方设置运行三次。

这两个结果可以共同说明 TapMem 在一个前沿检索型 benchmark 和一个前沿有状态 benchmark 上有效，但都必须写成事先固定的子集结果。

### 第三步：补统计稳定性

如果最短比较中 TapMem 优于 TokMem：

- 扩大到 MCP-Atlas no-key 24 题；
- 使用 3 个独立训练随机种子；
- Toolathlon 每题运行三次；
- 按同一道题做配对比较并报告置信区间；
- 检查收益是否来自更好的工具选择和 observation 后的下一步决策。

然后再考虑 MCP-Atlas 50 题和 Toolathlon 七类覆盖。

### 如果连训练数据生成时间也不够

可以再缩小为“同工具、受控菜单”实验：

- 用 MCPToolBench++ 的 Filesystem 样本作种子，重放 50–100 条；
- 自行生成 Git、memory 和 CLI 各 50 条；
- 从两套 benchmark 中选择确实需要这些工具的任务；
- 只提供已经训练的工具和等量同类干扰工具。

这种设置仍能比较 TokMem 和 TapMem 的 memory token 路由能力，但它改了原始候选菜单，不能报告成官方 benchmark 分数。论文中应称为
“Toolathlon/MCP-Atlas 任务上的受控工具子集”。它适合 1–2 天内验证方法是否有信号，不能替代后续官方子集实验。

### 三档实验规模

| 方案 | 实验内容 | 适合什么情况 |
|---|---|---|
| 最低配置 | MCP-Atlas 8–12 题 + Toolathlon `local-office-10`；TokMem vs. TapMem；1 个训练随机种子 | 时间非常紧，只判断有没有正向趋势 |
| 推荐配置 | MCP-Atlas 24/50 题；TokMem、EOC-only、TapMem；3 个训练随机种子；再加 Toolathlon 7/10 题 | rebuttal 主实验 |
| 完整配置 | MCP-Atlas public 500；扩容后的 Toolathlon 108 | 算力、账号和时间充足 |

## 10. 怎样才算验证了 TapMem

论文主结论必须来自官方最终分数，而不是只看工具预测准确率。

比较时至少保证：

- TapMem 和 TokMem 使用同一个基础模型；
- 使用同一批自行构造的训练练习；
- 使用相同候选工具、上下文长度、解码参数和任务顺序；
- 不开启 adaptation、LoRA 或 ground-truth tool forcing；
- 每次状态任务从干净环境开始；
- 超时和错误任务也保留在结果中，不能被评分脚本静默丢掉。

可以支持“TapMem 有效”的结果应当同时表现为：

1. 官方最终分数高于 TokMem；
2. 选错工具或格式错误没有增加；
3. 多个训练随机种子下方向基本一致；
4. 收益不是因为 TapMem 看到了更少的干扰工具或更长的上下文。

如果 TapMem 的工具选择更好，但 Toolathlon 成功率或 MCP-Atlas 答案覆盖率没有提高，只能说它改善了工具路由，不能说它提高了完整任务成功率。

## 11. 论文中应该怎样表述

可以写：

> TokMem 和 TapMem 先在自行构造、实际执行验证过的工具练习上训练。训练练习与测试题的任务、实体和环境状态均不重合。训练结束后模型完全冻结，再在事先固定的 Toolathlon 或 MCP-Atlas 题目上运行，并使用官方评分程序比较结果。

必须明确：

- MCP-Atlas 的 HF `train` 不是官方训练集；
- Toolathlon 官方轨迹没有用于训练；
- 训练数据来自外部许可数据的严格筛选与重放，或来自干净环境中的自行生成；
- 训练程序是否预先知道正式子集的候选工具；若知道，应明确称为 known-tool/transductive 设置；
- 24、50、7、10、14 题都是自定义子集；
- MCP-Atlas public 500 不是完整 1,000；
- 这个实验验证“相同工具上的新任务”，不是“从未见过的新工具”；
- 子集结果不能写成 benchmark 全集成绩；
- 只改善离线工具序列不能代替官方端到端分数。

## 12. 主要来源

Toolathlon：

- [论文](https://arxiv.org/abs/2510.25726)
- [ICLR 2026 正式论文页面](https://iclr.cc/virtual/2026/poster/10006492)
- [OpenReview 会议论文](https://openreview.net/forum?id=z53s5p0qhf)
- [官方仓库](https://github.com/hkust-nlp/Toolathlon)
- [Toolathlon-Verified 发布说明与新旧版修改统计](https://toolathlon.xyz/docs/blog/toolathlon-verified)
- [当前 Verified 榜单和旧版归档榜单](https://toolathlon.xyz/docs/leaderboard)
- [公开 MCP server 清单](https://toolathlon.xyz/docs/selected)
- [MCP server 运行配置](https://github.com/hkust-nlp/Toolathlon/tree/main/configs/mcp_servers)
- [任务数据结构](https://toolathlon.xyz/docs/dataset)
- [108 道题的官方完整索引](https://toolathlon.xyz/llms.txt)
- [任务示例：Find Alita Paper](https://toolathlon.xyz/docs/tasks/aca/10)
- [Find Alita Paper 的 Verified evaluator](https://github.com/hkust-nlp/Toolathlon/blob/d57361c0f1582cf9a0675c0753315bb6b004bd0e/tasks/finalpool/find-alita-paper/evaluation/main.py)
- [任务示例：Canvas Homework Grader Python](https://toolathlon.xyz/docs/tasks/campus/306)
- [任务示例：Inter Final Performance Analysis](https://toolathlon.xyz/docs/tasks/daily/351)
- [任务示例：NVIDIA Market](https://toolathlon.xyz/docs/tasks/finance/284)
- [任务示例：Travel Expense Reimbursement](https://toolathlon.xyz/docs/tasks/office/331)
- [任务示例：Filter Low-Selling Products](https://toolathlon.xyz/docs/tasks/shopping/301)
- [任务示例：Kubernetes PR Preview Testing](https://toolathlon.xyz/docs/tasks/tech/245)
- [decoupled agent loop](https://github.com/hkust-nlp/Toolathlon/blob/main/DECOUPLED_AGENT_LOOP.md)
- [Verified 轨迹及其禁训政策](https://huggingface.co/datasets/hkust-nlp/Toolathlon-Verified_Trajectories/blob/main/README.md)
- [旧版轨迹及其 CC-BY-4.0 数据卡](https://huggingface.co/datasets/hkust-nlp/Toolathlon-Trajectories/blob/main/README.md)
- [旧版轨迹提交历史](https://huggingface.co/datasets/hkust-nlp/Toolathlon-Trajectories/commits/main)
- [2025-10-27 旧版轨迹冻结文件清单](https://huggingface.co/datasets/hkust-nlp/Toolathlon-Trajectories/tree/b53e3f808707dbe90cd9d7216a1d425b61466a26)
- [旧版轨迹当前文件列表](https://huggingface.co/datasets/hkust-nlp/Toolathlon-Trajectories/tree/main)
- [旧版轨迹数据预览](https://huggingface.co/datasets/hkust-nlp/Toolathlon-Trajectories/viewer)
- [当前 Toolathlon finalpool](https://github.com/hkust-nlp/Toolathlon/tree/main/tasks/finalpool)
- [Toolathlon-GYM 本地工具环境](https://github.com/eigent-ai/toolathlon_gym)
- [CC BY 4.0 官方说明](https://creativecommons.org/licenses/by/4.0/)

MCP-Atlas：

- [论文](https://arxiv.org/abs/2602.00933)
- [官方仓库](https://github.com/scaleapi/mcp-atlas)
- [固定版本的 MCP server 配置](https://github.com/scaleapi/mcp-atlas/blob/main/services/agent-environment/src/agent_environment/mcp_server_template.json)
- [public 数据](https://huggingface.co/datasets/ScaleAI/MCP-Atlas)
- [官方 leaderboard](https://labs.scale.com/leaderboard/mcp_atlas)

外部训练数据和生成框架：

- [Toucan-1.5M](https://huggingface.co/datasets/Agent-Ark/Toucan-1.5M)
- [Toucan 数据生成代码](https://github.com/TheAgentArk/Toucan)
- [MCPToolBench++](https://github.com/mcp-tool-bench/MCPToolBenchPP)
- [MCP-Universe](https://github.com/SalesforceAIResearch/MCP-Universe)
- [MCPMark](https://github.com/eval-sys/mcpmark)

数据采集工具：

- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- [MCP Inspector](https://github.com/modelcontextprotocol/inspector)
