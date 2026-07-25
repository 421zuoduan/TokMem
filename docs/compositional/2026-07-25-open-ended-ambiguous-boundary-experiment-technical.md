# TapMem 开放式模糊边界实验：技术复现附录

> 主要阅读入口是
> [中文方案正文](./2026-07-25-open-ended-ambiguous-boundary-experiment.md)。
> 本文保留精确公式、字段、伪代码和自动检查，供实现与复核使用。

> 调研与方法审计日期：2026-07-25
>
> 文档性质：rebuttal 实验设计；文中的数据审计是预处理审计，不是模型实验结果。
>
> 方法口径：TokMem 指无 adaptation、无 EOC、无 logit bias；TapMem 指无
> adaptation、使用 EOC 和 logit bias（TCRA）。

## 0. 结论和对旧方案的修正

主实验仍推荐使用 **NL2Bash 训练 + InterCode-Bash 官方 200 题测试**，但旧版的
“semantic change score → 枚举 1--4 atoms → embedding k-means → 固定
`K=32` → top-3/noise views”不能继续使用。它没有统一的概率目标，切分依赖聚类、
聚类又依赖切分，而且长度、聚类数、view 数都可以事后调整。

最终方案改为 **Syntax-constrained Unigram Procedure Induction**：

1. `bashlex` 只确定不会切坏 Bash 语法的顶层 atoms 和字符位置，不判断
   procedure 语义；
2. 每个 atom 用确定性的 `(incoming connector, top-level utility)` 离散
   signature 表示；
3. procedure type 是训练语料中跨 command-template group 重复出现的、任意长度的
   连续 signature 序列；
4. 用带显式停止概率的 unigram sequence model，对一条命令的**所有合法完整切分**
   做 forward-backward 和 EM；
5. 用 held-out NL2Bash dev marginal NLL 选择 memory-bank capacity，而不是声称
   聚类数就是真实 procedure 数；
6. 主模糊边界设置从
   \(q(B\mid x,J=J_{\mathrm{MAP}})\) 精确采样：procedure 数量与 MAP 相同，只改变
   boundary placement，因而 memory/EOC 数和 TCRA 监督位置不变；
7. TokMem、EOC-only、TapMem 使用完全相同的切分 schedule；测试时不运行 parser
   或 segmenter，也不给模型任何 gold boundary；
8. 最终证据只看完整 200 题的官方交互式任务成功率，参考程序的结构和人工边界标注
   仅作诊断。

关键替换关系如下。

| 旧版不充分定义 | 最终可复现定义 |
|---|---|
| 相邻表示的“变化分数” | 不再存在该分数；完整切分由 unigram generative model 的概率评分 |
| span 长度固定 1--4 | 不设任意最大长度；只要求跨 group 重复且至少一次作为 proper subspan |
| 所有 raw spans 进入 k-means | occurrence 先折叠成唯一 canonical sequence type；不做 k-means |
| 嵌套 span 不知如何取舍 | 嵌套 types 可同时在词典中，但一条完整 partition 内不能重叠 |
| 固定 `K=32` | \(K\) 是容量，在 \([K_{\min},247]\) 内由 dev NLL 的 paired one-SE rule 选择 |
| top-3 softmax / 20% 随机移边界 | exact count-conditioned forward-filtering backward-sampling |
| “TapMem 边缘化 latent boundary” | 不这样声称；TapMem 优化的是固定 posterior 下的 stochastic-view expected CE |
| “200 题都有至少 4 个顶层 atoms” | 删除；官方先从 utility 数不少于 4 的 1,000 条候选构造，released gold 的顶层 AST 并非都如此 |

三个独立审计分别检查了概率分段、真实 Bash AST/data、以及因果与审稿风险。它们的
共同结论是：必须删除神经 change score 和 k-means，并把 procedure discovery、
boundary uncertainty、训练增强和 end-to-end evaluation 四件事分开定义。

## 1. 实验究竟要证明什么

### 1.1 Primary hypothesis

主假设只有一个：

$$
H_1:\quad
\Delta_{\mathrm{amb}}
=
\mathrm{SR}(\mathrm{TapMem}_{\mathrm{count\text{-}posterior}})
-
\mathrm{SR}(\mathrm{TokMem}_{\mathrm{count\text{-}posterior}})
>0.
$$

这里的 SR 是 InterCode-Bash 完整 200 题、最多 10 次交互的官方 Success Rate。
正式支持条件是 paired task-bootstrap 95% CI 的下界大于 0，且两个预注册
data schedules 各自对三个 matched model seeds 取平均后的差值都大于 0；不再使用
“CI 跨 0 但三个 seed 同方向也算成功”这样的替代标准。这里的 CI 只针对已经
预注册的六对 checkpoints 的平均 task effect，不外推成任意初始化或任意 boundary
schedule 的总体效应。

这个假设回答：

> 当 procedure 类型不是人工 API inventory，而是从原始程序中诱导的可变长 motifs，
> 且同一训练程序在相同 procedure 数量下可以采用不同 posterior-supported 边界时，
> TapMem 是否仍比
> TokMem 有更高的最终任务成功率？

### 1.2 Secondary questions

需要另外回答：

- `EOC-only - TokMem`：收益中有多少来自显式结束标记；
- `TapMem - EOC-only`：完整 TCRA component 在该场景是否还有增量作用；
- Fixed-MAP 与 count-conditioned sampling 的差异：结果是否依赖唯一固定切分；
- data-seed sensitivity：结果是否只存在于一份偶然的 posterior sample schedule；
- singleton-only 与 learned multi-atom motif 的差异：描述 atomic reference 与
  variable-length setting 的行为差异，不作 motif 因果归因；
- single-turn 与 shared-first-turn-failure recovery：多轮收益是否与执行反馈恢复一致。

Fixed-MAP 和 sampled setting 的 interaction：

$$
I=
\bigl(\mathrm{Tap}_{sample}-\mathrm{Tok}_{sample}\bigr)
-
\bigl(\mathrm{Tap}_{MAP}-\mathrm{Tok}_{MAP}\bigr)
$$

只作 secondary effect estimate。`I` 不显著不能解释为“已经证明等价或鲁棒”。

### 1.3 能说和不能说的结论

正结果可以支持：

> TapMem 不要求人工给定的 API/procedure inventory；memory token 可以索引从训练
> 程序中诱导的 variable-length Bash motifs，并在概率化 alternative boundary
> supervision 下保持相对 TokMem 的端到端优势。

它不能单独支持：

- 自动发现了唯一或真实的人类 procedure；
- 泛化到了任意开放世界、网页、桌面或具身 agent；
- 泛化到了训练中从未出现过的 procedure type；
- TapMem 自身对 latent boundary 做了概率边缘化；
- 任意不同的执行路径都会被 InterCode 判为成功。

InterCode 与 NL2Bash 同属 Bash 域，这是一项从“预定义工具调用”到“自由程序生成和
交互执行”的桥接实验，不是跨所有领域的终局证据。

## 2. 场景、数据和本地审计

### 2.1 InterCode-Bash 是什么

[InterCode](https://papers.neurips.cc/paper_files/paper/2023/file/4b175d846fb008d540d233c188379ff9-Paper-Datasets_and_Benchmarks.pdf)
把 interactive coding 表述为带 execution feedback 的 POMDP。Bash agent 每轮输出
一条 shell action，环境执行后返回 observation；agent 输出 `submit` 或达到轮数上限后，
官方 evaluator 根据最新 stdout 和文件系统变化评分。

官方构造过程是：

1. 从 NL2Bash 中先取 1,000 条至少含 4 个 utilities 的候选；
2. 删除不支持或不适合 Linux/Docker 的命令；
3. 将原本欠具体的路径、文件和 flags 落地到四种文件系统；
4. 得到 200 个任务。

官方满分并不比较 command string exact match，而要求最终可观察结果满足 gold：
最新输出匹配、修改路径/类型正确、被修改文件的内容 hash 正确。因此等价程序可以成功，
但“行为近似却多改文件”不一定成功。

正式数据必须来自
[InterCode 官方仓库](https://github.com/princeton-nlp/intercode) 的四个
`data/nl2bash/*.json` source files；PyPI 中较小的 bundled example 只能做 smoke test。

### 2.2 本次数据审计的可复现快照

审计环境为 `tokmem` conda env、`bashlex==0.18`。读取数据时只移除记录末尾的一枚 LF，
不对 command 调用 `.strip()`，不规范化或修复。

NL2Bash raw 文件：

| 文件 | 行数 | SHA-256 |
|---|---:|---|
| `all.nl` | 12,607 | `1db0c529c350b463919624550b8f5882a97c42ad5051c7d49fbc496bc4e8b770` |
| `all.cm` | 12,607 | `3a72eaced7fa14a0938354cefc42b2dcafb2d47297102f1279086e18c3abe57e` |

本次下载的四个 InterCode source JSON 快照为：

| 文件系统 | task 数 | SHA-256 |
|---|---:|---|
| fs1 | 60 | `60f88e1aacc7ebba535093f9890c5c33203f4e5f32958e0e94fbe90ec4f01c82` |
| fs2 | 53 | `8f4ce24e535fab782fda607e37db2ae1d6c5f99993c638d1ac0a7e0b542f633e` |
| fs3 | 60 | `a2d4ec8bc7ad69a4e2fb3eb84033994cf65ee9cfb355e3e63099df67a339b2e1` |
| fs4 | 27 | `ce41b89450f87765a02a51df259ca0c1762e8249185c022adb089147e2c16200` |

逐条把 raw command 交给 `bashlex.parse`：

| 项目 | 数量 |
|---|---:|
| raw pairs | 12,607 |
| parse success | 12,466 |
| parse failure | 141 |
| success root=`command` | 8,346 |
| success root=`pipeline` | 3,929 |
| success root=`list` | 163 |
| success root=`compound` | 28 |

141 个失败由 84 个 `ParsingError`、30 个 `MatchedPairError`、27 个
`NotImplementedError` 组成。parse success 只表示 parser 接受，并不保证命令执行语义
正确；例如 Unicode typography 也可能被当作普通 word。

主训练 atomizer 只接受：

- 一个 root simple `command`；或
- parts 严格为 `command, pipe, command, ..., pipe, command` 的 flat pipeline。

在该严格口径下有 12,196 条 eligible commands，顶层 simple-command 数 \(L\) 的分布为：

| \(L\) | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 数量 | 8,346 | 2,714 | 747 | 254 | 91 | 34 | 8 | 2 |

另外 270 条 parse-success command 不在该表中：79 条 pipeline 含 compound
component、163 条 root list、28 条 root compound。它们会从 **procedure lexicon
training** 排除并记录原因，但不能从 InterCode 正式测试中删题。

released InterCode-Bash 200 题的 gold AST 审计为：

| root / component 数 | 数量 |
|---|---:|
| parse success | 200 / 200 |
| root pipeline | 123 |
| root command | 74 |
| root list | 3 |
| pipeline \(L=2\) | 46 |
| pipeline \(L=3\) | 43 |
| pipeline \(L=4\) | 26 |
| pipeline \(L=5\) | 5 |
| pipeline \(L=6\) | 2 |
| pipeline \(L=7\) | 1 |

因此，论文中的“初始候选至少 4 utilities”不能改写成“released 200 题每题至少 4 个
top-level atoms”。utility 可能位于 command substitution、`xargs` 参数或
`find -exec` 内，而且 grounding 后的 released command 结构也会变化。

这些 released-gold 统计已经参与了 strict-flat v1 的范围设计，因此 123/77 题的
结构子集是 **gold-informed but pre-inference-frozen exploratory diagnostics**，
不是独立于测试集结构的预注册验证。完整 200 题的 official SR 仍可作为 primary，
前提是 prompt、harness、模型和所有选择规则在第一次候选模型推理前冻结。

### 2.3 为什么这里的 procedure boundary 是模糊的

模糊发生在“语法安全 atom 怎样组成一个执行阶段”，不是发生在 parser 不知道哪里有
pipe。

例如：

```bash
find /testbed -type f -print0 | sort -z | xargs -r0 md5sum | md5sum
```

`bashlex` 能确定四个顶层 atoms：

```text
START:find
PIPE:sort
PIPE:xargs
PIPE:md5sum
```

但它不能确定哪一种才是唯一 procedure partition：

```text
[find + sort] [xargs + md5sum]
[find] [sort + xargs] [md5sum]
[find + sort + xargs] [md5sum]
[find] [sort] [xargs] [md5sum]
```

不同粒度都可能对应可复用的数据处理阶段。原数据没有阶段名、procedure ID 或
boundary annotation；同一用户目标还可以由另一条 Bash 程序完成，所以参考程序的
切分也不是任务本身的唯一切分。

同时，语法安全位置并不等于每个表面 utility：

```bash
cd $(which python | xargs dirname)
```

这是一个顶层 `cd` atom；内部 substitution 不提升为三个可切 atoms。

```bash
find /testbed -name '*.php' -exec chmod 755 {} \; -exec echo {} \; | wc -l
```

顶层只有 `find` 和 `wc` 两个 atoms；`chmod`、`echo` 是 `find` 参数结构中的词。

```bash
xargs -0 cat
```

顶层 utility 是 `xargs`，`cat` 是参数中的 subcommand，不是独立顶层 atom。

因此：

- `bashlex` 给出 AST kind、source position、word、pipe/operator 等**语法事实**；
- 它不直接给出 utility semantics、procedure type 或合理阶段边界；
- 本方案只让 parser 限制“哪里可以切”，再由训练语料学习“哪些相邻 atoms 值得
  作为一个复用 piece”。

### 2.4 是否需要自己构造数据

不构造新 benchmark，不改 InterCode 的 200 个任务、Docker 初态或 evaluator。
需要构造的是从既有 NL2Bash 派生的训练 artifact：

- source/provenance 和去泄漏 manifest；
- AST atoms 与可逆 source ranges；
- canonical atom sequences；
- train-only procedure lexicon、概率和 \(K\)；
- MAP/count-conditioned posterior schedules；
- TokMem/EOC-only/TapMem token-ID targets。

这属于训练预处理，而不是手工发明测试题。

## 3. 原始数据到 train/dev 的确定性流程

### 3.1 固定版本和稳定 sample ID

正式实现先 pin：

- NL2Bash repository commit；
- InterCode repository commit；
- 六个 source files 的 SHA-256；
- NL2Bash `data/scripts/filter_data.py`、`data/scripts/split_data.py`、
  `nlp_tools`、legacy `bashlint` commits；
- official-split preprocessing container image digest（Python 3.9）；
- `bashlex==0.18` wheel SHA-256；
- preprocessing code commit；
- backbone tokenizer name、revision 和 vocabulary hash。

每个 NL2Bash raw pair 的稳定 ID 为：

```text
sample_id =
SHA256(source_file + "\0" + line_number + "\0" +
       instruction_raw + "\0" + command_raw)
```

不得用 Python 内置 `hash()`，因为它可能跨进程和运行变化。

#### 3.1.1 先忠实重放 NL2Bash 官方 filter/split

官方没有一个可直接贴到 12,607 个 raw source lines 上的预制 split manifest；split
只在 filtered rows 上定义。因此顺序必须与 repository 脚本一致，不能先删 test 再
重算 top utilities：

1. 在完整 12,607 对 `all.nl/all.cm` 上运行
   `data/scripts/filter_data.py` 的
   `filter_by_most_frequent_utilities(data_dir, 100)`，先统计完整 raw corpus 的
   utility frequency/black-grey list，再生成 9,305 对
   `all.nl.filtered/all.cm.filtered`；
2. 对这 9,305 对运行 `data/scripts/split_data.py`。脚本固定
   `RANDOM_SEED=100`、`num_folds=12`，先按 normalized-NL groups 排序并以
   `randint(0,11)` 分配 0--9/train、10/dev、11/test，再把 dev/test 中 command
   exact 出现在初始 train 的 rows 移回 train；
3. 最终 pair counts 必须为 train/dev/test =
   8,090/609/606，filtered-out raw rows 为 3,302，normalized-NL group 数为
   8,412；
4. 用 `(instruction_raw, command_raw)` 的 multiset 和 source-line stable IDs
   将六个 split outputs 确定性映射回 raw lines；同 key 多次出现时，按 raw source
   line 升序从 queue 取下一个 unmatched occurrence。物化：

   ```text
   official_split_manifest:
   sample_id -> FILTERED_OUT | TRAIN | DEV | TEST
   ```

5. 保存并 hash：两个 raw files、两个 filtered files、六个 train/dev/test files、
   每个阶段的 ordered-file SHA、pair-multiset SHA、行数，以及 12,607-row manifest。
   同时校验 repository `data/bash/random_tokens.txt` 为 8,412 行、SHA-256
   `5f770f78fe1534288b8872d8a32179c36a7ebc5413d0e3eba748f99d7b4b96b0`。

当前 `tokmem` 的 Python 3.10 不能直接 import legacy `bashlint`，会在已删除的
`collections.MutableSet` 上失败；官方脚本还要求 repository `nlp_tools` 在
`PYTHONPATH`。因此第一次 reference replay 在独立、只读 source mount 的 Python
3.9 preprocessing container 中完成，并 pin image digest、所有 source commits 和
环境 lock。若以后写 Python 3.10 compatibility port，其两个 filtered 与六个
split 输出、ordered-file SHA、pair-multiset SHA 和 12,607-row manifest 必须与
该 legacy reference **逐项完全相同**；仅 counts 相同不算忠实重放。

#### 3.1.2 明确定义 source component 和 template group

`source_group_id` 不能是实现者临时指定的标签。先把全部 12,607 个 raw rows（包括
`FILTERED_OUT` 和 official `TEST`）作为顶点；每个 row 计算：

```text
normalized_instruction = official basic_tokenizer normalization
exact_command = command_raw with only the storage LF removed
template_keys = {
  "BASHLINT:" + bashlint_full_template,   # if available
  "BASHLEX:"  + bashlex_fallback_fingerprint  # if available
}
```

两个 template 的逐字段算法就是第 3.2 节下文定义的版本；两处调用同一函数和
config hash，不能分别实现。

建立两个无向图：

1. `template_graph_v1`：两 rows 共享任一 `template_key` 就连边；
2. `source_graph_v1`：两 rows 的 `normalized_instruction` 相等、`exact_command`
   相等，或共享任一 `template_key`，满足任一条件就连边。

关系都按 UTF-8 bytes exact equality 判断，不做 embedding、编辑距离或人工合并；
没有可用 template key 的 row 仍可通过前两类边连接，否则是 singleton component。
分别取传递闭包的 connected components，并定义：

```text
template_group_id =
SHA256("TEMPLATE_GRAPH_V1\0" + "\n".join(sorted(member_sample_ids)))

source_group_id =
SHA256("SOURCE_GRAPH_V1\0" + "\n".join(sorted(member_sample_ids)))
```

`source_group_id` 专用于 provenance、deny-list 和 derived split isolation；
`template_group_id` 专用于 unigram corpus weighting 和第 7.1 节 group sampler，
二者不能混用。保存每种 edge reason 的数量、component size histogram、全部
member lists 和 graph-manifest SHA。

另定义 official-test component deny-list：

$$
D_{\mathrm{official\text{-}test}}
=
\left\{
\operatorname{source\_group\_id}(r):
r.\operatorname{official\_split}=\mathrm{TEST}
\right\}.
$$

即便同 component 中另有 official TRAIN/DEV row，也全部从 induction、神经训练和
\(K\)-selection 删除。这样“原 test 的 duplicate/template component 不进训练”是
实际执行的规则，不只是检查 raw split label。

### 3.2 先做 InterCode provenance mapping

InterCode 直接派生自 NL2Bash，必须在 lexicon induction 之前去泄漏。本地 raw exact
审计已经发现：

| overlap key | InterCode task 数 |
|---|---:|
| exact query | 25 |
| exact gold command | 24 |
| exact `(query, gold)` pair | 23 |

这是只删除行末 LF 后的 Unicode raw exact equality；canonical/template overlap 只会
更大。

provenance search 使用带官方 split 字段的全部 NL2Bash raw rows，包括原 test，
因为它们可能正是 InterCode 的来源；“可用于查明来源”不等于“可用于训练”。完成
mapping 后，原 test 无条件排除，任何与其同 group 的 train/dev duplicate 也按
deny-list 删除。

为每个 InterCode task 建 provenance record：

1. exact `(instruction, command)` 匹配；
2. exact instruction 匹配；
3. exact command 匹配；
4. deterministic full-command template 匹配；
5. 对剩余候选做人工 source adjudication：两人独立查看 InterCode
   `(query, gold)` 与检索到的 NL2Bash raw candidates，只回答是否为 grounding/flag
   更新前的同源 pair；该标注只用于确认具体 ancestor source group 和记录
   provenance，不产生 procedure label，也不能缩小后续 deny candidate set。

其中 normalized instruction 直接复用 NL2Bash 官方 split script 的定义：

```python
" ".join(nlp_tools.tokenizer.basic_tokenizer(instruction)[0])
```

不使用语义 encoder 阈值。

人工候选检索也固定：先取 ordered full utility/operator sequence exact match；若为空，
再取该离散序列 Levenshtein distance 不大于 1 的 rows；按 normalized-instruction
token Jaccard 从高到低排序；Jaccard 同分时按 stable sample ID 升序，空 token
union 的 Jaccard 定义为 0。UI 首屏可显示前 20 条，但 adjudicators 必须分页检查并
标记**全部未截断 candidate groups**；不能只看 top-20，也不能继续扩大阈值直到
“找到一个看起来像的”。

对 task \(t\)，预注册 source candidate set 明确定义为：

$$
C_t
=
C_t^{exact}
\cup C_t^{template}
\cup C_t^{sequence\text{-}retrieval},
$$

不是 adjudicator 临时搜索到的集合。若 \(C_t=\varnothing\)，该 task 无法形成可审计
source resolution，primary 立即失败。

在正式 200 题 resolution 前，用本地已知的 23 个 exact `(query,gold)` pairs 做两项
blinded retrieval positive-controls，二者都同时屏蔽 exact 和 template channels：

1. **exact-sequence branch**：只用原 ordered utility/operator sequence 运行
   sequence retrieval，真实 source group 必须在未截断集合中 23/23 被召回；
2. **edit-distance branch**：把每题 sequence 的第一个 utility token 替换为
   task-specific sentinel `<PC_UNSEEN_taskid>`；先硬断言该 synthetic sequence 在
   corpus 中无 exact match，再只运行 Levenshtein distance \(\le1\) fallback，真实
   source group也必须 23/23 被召回。

sentinel 只用于 retrieval unit test，不进入任何训练数据。保留 template channel
会让 raw-exact command 的 template 必然相同，形成没有召回意义的恒真检查，因此
明确禁止。上述检查只验证已知 positives 上的 sequence-retrieval recall，不把结果
当作另外 177 题已经自动 resolved。

full-command template 使用两个都可重放的 deny keys：

1. NL2Bash source repository 自带的：

   ```python
   bashlint.data_tools.cmd2template(
       command,
       recover_quotation=True,
       arg_type_only=True,
       loose_constraints=True,
       verbose=False,
   )
   ```

   该函数保留 utility、flag、reserved word 和 argument type；运行环境连同
   `bashlint`/`nlp_tools` source commit 一起 pin；
2. `bashlex` fallback fingerprint：按 AST preorder 输出 node kind；对
   pipe/operator/redirection 输出 raw operator；对每个 command 的首个 static word
   输出 executable basename；对其余 direct words，若以 `-` 开头则保留 flag，
   否则只输出 `WORD`；nested substitution 递归输出结构但不保留实体字符串。

若第一种 parser 失败，仍使用第二种；若两种都成功，任一 key 相同都视为同
full-template group。整个过程不用大模型、sentence embedding 或测试执行结果。

每题最终记录：

```text
resolution_mode = exact | template | manual | unresolved
resolved_source_group_ids
all_retrieved_candidate_group_ids
```

`exact` 要求 exact pair 命中，或 exact instruction/command 的全部命中落在同一
connected source group；`template` 要求两个 deterministic templates 指向同一
source group且无冲突；其余情况进入 `manual`。manual resolution 要求两名
adjudicators 的独立 confirmed-group 集合交集非空，交集作为
`resolved_source_group_ids`；只有一人确认或两人指向不同 groups 都记为
`unresolved`，不能以“删除得很宽”冒充找到了 source。

无论人工确认了哪些 rows，最终 deny set 都取：

$$
D_t
=
\operatorname{groups}(C_t)
\cup
\operatorname{connected\_component}
(\text{resolved source groups}),
$$

即所有被 retrieval 召回的 candidate groups 都删除，人工标签不能把它缩小。对
\(D_t\) 中每个 group，删除：

- 原始 source row；
- exact command duplicate group；
- normalized instruction duplicate group；
- 相同完整 \(T(c)\) 的 group。

primary 要求 200/200 tasks 的 `resolved_source_group_ids` 非空，并硬断言这些
source groups 及其 instruction/command/template connected components 均没有进入
train、dev、procedure induction、lexicon 或 \(K\)-selection。任意一题
`resolution_mode=unresolved`，full-200 实验就不能作为 rebuttal primary；不能事后
只在 resolved task subset 上改报 primary。宽 deny-list 仍可生成 exploratory
结果，但不解决 provenance validity。

只删除**完整命令/source group**；不能删除所有与测试 gold 共享局部
`find → xargs` 等 subsequence 的训练样本，因为局部 motif 正是要学习并迁移的知识。

### 3.3 过滤与 group split

在第 3.1.1 节已对完整 raw corpus 忠实重放官方 filter/split、并完成 raw
provenance mapping 后：

1. 只保留 official `TRAIN` 和 `DEV` rows；永久移除 `TEST` 和
   `FILTERED_OUT`，不再重算 top-100 utility allowlist；
2. 删除
   \(D_{\mathrm{official\text{-}test}}\cup D_{\mathrm{InterCode}}\)
   覆盖的全部 source components；
3. 在剩余 official train+dev rows 上直接复用第 3.1.2 节冻结的
   `source_group_id`；不得在删行后重新计算 component membership；
4. 整个 source component 只能进入一个 derived split；
5. 令 `component_id = source_group_id`；
6. 按该 stable hash 做 90% train / 10% dev：

```text
bucket = uint64(SHA256(component_id)[0:8]) mod 10
bucket == 0 -> dev
otherwise   -> train
```

NL2Bash 的原 test 不用于 procedure induction、神经训练或 \(K\) 选择；InterCode
是唯一最终测试。每条 `raw_pairs` 必须携带 materialized official split，构建器
硬断言进入 derived train/dev、lexicon 和 \(K\)-selection 的 official `TEST` 与
`FILTERED_OUT` 行数都为 0。
split 后必须输出：

- raw、official-filtered、official train/dev/test、decontaminated、parseable、
  strict-flat 数量；
- train/dev group 数和 pair 数；
- 每个 exclude reason 数量；
- train/dev 之间 exact instruction、command、template overlap，要求全部为 0；
- 与 InterCode deny keys 的 overlap，要求全部为 0；
- 与任何 official TEST source component 的 overlap，要求为 0；
- blinded sequence-only positive-control recall：exact-sequence branch 与
  edit-distance-one branch 都要求 23/23；
- 200 个 InterCode task 的 concrete resolved source-group 覆盖率，要求 200/200，
  并逐组证明其 connected component 未进入任何训练/选择 artifact。

procedure induction 的 corpus objective 以 template group 为单位加权：一个 group 的
总权重为 1；\(n_g\) 只统计去泄漏后、留在 derived train 的 rows，组内每条
paraphrase 权重为 \(1/n_g\)。这样同一 template 的大量改写不会虚增 motif 概率。
神经训练的 eligible rows 全部保留在 group 内，再由第 7.1 节的
无放回 group sampler 跨 epoch round-robin 选择；所有方法的 presentation 顺序和
次数相同。

### 3.4 测试 gold 使用防火墙

InterCode gold 只允许用于：

- 构造训练 deny-list；
- 在训练词典、\(K\)、prompt、checkpoint selection 全部冻结后生成 structural
  diagnostic manifest；
- 计算 reference-program segmentation statistics。

时间顺序必须写入带 hash 的 manifest：

```text
fit train lexicon and freeze all hyperparameters
→ hash lexicon/K/prompt/checkpoint rule
→ generate and hash test structural manifest
→ run model inference
→ read-only diagnostic analysis
```

禁止根据 200 个 gold 的结构分布或模型结果修改 atomizer、support threshold、\(K\)、
sampling rule、prompt 或 checkpoint。

## 4. 从 raw Bash 得到 syntax-safe atoms

### 4.1 v1 eligibility 和 flatten 规则

procedure lexicon 的 v1 只使用 strict-flat train commands：

```text
root command
or
root pipeline = command, pipe, command, ..., pipe, command
```

atom 是 root `CommandNode` 本身，或 strict root `PipelineNode` 的 direct
`CommandNode` child；不递归进入：

- `$()` 或 backtick command substitution；
- process substitution；
- quoted string 和 awk/sed program；
- `xargs` 的 subcommand 参数；
- `find -exec`；
- compound body。

含 heredoc、multiple roots、empty/comment-only input、root list、root compound 或
pipeline compound component 的训练样本从 v1 induction 排除并报告。未来若扩展，
整个 compound node 必须是 opaque atom，不能把内部 ranges 与外层 atoms 混在一起。

InterCode 正式 200 题一题也不能因为不符合 v1 而排除；parser 完全不参与主 SR
打分。123 个 strict flat-pipeline tasks 只形成一个查看 released gold 后冻结、
且在 inference 前写入 manifest 的 exploratory diagnostic subset。

第 2.2/4.3 节对 InterCode 200 题所做的 round-trip audit 使用一个更宽但只读的
reference flatten：root command 为一个 atom；strict root pipeline 取 direct command
children；root list 只递归展开其 root-level command/pipeline children，并把 root-level
operators 保存在完整 gaps 或 suffix；仍不进入 substitution/compound body。这个
audit-only rule 不用于 procedure lexicon、训练 target 或正式推理。

### 4.2 atom core、connector 和 canonical signature

对第 \(i\) 个 atom 保存：

```text
atom_id
node_kind
node_path
char_range = [start, end)
raw_core = command_raw[start:end]
incoming_connector_kind
static_utility_head
canonical_signature
```

`bashlex.pos` 是 Python Unicode character offset，不是 UTF-8 byte offset。byte range
只能通过 `len(raw[:char_pos].encode("utf-8"))` 派生。

primary canonical signature 定义为：

```text
(incoming_connector_kind, normalized_static_utility_head)
```

其中：

- 首 atom 的 connector 为 `START`；
- 普通 pipe 后为 `PIPE`；
- `|&` 后为 `PIPE_STDERR`；
- utility 取 command 的首个 static direct word，并将 `/bin/echo` 规范为 basename
  `echo`；
- 含 expansion 的动态 command head 不猜测真实 utility，映射为 connector-specific
  `<UNK_UTILITY>`；
- train template-group support 小于 2 的 head 映射为 connector-specific
  `<RARE_UTILITY>`。

这样做的目的不是宣称 utility name 已经是完整 procedure，而是得到有限、可复用的
基本 alphabet。路径、数字、flags、glob 和自由字符串不进入 primary signature，
仍逐字符保留在 raw code target 中，由冻结 backbone 在 instruction/context 条件下
生成。

旧方案把 `canonical_text + utility sequence + flags + operators + AST shape` 拼接后
编码，但没有给出各部分的权重和相似度含义。主方案不再做这一步。若以后做
`head+flags` structural signature，只能作为预注册 ablation，不能根据 InterCode
结果切换。

### 4.3 raw 序列化必须逐字符可逆

设 atoms 的 character ranges 按顺序为 \([a_i,b_i)\)。完整 inter-atom gap：

$$
g_i = \mathrm{raw}[b_i:a_{i+1}]
$$

包含全部左右空格、`|`/`|&`、反斜线换行和注释，不能只保存 AST PipeNode 的字符。

v1 的 serialization ownership 固定为：

- prefix `raw[0:a_1]` 给首 atom；
- gap \(g_i\) 给左 atom；
- suffix `raw[b_L:len(raw)]` 给末 atom。

因此 base chunks 为：

```text
chunk_1 = prefix + atom_1 + gap_1
chunk_i = atom_i + gap_i              (1 < i < L)
chunk_L = atom_L + suffix
```

若 \(L=1\)，唯一 chunk 就是完整 raw command。任意连续 atom partition 的 raw span
都是其 chunks 的拼接，所以必须满足：

```text
"".join(base_chunks) == command_raw
"".join(procedure_raw_spans) == command_raw
```

该 ownership 只是为了 lossless serialization；procedure 的概念边界仍是 atom
\(i\) 后的 gap ID。非末 span 可能以 `|` 结束，带 memory/EOC 的 annotated target
本身不是可执行 Bash；只有删除控制 token 后的完整 command 才能 parse/execute。

本地审计对 12,196 条 strict-flat NL2Bash 和 200 条 InterCode gold 都得到 0 个
character/UTF-8 round-trip failure。真实数据中还存在 9 个 `|&` gap、14 条
NL2Bash trailing-comment suffix 和一个 InterCode trailing `&`，所以不能只保存
`root.pos`。

### 4.4 tokenizer round-trip 是另一项 hard check

字符串可逆不保证独立 tokenizer encoding 后再拼接 token IDs 仍然可逆。为使 MAP
和不同 boundary views 的普通 Bash token 序列也完全相同，每个 backbone 必须：

1. 对第 4.3 节的每个 **base atom chunk** 分别用
   `add_special_tokens=False` 编码一次并缓存，不能随 procedure view 重新编码整个
   variable-length span；
2. procedure span 的 ordinary token IDs 只是所覆盖 base-chunk token lists 的顺序
   拼接；
3. 在这些固定 ordinary IDs 之间插入 memory/EOC IDs，并只在完整 target 末尾追加
   backbone 的 response-end sequence；
4. round-trip 时先在末尾完整 response-end sequence 处截断，再按 ID 删除
   memory/EOC；不能把 response-end sequence 的组成 IDs 在全文全局过滤；
5. 用 `clean_up_tokenization_spaces=False` decode；
6. 与原始 command 的 UTF-8 bytes 和 SHA-256 比较；
7. 检查所有 views 删除控制 IDs 后不仅 decode 相同，而且 ordinary token-ID sequence
   逐项相同。

要求 100% 相等；失败视为数据构造 bug，不允许自动移动 boundary 或修复 Bash。

## 5. 从 atom sequences 诱导 procedure vocabulary

### 5.1 occurrence、type 和 memory token

对训练 command 得到：

$$
x=(x_1,\ldots,x_L),
$$

其中 \(x_i\) 是 canonical atom signature。

一个 procedure occurrence 是某条 command 中的连续范围 \([i,j]\)。procedure type
是 canonical sequence：

$$
v=(x_i,\ldots,x_j).
$$

所有 canonical sequence 完全相同的 occurrences 共用一个 type 和一枚 memory token；
不同路径、flags、字符串和数值仍保留在各自 raw spans 中。

这不是 k-means：

- 不存在同一 occurrence 被两个 cluster 重复计数；
- 不存在相似度阈值或 encoder 选择；
- 一个 canonical sequence 只能有一个 procedure ID；
- memory token 的含义是“该重复结构 motif”，不是人工命名的搜索/过滤/聚合类别。

### 5.2 候选数量怎样控制

所有 mandatory singleton signatures 都进入初始 vocabulary，保证每条 train sequence
至少有一条完整 singleton path。

长度至少为 2 的候选 \(v\) 必须同时满足：

$$
\operatorname{df}(v)
=
\left|\{g:\exists x\in g,\ v\text{ 是 }x\text{ 的 contiguous subsequence}\}\right|
\ge 2,
$$

以及：

$$
\operatorname{proper}(v)
=
\left|\{g:\exists x\in g,\ v=x_{i:j},\ i>1\ \text{or}\ j<L\}\right|
\ge 1.
$$

第二条要求它至少在一条训练 command 中作为 proper subspan 出现，避免只把两条相似
完整命令压成一个“整题 memory”。即便如此，某个 piece 仍可能覆盖另一条较短命令的
全部 atoms，所以后续还要报告 whole-command posterior mass。

用 generalized suffix trie/array 在 canonical sequences 上统计 unique types 和 group
support，不把每个 raw overlapping occurrence 展开成聚类样本。没有 `max_len=4`：
最长候选由训练 command 的真实长度决定。

在尚未去泄漏的 raw audit 中，若只保留 strict-flat commands 且 top-level heads 属于
NL2Bash repository utility allowlist：

| 项目 | 数量 |
|---|---:|
| eligible commands | 11,379 |
| distinct `(connector, head)` singleton signatures | 193 |
| multi-atom types with command support \(\ge2\) | 782 |
| 同时至少一次为 proper subspan | 550 |

这是候选规模审计，不是最终 train split 结果。它说明当前数据不需要任意 span cap；
最终 decontaminated counts 必须重新输出。

嵌套候选可以同时存在：

```text
v1 = [START:find]
v2 = [START:find, PIPE:xargs]
v3 = [START:find, PIPE:xargs, PIPE:wc]
```

它们是不同粒度的词典项。在一条完整 segmentation path 中，覆盖同一 atom 的 arcs
互斥；模型通过完整 partition posterior 分配概率，而不是把三个 raw spans 当作三条
独立训练标签。

### 5.3 带 stop 概率的 unigram generative model

令 content procedure vocabulary 为 \(V\)：

$$
\theta_v>0,\qquad \sum_{v\in V}\theta_v=1.
$$

再学习一个停止概率 \(0<\rho<1\)。生成一个 content piece 的 arc weight 为：

$$
w_v=(1-\rho)\theta_v.
$$

一条含 \(J\) 个 pieces 的完整 segmentation：

$$
B=(v_1,\ldots,v_J),\qquad
v_1\oplus\cdots\oplus v_J=x
$$

具有概率：

$$
P(B)
=
\rho\prod_{j=1}^{J}w_{v_j}
=
\rho(1-\rho)^J\prod_{j=1}^{J}\theta_{v_j}.
$$

\(\rho\) 使所有有限 piece sequences 的总概率为 1，也把 segmentation 长度先验与
content probabilities 分开。\(K=|V|\) 只统计 content procedure pieces，不包含
\(\rho\)、TapMem EOC 或 response-end token。

一条切分的明确 log score 是：

$$
s(B)
=
\log\rho+J\log(1-\rho)+\sum_{j=1}^{J}\log\theta_{v_j}.
$$

例如：

```text
B1 = [find+xargs] [wc]
s(B1) = log rho + 2 log(1-rho)
        + log theta(find+xargs) + log theta(wc)

B2 = [find] [xargs+wc]
s(B2) = log rho + 2 log(1-rho)
        + log theta(find) + log theta(xargs+wc)

B3 = [find] [xargs] [wc]
s(B3) = log rho + 3 log(1-rho)
        + log theta(find) + log theta(xargs) + log theta(wc)
```

所有 \(\theta,\rho\) 只由 decontaminated train 学习；没有“变化分数”或 embedding
相似度。

### 5.4 exact forward-backward

用 positions \(0,\ldots,L\)。arc \((i,j,v)\) 表示
\(v=x_{i+1:j}\)。

Forward：

$$
\alpha[0]=1,
$$

$$
\alpha[j]
=
\sum_{(i,j,v)}
\alpha[i]w_v.
$$

Backward：

$$
\beta[L]=1,
$$

$$
\beta[i]
=
\sum_{(i,j,v)}
w_v\beta[j].
$$

实现必须在 log-space，并检查：

$$
\alpha[L]=\beta[0].
$$

整条 canonical command 的 normalized marginal probability 是：

$$
P(x)=\rho\alpha[L].
$$

某个 arc occurrence 的 posterior 是：

$$
\gamma(i,j,v\mid x)
=
\frac{\alpha[i]w_v\beta[j]}{\alpha[L]}.
$$

### 5.5 EM 学习什么

对 template group \(g\) 中第 \(n\) 个 pair 使用权重
\(\omega_n=1/|g|\)。piece expected count：

$$
C_v
=
\sum_n\omega_n
\sum_{(i,j,v)\in x_n}
\gamma_n(i,j,v).
$$

令：

$$
C_{\mathrm{all}}=\sum_v C_v,\qquad
N_{\mathrm{eff}}=\sum_n\omega_n.
$$

M-step：

$$
\theta_v
=
\frac{C_v+\epsilon}
{C_{\mathrm{all}}+K\epsilon},
\qquad \epsilon=10^{-8},
$$

$$
\rho=\frac{N_{\mathrm{eff}}}
{N_{\mathrm{eff}}+C_{\mathrm{all}}}.
$$

这里 \(\rho\) 的分母使用未加 pseudocount 的真实 expected piece count。
\(\epsilon\) 等价于对 content probabilities 使用 Dirichlet
\(\alpha_v=1+\epsilon\)，因此固定 vocabulary 的 EM 优化：

$$
\ell_{\mathrm{MAP}}
=
\ell+\epsilon\sum_{v\in V}\log\theta_v.
$$

connector-specific `<UNK_UTILITY>` 是 mandatory piece；pseudocount 保证即使 train
未出现某个 fallback，其概率也不严格为 0。它不是未定义的 post-hoc probability
floor。

初始化固定跑三种：

1. \(\theta_v\propto\operatorname{df}(v)+\epsilon\)；
2. uniform；
3. \(\theta_v\propto\operatorname{df}(v)|v|+\epsilon\)。

\(\rho\) 初始化为 singleton-only paths 的 ML estimate。每种初始化运行到：

```text
relative train MAP-objective improvement < 1e-6
for 3 consecutive iterations
or 100 EM iterations
```

三种解分别继续执行下一节的完整 pruning path；对每个整数 \(K\)，只用 train
MAP objective 选择三条 path 中最高的 \((V_K,\theta_K,\rho_K)\)，再交给
dev 做 \(K\) 选择。报告三条 path 的 train marginal likelihood/MAP objective、
同一 \(K\) 下的
expected-count weighted vocabulary Jaccard 和 segmentation statistics。EM 非凸，
因此不能只隐藏一个幸运初值。

### 5.6 如何裁掉嵌套或冗余 pieces

每次固定 vocabulary 的 EM 收敛后，对每个非 mandatory piece \(v\) 计算
**one-step deletion ablation score**，而不把它错误称为“删除后重新训练的真实损失”。

删除 \(v\) 时固定 \(\rho\)，只把剩余 content probabilities 重归一化：

$$
\theta_u^{-v}
=
\frac{\theta_u}{1-\theta_v},\qquad u\ne v.
$$

在整个 train 上重新跑 forward，得到：

$$
\Delta_v^{\mathrm{one}}
=
\ell(V,\theta,\rho)
-
\ell(V\setminus\{v\},\theta^{-v},\rho),
$$

其中：

$$
\ell(V,\theta,\rho)
=
\sum_n\omega_n
\log\bigl[\rho\alpha_n[L_n]\bigr].
$$

该分数精确表示“在当前参数映射、固定 length prior 下删除这个 piece”的即时
likelihood 变化；它不等于删除后重新 EM 的最优 likelihood 差。分数可以为负，
batch deletions 之间也会 interaction，所以 pruning step 本身不保证 likelihood
单调；每次 pruning 后的新 vocabulary 会重新 EM。

确定性 pruning：

1. 对三种 initialization path 分别在初始 seed vocabulary 上 EM；
2. 按 \(\Delta_v^{one}\) 从小到大排序，canonical tuple 的 UTF-8
   lexicographic order 作为 tie-break；
3. 在 \(K_{\min}\le247\)、\(K>247\)、\(n_{\rm removable}\ge1\) 的前提下，令当前
   removable piece 数为 \(n_{\rm removable}\)，每轮删除

   $$
   n_{\rm del}
   =
   \min\left(
   \max\left(1,\lfloor0.2n_{\rm removable}\rfloor\right),
   K-247
   \right)
   $$

   个 deletion score 最低的 pieces；即通常删除向下取整的 20%，不足 1 个时删除
   1 个，最后一轮准确停在 \(K_{\max}=247\)，不会跨过 247；
4. 若同时删除集合为 \(S\)，初始化剩余参数：

   $$
   \theta_u'
   =
   \frac{\theta_u}{1-\sum_{v\in S}\theta_v};
   $$

5. 重新 EM；
6. 到 247 后，每次只删一个 piece、重新 EM，并保存每个整数
   \(K\in[K_{\min},247]\) 的 train model；
7. 同一 \(K\) 的三条 path 中，取 train MAP objective 最高者作为交给 dev 的
   \(V_K\)，tie-break 为 lexicon hash。

mandatory singleton/fallback pieces永不删除。该过程会让 `[find]` 和
`[find+xargs]` 在完整 paths 中竞争；若长 piece 不能提高 corpus likelihood，它的
deletion score 会低并被裁掉。

### 5.7 \(K\) 怎样确定

\(K\) 是 TapMem memory-bank capacity，不是真实 procedure 数。

当前 Llama 路径有 248 个 reserved slots；TapMem EOC 占一个，所以：

$$
K_{\max}=247.
$$

\(K_{\min}\) 是 mandatory singleton/fallback type 数。若 \(K_{\min}>247\)，当前
signature 定义与模型容量不兼容；必须在看 InterCode 结果前将低 group-support heads
并入 connector-specific `<RARE_UTILITY>`，重新生成完整 manifest，或者报告构造失败。

每个整数 \(K\in[K_{\min},247]\) 的 \((V_K,\theta_K,\rho_K)\) 都只在 train 拟合。
dev 指标：

$$
\operatorname{NLL}_{dev}(K)
=
-
\frac{
\sum_{x\in D_{dev}}\omega_x
\log\left[\rho_K\alpha_K^{(x)}[L_x]\right]
}{
\sum_{x\in D_{dev}}\omega_x L_x
}.
$$

dev 中同样令一个 template group 的总权重为 1。

bootstrap 独立单位是 command-template group，不是相关的 paraphrase row。对同一份
bootstrap group sample 同时计算全部 \(K\)，共 1,000 次。令：

$$
K_{\mathrm{best}}
=
\arg\min_K \operatorname{NLL}_{dev}(K),
$$

$$
d_K
=
\operatorname{NLL}_{dev}(K)
-
\operatorname{NLL}_{dev}(K_{\mathrm{best}}).
$$

由 paired bootstrap 得到 \(se(d_K)\)，选择：

$$
\hat K
=
\min\{K:d_K\le se(d_K)\}.
$$

其中 \(K_{\mathrm{best}}^{full}\) 始终由完整 dev 一次性确定并在 1,000 个
replicates 中固定。第 \(b\) 次 paired bootstrap 计算：

$$
d_K^{(b)}
=
\operatorname{NLL}_{dev}^{(b)}(K)
-
\operatorname{NLL}_{dev}^{(b)}(K_{\mathrm{best}}^{full}),
$$

$$
se(d_K)=\operatorname{sd}_b(d_K^{(b)}).
$$

不能在每个 replicate 中重新选择一个不同的 \(K_{\mathrm{best}}^{(b)}\)。这是明确的
**paired-difference one-SE rule**。InterCode 的 gold、SR 或 reference entropy
不参与 \(\hat K\) 选择。正式运行前保存全 \(K\) dev curve、bootstrap seed、
\(\hat K\) 和 lexicon hash。

如果 \(\hat K=K_{\min}\) 或触发第 13 节预注册的 vocabulary-richness 阈值，数据没有
支持 variable-length motif memory；不能强行把该实验写成模糊 procedure 实验。

## 6. 怎样得到并衡量不同切分

### 6.1 MAP segmentation

Viterbi DP 用第 5.3 节的 \(s(B)\)：

$$
B_{\mathrm{MAP}}(x)=\arg\max_{B\in\mathcal S(x)}s(B).
$$

由于主 posterior 要条件化在 \(J_{\mathrm{MAP}}\)，MAP 并列不能依赖代码遍历顺序。
Viterbi state 和最终 backtrace 使用以下确定性全序：

1. \(s(B)\) 较大者优先；
2. 分数按 float64 bitwise equality 相同时，\(J=|B|\) 较小者优先；
3. \(J\) 仍相同时，按从左到右的 boundary bit vector
   \((b_1,\ldots,b_{L-1})\) 取 lexicographically smallest；
4. 最后按 piece-ID sequence 取 lexicographically smallest。

分数计算全部使用 float64；不使用可调的“近似相等”容差。若需要跨硬件完全重放，
artifact 同时保存每条候选 arc 的 log-weight 和最终 tie-break fields。穷举 toy
test、Viterbi、schedule builder 必须调用同一个 comparator。

保存：

```text
MAP piece IDs
MAP boundary bit vector
J_MAP
MAP log probability
MAP tie-break fields/version
```

### 6.2 full posterior 和 exact FFBS

完整 segmentation posterior：

$$
q(B\mid x)
=
\frac{P(B)}
{\sum_{B'\in\mathcal S(x)}P(B')}.
$$

从末位置 \(j=L\) 开始，上一条 arc \((i,j,v)\) 的条件概率：

$$
P((i,j,v)\mid j,x)
=
\frac{\alpha[i]w_v}{\alpha[j]}.
$$

采样该 arc 后令 \(j\leftarrow i\)，直到 0，再反转 arcs，即得到 exact
forward-filtering backward-sampling。主设置温度固定为 1，不做 top-\(k\) 截断。

### 6.3 主模糊设置：固定 procedure 数，只采 boundary placement

直接从 full posterior 采样会同时改变：

- procedure/memory token 数；
- EOC 数；
- TCRA route-loss positions；
- target length 和各 memory token 频次。

这样 `sampled - MAP` 不能只解释为 boundary ambiguity。因此主设置条件化在
\(J=J_{\mathrm{MAP}}\)。

定义 count-aware forward：

$$
\alpha[0,0]=1,
$$

$$
\alpha[j,r]
=
\sum_{(i,j,v)}
\alpha[i,r-1]w_v.
$$

则：

$$
Z_J(x)=\alpha[L,J],
$$

$$
q_J(B\mid x)
=
q(B\mid x,J=J_{\mathrm{MAP}})
=
\frac{\prod_{v\in B}w_v}{Z_{J_{\mathrm{MAP}}}(x)}.
$$

count-conditioned FFBS 从 \((j,r)=(L,J_{\mathrm{MAP}})\) 开始：

$$
P((i,j,v)\mid j,r,x)
=
\frac{\alpha[i,r-1]w_v}{\alpha[j,r]}.
$$

采样后更新 \((j,r)\leftarrow(i,r-1)\)。结合第 4.4 节的固定 base-chunk
tokenization，该 schedule 与 MAP 有完全相同的 ordinary Bash IDs、memory-token 数、
EOC 数、routing sites 和总 target token 数，只改变 memory IDs 所控制的 chunk
分组和 boundary placement。

若某条 command 在该 \(J\) 下只有一个合法 partition，采样自然退化成 MAP；不能强迫
它产生三个 views。

### 6.4 full-posterior granularity stress

从 \(q(B\mid x)\) 不条件化地 FFBS，只作为 secondary stress test。它同时测试
boundary 位置和 procedure 粒度不确定性，必须报告平均 \(J\)、EOC 数和 routing-site
数，不能把结果单独归因于 boundary placement。

### 6.5 uniform-lattice control

一个高价值 1B control 是在同一 retained lexicon、同一
\(J=J_{\mathrm{MAP}}\) 下，把每条合法 partition 设为等概率。实现上把 count-DP 的
所有合法 arc weight 临时设为 1 再 FFBS。

它保持 memory vocabulary、procedure 数和语法合法性不变，只检验：

> 在 retained lexicon 和固定 \(J\) 给定后，结果是否依赖 unigram
> \(\theta\) 对合法 partitions 的相对 weighting？

不能从所有 syntax gaps 均匀切后再临时新建 memory ID，因为未保留的 span 没有
procedure token，会改变 memory bank。

### 6.6 boundary probability 和 ambiguity 指标

full posterior 下，内部 gap \(t\in\{1,\ldots,L-1\}\) 是 boundary 的概率：

$$
p_t
=
P(b_t=1\mid x)
=
\frac{\alpha[t]\beta[t]}{\alpha[L]}.
$$

一致性检查：

$$
\mathbb E[|B|\mid x]
=
\sum_{(i,j,v)}\gamma(i,j,v)
=
1+\sum_{t=1}^{L-1}p_t.
$$

gap entropy：

$$
h_t=-p_t\log p_t-(1-p_t)\log(1-p_t).
$$

command-level mean normalized gap entropy：

$$
A_{\mathrm{gap}}(x)
=
\frac{1}{L-1}
\sum_{t=1}^{L-1}\frac{h_t}{\log2}.
$$

\(L=1\) 时记为 `NA`，不做除零。

完整 segmentation entropy：

$$
H_{\mathrm{seg}}(x)
=
\log\alpha[L]
-
\sum_{(i,j,v)}
\gamma(i,j,v)\log w_v.
$$

\(\sum_t h_t\) 不等于 \(H_{\mathrm{seg}}\)，因为不同 gaps 的事件相关。两者必须分开
报告。

count-conditioned boundary marginal 为：

$$
\beta[L,0]=1,\qquad
\beta[i,k]=\sum_{(i,j,v)}w_v\beta[j,k-1],
$$

$$
p_t^{(J)}
=
\frac{
\sum_{r=1}^{J-1}
\alpha[t,r]\beta[t,J-r]
}{
\alpha[L,J]
},
$$

其中 count-aware \(\beta[t,k]\) 表示从位置 \(t\) 到结尾恰用 \(k\) 个 pieces 的
suffix mass。主 sampled schedule 的 boundary-position ambiguity 使用
\(p_t^{(J_{\mathrm{MAP}})}\)。

count-conditioned arc posterior 和完整 view entropy 分别为：

$$
\gamma_J(i,j,v)
=
\frac{
\sum_{r=0}^{J-1}
\alpha[i,r]w_v\beta[j,J-r-1]
}{
\alpha[L,J]
},
$$

$$
H(q_J)
=
\log\alpha[L,J]
-
\sum_{(i,j,v)}
\gamma_J(i,j,v)\log w_v.
$$

还必须报告：

- \(q_J(B_{\mathrm{MAP}}\mid x)\)；
- effective view count \(\exp H(q_J)\)；
- `sample != MAP` 比例；
- 实际 presentation manifests 中，同一 sample 跨 epoch/data seed 出现的
  distinct views；
- \(P(|B|=1\mid x)\)；
- \(\mathbb E|B|\)；
- multi-atom piece expected segment rate；
- atoms 被 multi-atom pieces 覆盖的 expected proportion。

这些量说明数据实际上提供了多少边界歧义，不能只看原 command 有几个 utilities。

## 7. 生成 TokMem/TapMem 训练数据

### 7.1 离线 view schedule

仅在 \(L\ge3\) 子集报告高 entropy 不足以证明神经训练真的接触了模糊边界；本地
raw audit 中 8,346/12,196 条 eligible commands 的 \(L=1\)。因此正式神经训练使用
一个**预注册且所有条件共享**的 group-balanced presentation sampler。

在 procedure model、\(\hat K\) 和 MAP 都冻结后，定义 alternative-bearing pair pool：

$$
\mathcal A
=
\left\{
x:
J_{\mathrm{MAP}}(x)\ge2,\;
|\mathcal S_{J_{\mathrm{MAP}}}(x)|\ge2,\;
1-q_J(B_{\mathrm{MAP}}\mid x)\ge0.25
\right\},
$$

阈值 0.25 只用 train posterior，一次冻结；不能根据神经结果调整。令：

$$
\mathcal G_A
=
\{g:g\cap\mathcal A\ne\varnothing\},
\qquad
\mathcal G_R
=
\{g:g\cap\mathcal A=\varnothing\},
$$

并令 \(\mathcal G_{L\ge3}\) 为至少含一条 \(L\ge3\) strict-flat train pair 的
template groups。primary 构造先检查：

$$
|\mathcal G_A|
\ge
\max\left(100,\left\lceil0.20|\mathcal G_{L\ge3}|\right\rceil\right),
\qquad
|\mathcal G_R|\ge100.
$$

该 gate 防止只靠一两个高熵 group 反复出现来伪造 ambiguity dose。含
\(\mathcal A\) row 的 mixed group 只属于 \(\mathcal G_A\)；它的非-\(\mathcal A\)
rows 不进入 primary \(\mathcal G_R\) stream，保证两个 pool 的 template groups
互斥。

定义每个 epoch、每个 pool 使用的不同 group 数：

$$
M
=
2\left\lfloor
\frac{\min(|\mathcal G_A|,|\mathcal G_R|)}{2}
\right\rfloor.
$$

因此每个 epoch 有 \(M/2\) 个 effective batches、共 \(2M\) 个 presentations；
每个 effective batch 恰有 2 个 \(\mathcal G_A\) 和 2 个
\(\mathcal G_R\) presentations。构造流程是：

1. 分别按 `SHA256(presentation_seed, epoch, pool, group_id)` 对
   \(\mathcal G_A,\mathcal G_R\) 作 deterministic permutation；
2. 每个 pool 只取前 \(M\) 个**不同** groups，不循环；一个 group 在同一 epoch
   最多贡献一个 presentation；
3. 对 \(\mathcal G_A\) group，只从 \(g\cap\mathcal A\) 的 stable-ID 排序 rows
   中选择；对 \(\mathcal G_R\) group，从该 group 全部 rows 中选择；跨 epoch 按该
   group 已被选中的次数 round-robin；
4. 第 \(b\) 个 batch 取两个相邻 A-group rows 和两个相邻 R-group rows，再用
   `SHA256(presentation_seed, epoch, batch_id)` 对四个位置作固定 permutation。

禁止 pool exhausted 后循环。旧的“循环少量 \(\mathcal G_A\) 直到凑够原数据集
epoch 长度”只可另作 oversampling stress test，不能进入 primary 或替代上述
group-support gate。

每个 presentation 的跨 artifact join key 固定为：

```text
presentation_id =
SHA256("PRESENTATION_V1\0" + presentation_seed + "\0" +
       epoch + "\0" + effective_batch_id + "\0" +
       position_in_batch + "\0" + template_group_id + "\0" +
       sample_id + "\0" + within_epoch_occurrence_index)
```

三-epoch manifest 内 `presentation_id` 必须全局唯一；同一 ID 原样写入
presentation、view 和 serialized-target records，禁止用可能跨 epoch 重复的
`sample_id` 代替 join key。

主 `presentation_seed=314159`。Fixed-MAP、count-conditioned、TokMem、EOC-only 和
TapMem 使用逐 presentation 完全相同的 sample-ID/batch manifest；不能只给 sampled
setting oversample \(\mathcal A\)。另外必报不使用 balanced sampler 时、按自然
train pairs 计算的 expected change rate：

$$
R_{\mathrm{natural}}^{expected}
=
\frac1{|D_{train}|}
\sum_{x\in D_{train}}
\left[1-q_{J_{\mathrm{MAP}}}
(B_{\mathrm{MAP}}\mid x)\right],
$$

明确区分“原语料 prevalence”和“训练 treatment dose”。

对每个 train presentation（而不是每个唯一 pair）：

```text
seed_bytes =
SHA256(data_seed + "\0" + sample_id + "\0" +
       epoch + "\0" + within_epoch_occurrence_index)
```

取前 128 bits 初始化 NumPy `PCG64`，精确采一个 segmentation。presentation
manifest 和 view schedule 在训练前 materialize 为 JSONL 并 hash。Fixed-MAP 条件
对同一个 presentation manifest 使用 \(B_{\mathrm{MAP}}\)；sampled 条件使用 FFBS。

随机性分开：

- `data_seed` 只决定 FFBS schedule；
- `presentation_seed` 只决定 sample/group presentation 顺序；
- `model_seed` 只决定 memory/TCRA 参数初始化和其余训练 stochasticity。

同一个 data seed 内，TokMem、EOC-only、TapMem 使用逐 presentation 完全相同的
view ID。固定 `data_seed ∈ {1729, 7919}`、`model_seed ∈ {40,41,42}`；不能把三个
model seeds 误写成三份独立 boundary datasets。

### 7.1.1 实际边界变化剂量怎样计算

令 \(u\) 遍历全部 epoch 的全部 training presentations，\(B_u\) 是该次 FFBS
切分，\(B_u^{MAP}\) 是对应命令的 MAP 切分。主剂量是：

$$
R_{\mathrm{changed}}
=
\frac{\sum_u\mathbf 1[B_u\ne B_u^{MAP}]}
{\#\{u\}}.
$$

这是逐 optimizer-presentation 计数，分母包含 \(L=1\)、唯一切分和实际采回 MAP 的
样本，不在 \(L\ge3\) 子集内美化比例。对 method \(m\)，令
\(n_{AR}^{m}(u)\) 为该 presentation 进入 autoregressive CE 的 supervised target
token 数；另令 \(n_{EOC}(u)\) 为 EOC-only/TapMem serialization 中的 EOC 数，
\(n_{route}(u)\) 为 TapMem route-loss sites 数，计算：

实现顺序必须是：先只从 `views.jsonl` 计算 \(R_{\mathrm{changed}}\)；待三个方法的
targets 都完成 serialization 和 tokenizer encoding 后，再从实际 supervised target
IDs 计算下面的 token/EOC/route-weighted ratios。不能在 targets 尚不存在时用估计
长度冒充实际 exposure。weighted counts 用两条独立路径复核：一条按
`presentation_id` join `views.jsonl` 的 `differs_from_map/is_A` 与 target-count
records；另一条直接重解析 serialized target 的 controls/raw token IDs、重建
boundary vector 并与 frozen MAP 比较。两条路径的每个 integer numerator 和
denominator 必须逐项相等。

$$
R_{\mathrm{changed}}^{AR,m}
=
\frac{
\sum_u\mathbf 1[B_u\ne B_u^{MAP}]n_{AR}^{m}(u)
}{
\sum_u n_{AR}^{m}(u)
},
$$

并以同一公式分别把 \(n_{AR}^{m}\) 换成 \(n_{EOC}\) 和 \(n_{route}\)，得到
\(R_{\mathrm{changed}}^{EOC}\) 与 \(R_{\mathrm{changed}}^{route}\)。TokMem 的
EOC/route ratio 记为 N/A，不做 \(0/0\)。

同时把 indicator 换成 \(\mathbf1[x_u\in\mathcal A]\)，报告
\(R_{\mathcal A}^{AR,m}\)、\(R_{\mathcal A}^{EOC}\) 和
\(R_{\mathcal A}^{route}\)，从而知道 ambiguity-bearing pool 实际贡献了多少
loss tokens 和 transition supervision。因为主 sampled posterior 固定
\(J=J_{\mathrm{MAP}}\)，同一 presentation 的 \(n_{EOC}\) 和 \(n_{route}\) 在
MAP/sampled 间应完全一致；同一 method 的 AR target 总数也要在序列未截断时一致。

令 \(n_g\) 为一个 template group 在完整三-epoch presentation manifest 中出现的
次数；另报：

$$
C_{\max}
=
\frac{\max_g n_g}{\sum_g n_g},
\qquad
\mathrm{ESS}_{group}
=
\frac{(\sum_g n_g)^2}{\sum_g n_g^2}.
$$

还要分别给 \(\mathcal G_A,\mathcal G_R\) 的 group 数、被选 group 数、
`min/median/max n_g`。primary hard test 要求 epoch 内每个 \(n_{g,e}\le1\)，所以
\(R_{\mathrm{changed}}\) 不能由同一 group 在一个 epoch 中重复堆高。

按 \(\mathcal A\) 的定义，balanced sampler 下
\(\mathbb E[R_{\mathrm{changed}}]\ge0.5\times0.25=0.125\)。两个预注册 data
schedules 都必须在**全部 presentations** 上实测
\(R_{\mathrm{changed}}\ge0.10\)；不能反复换 seed 直到通过。未达到时仍可报告
regularization 结果，但不能作为“模型接受了实质模糊边界监督”的 primary 证据。

### 7.2 piece ID 到 memory token

最终 \(V_{\hat K}\) 的 canonical tuples 按：

```text
mandatory first
then descending theta
then canonical UTF-8 lexicographic tie-break
```

分配稳定 `piece_id` 和 reserved memory token。所有方法共享同一映射。

如果一个 sampled partition 为：

```text
B = [a1+a2] [a3]
piece IDs = [v12, v3]
raw spans = [chunk1+chunk2, chunk3]
```

TokMem token-ID target：

```text
<m_v12> raw_span_12
<m_v3>  raw_span_3
<response_end>
```

EOC-only 和 TapMem：

```text
<m_v12> raw_span_12 <EOC>
<m_v3>  raw_span_3  <EOC>
<response_end>
```

`<EOC>` 后若下一个 supervised token 是 memory token，形成一个 later routing
example；最后一个 `<EOC>` 后是 response end，不伪造成 next-procedure target。

### 7.3 训练目标的准确口径

固定 induction posterior \(q_J\) 后，sampled setting 优化：

$$
\mathcal J_m(\phi)
=
\mathbb E_{B\sim q_J(B\mid x)}
\left[
\mathcal L_m(\operatorname{serialize}(x,B);\phi)
\right].
$$

每个 epoch 一个 FFBS sample 是该 expected loss 的 Monte Carlo estimator。对
TokMem/EOC-only，\(\mathcal L_m\) 是当前 response-token mean autoregressive CE；
TapMem 为：

$$
\mathcal L_{\mathrm{TapMem}}
=
\mathcal L_{\mathrm{AR}}
+\gamma\mathcal L_{\mathrm{route}}.
$$

这必须称为：

> fixed-posterior stochastic boundary-view augmentation / segmentation regularization。

它**不是**：

$$
-\log\sum_B q(B\mid x)P_\phi(y_B\mid instruction),
$$

也不是 TapMem 对 latent boundary 的 exact marginalization。procedure unigram EM
在预处理阶段结束，TapMem 不反向更新 \(q\)。

### 7.4 三种方法的参数与公平性

所有方法：

- frozen backbone；
- 相同 \(\hat K\)、piece IDs、raw pairs、view schedule；
- 相同 prompt、batch order、epochs、optimizer steps；
- 相同 max context/max new tokens；
- 相同 orthogonal memory-token initialization routine 和 model seed；
- 不使用 LoRA、adapter、retriever、planner 或 test-time training。

TokMem 只训练 memory-token parameters；EOC-only 额外训练 EOC；TapMem 再增加论文
原有 TCRA/logit-bias head。TCRA 只在 assistant start 和实际 EOC 后的
next-memory decision 位置工作，不修改普通 Bash token logits 的定义。

不存在“沿用默认或再调参”的二选一。正式配置在第一次 InterCode 候选模型调用前
固定为：

| field | Llama-1B primary | Llama-8B confirmation |
|---|---:|---:|
| backbone | `Llama-3.2-1B-Instruct` | `Llama-3.1-8B-Instruct` |
| epochs | 3 | 3 |
| memory/head LR | \(5\times10^{-3}\) | \(5\times10^{-3}\) |
| optimizer | AdamW, \(\beta=(0.9,0.999)\), \(\epsilon=10^{-8}\), weight decay 0 | same |
| schedule | linear to 0; warmup `total_steps // 10` | same |
| per-device batch / grad accumulation | 4 / 1 | 1 / 4 |
| effective batch | 4 | 4 |
| train dtype | bfloat16 | bfloat16 |
| train max length | 1024 tokens | 1024 tokens |
| epochs/checkpoint | final successful optimizer step of epoch 3 | same |
| decoding | greedy, `do_sample=false` | same |
| max new tokens per turn | 512 | 512 |
| inference context cap | 8192 tokens | 8192 tokens |
| TCRA route-loss weight \(\gamma\) | 0.1 | 0.1 |
| TCRA network / scale | linear / 1.0 | same |
| TCRA gradients | `detach=true`, `use_logit_train_add=true` | same |

这些值对应当前 maintained Llama-1B TokMem launcher 和
`main_sequential.py` 的 paper-method defaults；本实验不做 LR、epoch、\(\gamma\)、
scale、temperature 或 checkpoint search。训练不得静默截断：任一 serialized
training target 超过 1024 时构造失败并报告 overflow 数，不能按 method 分别丢弃。
非有限梯度导致 optimizer step 被跳过也将该 run 标为失败，不拿更早 checkpoint
替代。

inference history 超过 8192 时，保留 system prompt、原始 instruction 和最新
observation，只从最早开始删除完整的
`previous stripped action + observation` turn pair，直到 fits；所有方法使用同一
确定性 truncation manifest。prompt 严格复用 pinned InterCode Bash chat template，
不含示例、gold、procedure list 或 gold utility；唯一的 exact bytes 写入
`prompt.txt`。backbone revision/tokenizer hash、完整训练配置、prompt、history
truncation 和 runner code commit 一起写入 `neural_config.json` 并 hash。没有 dev
checkpoint selection，也不查看 NL2Bash dev neural score 来改变上述值。

必须报告每种方法：

- trainable parameter 数；
- supervised raw code token 数；
- memory token 数、EOC 数、route-loss site 数；
- truncation rate。

count-conditioned schedule 下，MAP 与 sampled 的后三项应逐样本相等。

`TapMem - EOC-only` 同时加入完整的 transition-conditioned linear routing head，
所以只支持“完整 TCRA component 有增量价值”。若要进一步声称收益来自 hidden-state
conditioning 而不是额外 routing capacity，必须另加同参数量、query-independent
static routing head；该机制 control 不影响本实验的 primary TapMem-vs-TokMem
结论。

## 8. InterCode 推理协议

### 8.1 测试时没有 segmenter

推理时输入只有：

- 原始用户 instruction；
- 之前实际执行的 stripped Bash actions；
- 官方 stdout/stderr/environment observation；
- 剩余 turn budget 和 `submit` 规则。

模型自由生成：

```text
memory → Bash span → [EOC → memory → Bash span]* → response end
```

不运行 unigram segmenter，不给 test piece ID、boundary、gold utility 或 gold program。

### 8.2 action detokenization

harness 只处理本轮 newly generated token IDs：

1. 找到第一段完整的 backbone response-end token sequence，在它之前截断；若在
   `max_new_tokens` 内没有完整 terminator，则保留全部 newly generated IDs 并记录
   `missing_terminator=true`；
2. 从截断后的前缀中按精确 token ID 过滤 reserved procedure-memory IDs 和 EOC ID；
3. 剩余 ordinary token IDs decode 成 Bash action。

response end 可能由多个 IDs 组成；禁止逐个 ID 在全文全局删除，也禁止保留首个完整
terminator 之后的文本。另禁止：

- decoded string 模糊替换；
- 自动补 quote、pipe、路径或 flag；
- 根据 gold 修复；
- parse 失败后让 harness 改写。

只有 stripped action 进入 InterCode Docker。训练或预处理中的 Bash 绝不在宿主机直接
执行；宿主机只允许 `bash -n`/parser 等无执行 syntax checks。

### 8.3 多轮历史和 reset

下一轮历史回放：

```text
actual stripped Bash action + official observation
```

不回放 memory/EOC 内部 trace，避免 TapMem 获得额外可见状态。每个
`task × method × data_seed × model_seed` 使用独立、完全 reset 的 container；
observation truncation、stop、submit 和 turn budget 全部固定。

同一 checkpoint 同时评测：

- single-turn；
- 官方 Try Again，最多 10 turns。

主结果是 10-turn；single-turn 只作 secondary。

## 9. 实验矩阵

### 9.1 Llama-1B 主矩阵

| Procedure/view setting | Frozen | TokMem | EOC-only | TapMem | seeds |
|---|---:|---:|---:|---:|---|
| no task training | ✓ | - | - | - | 1 deterministic decode |
| learned motifs, Fixed-MAP | - | ✓ | ✓ | ✓ | model `{40,41,42}` |
| learned motifs, count-conditioned posterior | - | ✓ | ✓ | ✓ | data `{1729,7919}` × model `{40,41,42}` |
| singleton-only atomic control | - | ✓ | - | ✓ | model `{40,41,42}` |
| uniform-lattice count-matched control | - | ✓ | - | ✓ | data `1729` × model `{40,41,42}` |
| full-posterior granularity stress | - | ✓ | - | ✓ | exploratory only：data `1729` × model `42` |

singleton-only control 强制每个 top-level atom 为一个 procedure，代表 reviewer 所说的
clean atomic setting。它只有 \(K_{\min}\) 个 active memory types，因而是 structural
diagnostic，不是 parameter-matched causal baseline；trainable parameter 数必须单独
报告，不能用它替代同 \(\hat K\) 下的 TokMem/EOC-only/TapMem 主比较，也不能把它
性能较低解释为 multi-atom motifs 的因果收益。variable-length memory 是否实际存在，
由 retained multi-piece 数、coverage 和 posterior mass 直接审计，不靠
singleton-only 的任务性能来“证明”。

### 9.2 Llama-8B confirmation

只确认最关键设置：

- learned motifs；
- count-conditioned posterior；
- TokMem / EOC-only / TapMem；
- data seed `1729`；
- model seeds `{40,41,42}`；
- 完整 200 题 single-turn 和 10-turn。

### 9.3 \(K\) sensitivity

完整 dev NLL curve 是必报的数据层 sensitivity。神经 sensitivity 在 1B、
count-conditioned setting 上使用：

$$
\max(K_{\min},\hat K-16),\quad
\hat K,\quad
\min(247,\hat K+16)
$$

去重后的容量，各跑 TokMem/TapMem，固定 data seed `1729`、model seed `42`。
该分析不能反向改变主 \(\hat K\)。

### 9.4 执行顺序

1. 数据/provenance/round-trip unit tests；
2. lexicon induction、\(K\) selection、人工 annotation sample manifest；
3. 在独立的 synthetic Docker suite 上做 engineering/model smoke test；
4. 1B Fixed-MAP 与 count-conditioned 主矩阵；
5. 1B controls 和第二 data schedule；
6. 8B confirmation；
7. 只读 structural diagnostics 和统计汇总。

synthetic smoke suite 只从 NL2Bash train/dev 的 utility/operator patterns 构造临时
文件和预期 stdout/file-state，共 24 题，覆盖 response terminator、memory/EOC
stripping、stdout、stderr、nonzero exit、container reset、history replay 和 turn
limit；它不包含 bundled InterCode 200 题的 instruction、gold 或 filesystem。
允许在这个 suite 上修 runner bug，但每次修改都生成新 runner/prompt hash。

官方 repository 的 `data/nl2bash/test_queries.json` 是另一个 **distinct 24-task
file**，不是四个 fs JSON 组成的 200 题子集。本地逐项审计得到它与 200 题的 raw
exact query/gold/pair overlap 均为 0；固定文件 SHA-256 为
`d24a7a1eb61c2621c48a42f942d08f6aa02066630ab49c2a07de2530a226e0aa`。本方案不把
该 24 题混入 200 题 primary，也不存在“200-24=176”的口径。

若使用这个 distinct 24-task file 做 development/smoke，必须单独披露、pin 上述
hash，并单独报告其结果；它不能和 200 题合并。主方案仍优先使用上一段自建 synthetic
suite 调 runner。最终 `prompt.txt`、runner 和 `neural_config.json` 的 hash 必须
早于第一条 200-task InterCode candidate-model generation。

## 10. 指标和统计

### 10.1 Primary

完整 200 题、10-turn：

- official binary Success Rate；
- primary contrast
  `TapMem_count_posterior - TokMem_count_posterior`。

设 primary task 数为 \(N=200\)，task \(i\)、data seed \(d\)、matched model seed
\(s\) 的二元结果为
\(y_{i,d,s}^{m}\)，其中 \(D=2,S=3\)。先在每个 task 内平均：

$$
d_i
=
\frac{1}{DS}
\sum_{d=1}^{D}\sum_{s=1}^{S}
\left(
y_{i,d,s}^{Tap}
-
y_{i,d,s}^{Tok}
\right).
$$

再以 task 为唯一 bootstrap unit，对 \(N\) 个 \(d_i\) 做 10,000 次 paired
bootstrap，报告 \(\bar d\) 的 percentile 95% CI。不能把
`tasks × data seeds × model seeds` 当作独立样本。这个区间的 estimand 明确是六对
预注册 checkpoints 在这组 tasks 上的平均效果，只覆盖 task sampling uncertainty，
不声称覆盖任意训练初始化或任意 posterior schedule。

另对每个 matched run 计算：

$$
\delta_{d,s}
=
\frac1N\sum_{i=1}^{N}
\left(y_{i,d,s}^{Tap}-y_{i,d,s}^{Tok}\right),
$$

并对每个 data schedule 计算
\(\bar\delta_d=S^{-1}\sum_s\delta_{d,s}\)。primary 支持条件同时要求：

$$
\operatorname{CI}_{task,0.025}(\bar d)>0,
\qquad
\bar\delta_{1729}>0,
\qquad
\bar\delta_{7919}>0.
$$

两个 data seeds 太少，不用 multilevel bootstrap 假装精确估计 schedule population
variance；相反，完整报告六个 \(\delta_{d,s}\) 和两个
\(\bar\delta_d\)，把跨 schedule 同方向作为正式 stability 条件。

同时报告：

- 六个 matched model/data run delta 和两个 data-schedule mean delta；
- seed mean/std；
- 逐 task result JSON；
- paired win/loss/tie counts。

### 10.2 Secondary end-to-end metrics

- single-turn SR；
- 官方 continuous reward；
- inadmissible/error action rate；
- parse rate；
- 平均 turns；
- first-action success；
- shared-first-turn-failure recovery：

  ```text
  只在 TokMem 与 TapMem 第一轮都未成功的共同 task 上，
  比较到第 10 轮是否恢复成功。
  ```

  该指标不是 execution-feedback 的反事实验证。若论文要作“更会使用 observation”
  的机制声称，另跑 1B、data seed `1729`、model seed `42` 的
  observation-masked control：保留相同 previous stripped actions、turn budget 和
  observation token budget，但把每轮 stdout/stderr/file observation 替换为同长度的
  `<OBS_MASK>` tokens；比较正常与 masked 的 TapMem-vs-TokMem delta interaction。
  不跑该 control 就只写“在提供 execution feedback 的环境中 SR 更高”。

- extraneous file-system changes；
- Fixed-MAP/sample interaction；
- EOC-only - TokMem 和 TapMem - EOC-only。

多个 secondary contrasts 使用 Holm correction，或明确标记为 exploratory；不能把
其中最有利的一项事后升格为 primary。

### 10.3 词典和训练-view construct metrics

在 decontaminated train/dev 报告：

$$
\mathrm{multi\_segment\_rate}
=
\frac{
\sum_x\sum_{(i,j,v):|v|>1}\gamma_x(i,j,v)
}{
\sum_x\sum_{(i,j,v)}\gamma_x(i,j,v)
},
$$

$$
\mathrm{multi\_atom\_coverage}
=
\frac{
\sum_x\sum_{(i,j,v):|v|>1}|v|\gamma_x(i,j,v)
}{
\sum_x L_x
}.
$$

另报：

- retained singleton/multi-piece 数；
- 每个 piece 的 \(\theta\)、expected count、group support、长度；
- top pieces 的 raw occurrence examples；
- whole-command piece 数和 posterior mass；
- \(P(J=1)\)、\(\mathbb E J\)、MAP \(J\) 分布；
- count-conditioned view entropy、MAP mass、distinct-view 数；
- 自然 pair prevalence 下的 expected change rate；
- balanced presentation schedule 下的 \(R_{\mathrm{changed}}\)、
  \(R_{\mathrm{changed}}^{AR,m/EOC/route}\) 和
  \(R_{\mathcal A}^{AR,m/EOC/route}\)；
- `<RARE>/<UNK>` atom 和 posterior 占比；
- 三种 EM initialization 的稳定性。

这些指标回答“实验中是否真的有可变长 procedure 和 alternative boundaries”，不能用
silhouette score。

## 11. 人工验证边界确实不是唯一的

### 11.1 抽样和标注协议

在拟合 unigram posterior 之前，从 decontaminated NL2Bash dev 的 strict-flat、
\(L\ge3\) commands 中按 stable sample-ID hash 随机抽 100 条；不足 100 条则全部使用。
抽样不按模型 entropy 挑选。

三名熟悉 Bash 的标注者独立看到：

- 完整 instruction；
- 完整 raw command；
- parser 给出的 syntax-safe gaps；
- 操作性定义：“在什么位置，一个可复用的执行阶段结束，下一阶段开始处理新的中间
  目标？”。

不提供 memory vocabulary、MAP、posterior、建议段数或其他标注者答案。允许：

- 不切，整条程序一个阶段；
- 选择任意多个合法 gaps；
- 写一句简短 rationale。

这 100 条只作验证，不能训练 procedure model、调温度、选 \(K\) 或选 checkpoint。

### 11.2 明确计算哪些指标

设 gap \(t\) 的三人投票数为 \(r_t\in\{0,1,2,3\}\)。

- split-vote rate：

  $$
  \frac{\#\{t:r_t\in\{1,2\}\}}{\#\{t\}};
  $$

- whole-partition unanimous rate；
- 三对 annotators 的 boundary precision/recall/F1，先 command-macro 再平均；
- 所有 gap 上的 Fleiss' \(\kappa\)；
- 每人/每条 command 的 segment-count distribution；
- frozen count-conditioned model boundary probability
  \(p_t^{(J_{\mathrm{MAP}})}\) 相对 vote fraction \(r_t/3\) 的 Brier score：

  $$
  \mathrm{Brier}
  =
  \frac1G\sum_t(p_t-r_t/3)^2;
  $$

- 5-bin reliability diagram；
- model gap entropy \(h_t\) 与 human vote entropy 的 Spearman correlation。

人工 disagreement 证明的是“在该操作性 procedure 定义下，人们存在粒度/边界分歧”，
不证明 unigram posterior 就是人的真实 posterior，也不证明 InterCode 每个任务都模糊。

为了检查 unigram posterior 的不确定位置是否至少与人类分歧对齐，预注册两个不使用
motif lexicon 的 baseline。对 command \(c\) 的 gap \(t\)，目标记为
\(y_{ct}=r_{ct}/3\)：

1. leave-one-command-out global-rate predictor：

   $$
   p_{ct}^{const}
   =
   \frac{\sum_{c'\ne c}\sum_t y_{c't}}
   {\sum_{c'\ne c}G_{c'}};
   $$

2. legal-partition-uniform predictor：在该 command 的
   \(2^{L_c-1}\) 个 syntax-safe boundary bit vectors 上均匀分布，因此每个合法 gap
   的 marginal \(p_{ct}^{uniform}=0.5\)。

令实际标注 command 数为 \(C\le100\)。先对每条 command 分别平均其 gaps 得到
\(\mathrm{Brier}_{model,c}\)、
\(\mathrm{Brier}_{const,c}\) 和
\(\mathrm{Brier}_{uniform,c}\)，避免长命令支配结果。以 100 条 command 为 cluster
做 10,000 次 paired bootstrap，主 construct contrast 为：

$$
D_{\mathrm{Brier}}
=
\frac1C\sum_c
\left(
\mathrm{Brier}_{const,c}
-
\mathrm{Brier}_{model,c}
\right).
$$

要求 \(D_{\mathrm{Brier}}\) 的 95% CI 下界大于 0；uniform baseline 只作第二参考。
Spearman 在全部 gaps 上计算，但 95% CI 仍通过 \(C\) 个 command 的 cluster bootstrap
重采样、每次重新计算 correlation，要求其下界大于 0。这样检验的是“模型高不确定的
gap 是否也是人类更分歧的 gap”，而不只是两边各自都有 entropy。

预注册 construct-validity 警戒线：

- split-vote gaps 少于 20%；或
- 超过 80% commands 的完整 partition 三人完全一致。

触发任一项时，只能把 sampled setting 称为 stochastic segmentation regularization，
不能声称数据呈现了强天然边界歧义。

即使上述 disagreement 阈值通过，只要 unigram Brier 没有显著优于
leave-one-command-out constant baseline，或 Spearman CI 下界不大于 0，全文也必须
把 sampled views 降格称为 **corpus-induced stochastic tokenizations**，不能称
“plausible/reasonable alternative procedure boundaries”。

## 12. InterCode structural diagnostics 的正确口径

完整 200 题始终是 primary。下面两个 subset 是查看 released gold AST 后定义的
**gold-informed but pre-inference-frozen exploratory manifests**：

1. 123 个 gold reference 是 strict flat pipeline 的 task IDs；
2. 其中 77 个 \(L\ge3\)、至少有两个 internal gaps 的 task IDs。

这两个 subset 只能称为：

> reference program 可被当前 flat-pipeline atomizer 完整分析的任务。

不能称为“真正模糊”“更开放”或“更难”。还需报告其余任务在文件系统、root kind、
gold atom length 和 base difficulty 上的差异。

它们不能用于选模型、调 prompt/\(K\)、增加完整 200 题 primary 的可信度，或被包装成
独立外部分布验证。它们唯一的价值是解释：当前 flat-pipeline atomizer 能在 released
reference program 的哪一部分上计算结构统计。

训练词典冻结后，可对这 123 个 gold reference programs 计算
\(A_{\mathrm{gap}}\)、\(H_{\mathrm{seg}}\) 和 count-conditioned entropy。字段必须命名：

```text
reference_program_segmentation_entropy
```

它不是 task ambiguity：同一任务的另一条正确程序可能有完全不同的结构。若按 entropy
作 low/mid/high 诊断，bin boundaries 在模型推理前冻结，并同时报告/控制 gold atom
count、rare-piece rate、file-system group。高熵组结果只作 exploratory secondary。

## 13. 失败判据和结论边界

下列数值阈值只用于在看 InterCode 结果前判断“构造是否真的包含足够的 multi-piece 和
alternative boundaries”，不用于选择最有利的 \(K\) 或 model checkpoint。

以下结果必须如实缩小或放弃结论：

1. sequence-only exact 或 edit-distance positive-control 任一未达到 23/23，
   200 个 InterCode task 中任一 concrete `resolved_source_group_ids` 为空，任一
   resolved source 或 official TEST source component 进入训练/选择 artifact，
   或去泄漏后仍有完整 command/template overlap：实验无效；
2. 触发任一预注册 vocabulary-richness 条件：retained multi-atom pieces 少于 25、
   train `multi_atom_coverage < 0.25`、`<RARE>/<UNK>` atom occurrence rate 大于
   0.10，或在原始 \(L\ge2\) commands 中
   \(P(J_{\mathrm{MAP}}\ge2)<0.50\)。此时没有构成足够的 variable-length
   transition setting；
3. \(|\mathcal G_A|<
   \max(100,\lceil0.20|\mathcal G_{L\ge3}|\rceil)\) 或
   \(|\mathcal G_R|<100\)；或在 \(L\ge3\) commands 中，满足
   \(J_{\mathrm{MAP}}\ge2\) 且
   \(\exp H(q_J)\ge1.5\) 的比例低于 0.25，或者任一预注册 sampled schedule 在
   全部 training presentations 上的 \(R_{\mathrm{changed}}<0.10\)。此时主设置没有
   提供足够的实际 boundary variation；
4. 人工标注触发第 11.2 节 disagreement 警戒线：不能声称强天然 ambiguity；若
   Brier improvement CI 或 Spearman CI 未通过，则只能称 corpus-induced
   stochastic tokenizations；
5. primary \(\Delta_{\mathrm{amb}}\) 的 paired 95% CI 下界不大于 0，或任一
   data schedule 的三-seed mean delta 不大于 0：没有稳定证据证明 TapMem 在该
   模糊监督下优于 TokMem；
6. TapMem 只在 Fixed-MAP 提升、sampled setting 不提升：
   方法可能依赖稳定边界；
7. EOC-only 解释全部收益且 `TapMem - EOC-only` 不为正：
   只能支持显式 completion，不能支持完整 TCRA component 的增量价值；
8. 只有 routing/EOC F1 提高而官方 SR 不提高：
   只能报告机制改善，不能报告开放式任务泛化；
9. primary TokMem 和 TapMem 的六-run mean SR 都小于 0.05，或都大于 0.95：
   存在预注册的地板/天花板，比较判别力不足；
10. 提升方向随 data schedule、model seed 或 \(K\) sensitivity 翻转：
    结论不稳定；
11. multi-turn 增益只来自 first action，shared-failure recovery 不改善，或者没有
    observation-masked/shuffled 反事实 control：不能作“更会利用 execution
    feedback”的因果声称，只能说在提供 feedback 的环境中 SR 更高；
12. uniform-lattice 与 corpus posterior 结果相同或更好：
    只能说收益不依赖 retained lexicon 内的 unigram \(\theta\) weighting，不能扩大
    解释成“任意 syntax-safe boundary noise 都有效”。

还应承认公开 NL2Bash/InterCode 可能出现在 backbone 预训练语料中。相同 backbone 的
paired method comparison 减轻方法间差异污染，但绝对 SR 不能解释为严格 unseen-data
能力。

## 14. 实现 artifact 和 hard tests

建议给该 track 单独目录，不把 Bash 特例塞入现有 APIGen `dataset.py`：

```text
compositional/bash_latent/
  build_raw_manifest.py
  atomize_bash.py
  induce_unigram_procedures.py
  build_view_schedule.py
  dataset.py
  intercode_runner.py
  evaluate.py
  test_atomizer.py
  test_unigram_dp.py
  test_serialization.py
```

训练/atomization 环境至少新增并 pin `bashlex==0.18`；当前 `requirements.txt`
尚未包含它。官方 filter/split 不在 Python 3.10 训练环境临时 monkeypatch，而使用
第 3.1.1 节已 pin digest 的 Python 3.9 legacy preprocessing container，其中
`nlp_tools` 与 legacy `bashlint` commits 同时固定。

### 14.1 Artifact schema

`raw_pairs.jsonl`：

```text
schema_version
source_repo / source_commit / source_file / source_line
sample_id
instruction_raw / command_raw
instruction_sha256 / command_sha256
official_split / derived_split / source_group_id / template_group_id
decontamination_status / matched_intercode_task_ids
provenance_candidates / adjudicator_labels / final_deny_group_ids
```

`official_split_replay.json`：

```text
nl2bash_commit / filter_script_sha256 / split_script_sha256
nlp_tools_commit / bashlint_commit / python_version / container_digest
raw_file_sha256 / filtered_file_sha256 / six_split_file_sha256
raw_pair_multiset_sha256 / filtered_pair_multiset_sha256
train_pair_multiset_sha256 / dev_pair_multiset_sha256 / test_pair_multiset_sha256
raw=12607 / filtered=9305 / filtered_out=3302
official_train=8090 / official_dev=609 / official_test=606
normalized_nl_groups=8412
random_tokens_sha256=5f770f78fe1534288b8872d8a32179c36a7ebc5413d0e3eba748f99d7b4b96b0
source_line_to_official_split_manifest_sha256
```

`source_and_template_groups.json`：

```text
graph_version
sample_id / normalized_instruction / exact_command / template_keys
source_group_id / template_group_id
source_edge_reasons / template_edge_keys
source_component_members / template_component_members
source_component_size_histogram / template_component_size_histogram
official_test_source_group_ids
source_graph_manifest_sha256 / template_graph_manifest_sha256
```

`intercode_provenance.jsonl`：

```text
task_id / official_split
candidate_rule_version / candidate_rows / candidate_group_ids
exact_pair_hits / exact_instruction_hits / exact_command_hits / template_hits
adjudicator_1_labels / adjudicator_2_labels
resolution_mode={exact,template,manual,unresolved}
resolved_source_group_ids
all_retrieved_candidate_group_ids / final_deny_group_ids
resolved_component_absent_from_train_dev_lexicon_K
record_sha256
```

`provenance_positive_control.json`：

```text
23 exact-pair task IDs
exact_channels_masked=true / template_channels_masked=true
original_sequence_untruncated_retrieved_group_ids
mutated_sequence / sentinel_absent / exact_sequence_hits=0
edit_distance_untruncated_retrieved_group_ids
known source group ID / recalled_by_exact_sequence / recalled_by_edit_distance
exact_sequence_recall=23/23 / edit_distance_recall=23/23
retrieval_config_hash
```

`parsed_atoms.jsonl`：

```text
sample_id
parser_name / parser_version / wheel_sha256
parse_status / exclude_reason
root_kind
atoms[
  atom_id, node_path, node_kind, char_range, raw_core,
  incoming_connector_kind,
  static_utility_head, normalized_static_utility_head,
  head_status={STATIC,DYNAMIC,NONE,RARE},
  canonical_signature
]
gaps[
  gap_id, char_range, raw_text, operator_kind
]
prefix / suffix
raw_roundtrip_sha256
```

`tokenized_base_chunks.jsonl`：

```text
sample_id
tokenizer_name / tokenizer_revision / vocab_sha256
base_chunks[
  atom_id, raw_chunk_sha256, token_ids
]
concatenated_ordinary_token_ids_sha256
decoded_raw_sha256
```

`procedure_lexicon.json`：

```text
config_hash
K_min / K_hat / K_max
rho
piece_id -> canonical signature tuple
theta / expected_count / group_df / proper_df / length
mandatory flag / removal round
train likelihood / dev NLL
map_comparator_version
```

`training_presentations.jsonl`：

```text
presentation_id
presentation_seed / epoch / effective_batch_id / position_in_batch
pool={GA,GR} / template_group_id / sample_id / within_epoch_occurrence_index
J_MAP / qJ_MAP_mass / alternative_mass
group_occurrences_this_epoch / group_occurrences_all_epochs
```

`views.jsonl`：

```text
presentation_id / config_hash / data_seed
epoch / sample_id / within_epoch_occurrence_index / is_pair_A
view_type
boundary_bit_vector
J
piece_ids
raw_char_ranges
log_probability / conditional_log_probability
map_boundary_bit_vector / differs_from_map
view_id = SHA256(config_hash + data_seed + presentation_id + boundaries)
```

`serialized_targets.jsonl`：

```text
presentation_id / data_seed / method / sample_id
serialized_token_ids_sha256 / target_token_ids
reparsed_piece_ids / reparsed_boundary_bit_vector
n_AR / n_EOC / n_route
ordinary_token_ids_sha256 / truncation_status
```

`neural_config.json`：

```text
backbone_revision / tokenizer_revision / vocabulary_hash
prompt_sha256 / runner_commit / history_truncation_rule
optimizer / lr / epochs / batch / grad_accum / scheduler / warmup
gamma / logit_bias_scale / logit_bias_network / detach / use_logit_train_add
train_max_length / inference_context_cap / max_new_tokens / decoding
presentation_manifest_hash / view_manifest_hash / checkpoint_rule
frozen_before_first_intercode_inference_timestamp
```

`ambiguity_exposure.json`：

```text
data_seed / total_training_presentations
natural_pair_expected_change_rate
pair_A_presentation_rate / G_A_presentation_rate
G_A_count / G_R_count / G_L_ge_3_count / M_groups_per_pool_per_epoch
group_count_min_median_max / max_group_share / group_ESS
R_changed
R_changed_AR_by_method / R_changed_EOC / R_changed_route
R_A_AR_by_method / R_A_EOC / R_A_route
distinct_views_per_repeated_sample / cross_data_seed_view_disagreement_rate
```

`human_boundary_validation.json`：

```text
annotation_manifest_hash / command_id / legal_gap_ids
three_boundary_bit_vectors / rationales
vote_fractions / model_count_conditioned_boundary_probabilities
loo_constant_probabilities / uniform_partition_probabilities
command_macro_brier_model / constant / uniform
cluster_bootstrap_seed / Brier_improvement_CI / Spearman_CI
```

`test_structure_manifest.json`：

```text
intercode_commit / task_file_hashes
task_id
root_kind / reference_atom_count
strict_flat_pipeline flag
reference_program_segmentation_entropy
manifest_created_after_lexicon_hash
```

### 14.2 必须自动失败的检查

- source line 数、hash、stable ID；
- legacy official replay 的 raw/filtered/split counts 分别为
  12,607/9,305/(8,090,609,606)，normalized groups 为 8,412，并且所有 ordered-file
  SHA、pair-multiset SHA 和 12,607-row split manifest 与冻结 reference 相同；
- 从 raw equality keys 独立重建 source/template graphs，要求 member lists、
  component IDs、edge-reason counts 和 graph hashes 与 artifact 完全相同；
- NL2Bash official `TEST` 或 `FILTERED_OUT` rows 进入
  induction/train/dev/\(K\)-selection 的计数均为 0；
- 含任一 official `TEST` row 的 source component，其任何 TRAIN/DEV member 进入
  induction/train/dev/lexicon/\(K\)-selection 的计数均为 0；
- exact/template channels 同时 masked 时，sequence-only positive-control 的
  original-sequence 与 forced-edit-distance branches 都在未截断 candidate groups
  中召回已知 source 23/23；
- 200/200 InterCode provenance records 的 concrete
  `resolved_source_group_ids` 非空，且其所有 connected-component rows 进入
  induction/train/dev/lexicon/\(K\)-selection 的计数均为 0；
- 最终选取的 top-level atom ranges 按 source order 且两两不重叠；完整 AST 的
  parent/child ranges 允许嵌套，只检查 child 被合法 parent 包含及 heredoc guards；
- prefix + atom cores + full gaps + suffix 逐字符恢复 raw；
- UTF-8 bytes/hash 恢复；
- 每个 view 覆盖每个 atom 恰好一次，无 overlap/hole；
- 每个 piece ID 的 canonical sequence 与 covered atoms 一致；
- tokenizer 控制-ID删除后逐字恢复 raw；
- 同一 sample 的所有 views 删除 memory/EOC 后，ordinary token-ID sequence 与
  `tokenized_base_chunks.jsonl` 逐项相同；
- `bashlex` reparse 和 `bash -n -c` 只作 syntax check；
- 在小型人工序列上穷举全部 partitions，并与 DP 的
  \(P(x)\)、arc posterior、boundary marginal、entropy 对齐；
- 构造含等分 partitions 且 \(J\) 相同/不同的 toy lattices，确认穷举、Viterbi、
  schedule builder 都按同一 MAP comparator 选择较小 \(J\)、随后 boundary bits、
  随后 piece IDs；
- 检查
  \(\alpha[L]=\beta[0]\)；
- 检查
  \(\mathbb E J=1+\sum p_t\)；
- FFBS 采样频率在固定 100k toy samples 上落入理论 posterior 的统计容差；
- count-conditioned samples 的 \(J\) 永远等于 \(J_{\mathrm{MAP}}\)；
- TokMem/EOC-only/TapMem 同 data seed 的 view IDs 完全一致；
- `presentation_id` 在三-epoch manifest 中唯一；
  `(presentation_id,data_seed)` 在 views 中唯一；
  `(presentation_id,data_seed,method)` 在 serialized targets 中唯一；按这些键
  inner-join 后无缺失或多重匹配；
- 每个 effective batch 的 \(\mathcal G_A/\mathcal G_R\) presentation 数严格为 2/2，
  每个 template group 在同一 epoch 最多出现一次，
  Fixed-MAP 与 sampled 的 presentation IDs 完全一致；
- 用 `(views flags JOIN target-count records)` 和“直接重解析 serialized targets”
  两条路径重算 token/EOC/route-weighted exposure，要求每个 numerator/denominator
  逐整数相等；unweighted \(R_{\mathrm{changed}}\) 另从 views 直接重算；
- train/dev/InterCode deny keys 零 overlap；
- 第一次 InterCode candidate inference 的 timestamp 晚于 lexicon、prompt、
  runner、presentation、views 和 neural-config hashes；
- test structural manifest 的创建时间和 hash 晚于已冻结 lexicon，但早于 inference。

heredoc、multiple roots、leading reserved word、nested substitution、compound、`|&`、
trailing comment/`&` 都应有显式 unit tests。任何训练 command 都不能在宿主 shell
执行；正式行为评测只在 reset 后的 InterCode Docker 中进行。

## 15. 伪代码

### 15.1 Procedure induction

```python
official_replay = replay_official_filter_then_split(
    raw_nl2bash,
    num_utilities=100,
    random_seed=100,
    num_folds=12,
    pinned_legacy_container=True,
)
assert official_replay.counts == {
    "raw": 12607,
    "filtered": 9305,
    "filtered_out": 3302,
    "train": 8090,
    "dev": 609,
    "test": 606,
    "normalized_groups": 8412,
}
assert_matches_frozen_ordered_and_multiset_hashes(official_replay)

raw_nl2bash = attach_official_split(
    raw_nl2bash,
    official_replay.source_line_manifest,
)
groups = build_source_and_template_graphs_v1(
    raw_nl2bash,
    instruction_key="official_normalized_instruction",
    command_key="raw_exact_without_storage_lf",
    template_keys=("bashlint_full", "bashlex_fallback"),
)
raw_nl2bash = attach_frozen_group_ids(raw_nl2bash, groups)
official_test_group_denylist = {
    row.source_group_id
    for row in raw_nl2bash
    if row.official_split == "TEST"
}

positive_control = run_sequence_only_positive_controls(
    known_exact_pairs=23,
    raw_nl2bash=raw_nl2bash,
    mask_exact_channels=True,
    mask_template_channels=True,
    mutation="replace_first_utility_with_task_specific_unseen_sentinel",
)
assert positive_control.original_sequence_source_recall == (23, 23)
assert positive_control.mutated_sequences_have_zero_exact_hits
assert positive_control.edit_distance_source_recall == (23, 23)

provenance = resolve_all_200_intercode_sources(raw_nl2bash)
assert len(provenance) == 200
assert all(
    p.resolution_mode in {"exact", "template", "manual"}
    and len(p.resolved_source_group_ids) > 0
    for p in provenance
)
intercode_denylist = union(
    p.all_retrieved_candidate_group_ids
    | p.resolved_source_group_ids
    for p in provenance
)
full_source_denylist = (
    official_test_group_denylist | intercode_denylist
)

official_train_dev = [
    row for row in raw_nl2bash
    if row.official_split in {"TRAIN", "DEV"}
]
assert not any(
    row.official_split in {"TEST", "FILTERED_OUT"}
    for row in official_train_dev
)

train, dev = build_grouped_decontaminated_split(
    official_train_dev,
    full_source_denylist,
    component_key="source_group_id",
)
assert count_official_test_or_filtered_rows(train, dev) == 0
assert official_test_source_component_overlap(
    official_test_group_denylist,
    train=train,
    dev=dev,
) == 0
assert resolved_source_component_overlap(
    provenance,
    train=train,
    dev=dev,
) == 0

for pair in train + dev:
    parsed = strict_flat_atomize(pair.command_raw)
    assert roundtrip(parsed) == pair.command_raw

alphabet = build_train_atom_alphabet(train, rare_group_df=2)
train_sequences = canonicalize(train, alphabet)
dev_sequences = canonicalize(dev, alphabet)

V = mandatory_singletons(alphabet)
V += {
    v for v in repeated_contiguous_types(train_sequences)
    if group_df(v) >= 2 and proper_group_df(v) >= 1
}

paths_by_init = []
for init in ("group_df", "uniform", "group_df_times_length"):
    V_i = V.copy()
    theta_i, rho_i = run_em(V_i, init, train_sequences)

    while len(V_i) > 247:
        delta = one_step_deletion_scores(
            V_i, theta_i, rho_i, train_sequences
        )
        remove = floor_20_percent_at_least_one_without_crossing_247(delta)
        V_i = V_i - remove
        theta_i = renormalize_content_only(theta_i, remove)
        theta_i, rho_i = run_em(
            V_i, theta_i, rho_i, train_sequences
        )

    models_i = {}
    while len(V_i) >= K_min:
        models_i[len(V_i)] = (V_i, theta_i, rho_i)
        if len(V_i) == K_min:
            break
        v = argmin_one_step_deletion_score(
            V_i, theta_i, rho_i, train_sequences
        )
        V_i = V_i - {v}
        theta_i = renormalize_content_only(theta_i, {v})
        theta_i, rho_i = run_em(
            V_i, theta_i, rho_i, train_sequences
        )
    paths_by_init.append(models_i)

models_by_K = select_highest_train_map_objective_path_at_each_K(
    paths_by_init
)

K_hat = paired_one_se_select(models_by_K, dev_sequences)
frozen_lexicon, theta, rho = freeze_and_hash(
    models_by_K[K_hat]
)
```

### 15.2 Count-matched view generation

```python
def map_order_v1(candidate, incumbent):
    # Return True iff candidate replaces incumbent.
    # Scores are accumulated in a fixed arc order as np.float64.
    if candidate.score > incumbent.score:
        return True
    if candidate.score < incumbent.score:
        return False
    return (
        candidate.J,
        tuple(candidate.boundary_bits),
        tuple(candidate.piece_ids),
    ) < (
        incumbent.J,
        tuple(incumbent.boundary_bits),
        tuple(incumbent.piece_ids),
    )


map_info = {}
for sample in train:
    lattice = build_piece_lattice(sample.atom_signatures, frozen_lexicon)
    B_map = viterbi(
        lattice,
        theta,
        rho,
        comparator=map_order_v1,  # max score, then min J/bits/piece IDs
    )
    J_map = len(B_map)
    qj_map_mass = conditioned_partition_probability(
        lattice,
        B_map,
        theta=theta,
        rho=rho,
        piece_count=J_map,
    )
    map_info[sample.id] = {
        "B_map": B_map,
        "J_map": J_map,
        "qj_map_mass": qj_map_mass,
        "n_same_count_paths": count_lattice_paths(
            lattice,
            piece_count=J_map,
            stop_after=2,
        ),
    }

pair_A = {
    sample.id for sample in train
    if map_info[sample.id]["J_map"] >= 2
    and map_info[sample.id]["n_same_count_paths"] >= 2
    and 1.0 - map_info[sample.id]["qj_map_mass"] >= 0.25
}
rows_by_group = group_rows_by_template(train)
G_A = {
    g for g, rows in rows_by_group.items()
    if any(row.id in pair_A for row in rows)
}
G_R = set(rows_by_group) - G_A
G_L_ge_3 = {
    g for g, rows in rows_by_group.items()
    if any(len(row.atom_signatures) >= 3 for row in rows)
}
assert len(G_A) >= max(100, ceil(0.20 * len(G_L_ge_3)))
assert len(G_R) >= 100

M = 2 * floor(min(len(G_A), len(G_R)) / 2)

presentations = make_group_balanced_presentations_without_replacement(
    train=train,
    pair_A=pair_A,
    G_A=G_A,
    G_R=G_R,
    groups_per_pool_per_epoch=M,
    epochs=3,
    effective_batch_size=4,
    A_per_batch=2,
    presentation_seed=314159,
)
assert max_group_occurrences_in_any_epoch(presentations) == 1
assert all_unique(u.presentation_id for u in presentations)

for data_seed in (1729, 7919):
    for u in presentations:
        sample = train[u.sample_id]
        B_map = map_info[sample.id]["B_map"]
        J_map = map_info[sample.id]["J_map"]
        rng = PCG64(sha256_128(
            data_seed,
            sample.id,
            u.epoch,
            u.within_epoch_occurrence_index,
        ))
        B = ffbs_conditioned_on_count(
            lattice=build_piece_lattice(
                sample.atom_signatures, frozen_lexicon
            ),
            theta=theta,
            rho=rho,
            piece_count=J_map,
            rng=rng,
        )
        assert len(B) == J_map
        assert strip_controls(serialize(sample, B)) == sample.command_raw
        save_view(
            data_seed,
            u,
            B,
            B_map,
            presentation_id=u.presentation_id,
            is_pair_A=(sample.id in pair_A),
        )

    view_exposure = recompute_boundary_change_from_views(data_seed)
    assert view_exposure.R_changed >= 0.10

    serialized_by_method = {}
    for method in ("tokmem", "eoc_only", "tapmem"):
        serialized_by_method[method] = serialize_and_tokenize_all_targets(
            presentations=presentations,
            views=load_views(data_seed),
            method=method,
            tokenizer=frozen_tokenizer,
        )
        save_serialized_targets(
            data_seed,
            method,
            serialized_by_method[method],
        )

    joined_counts = compute_weighted_exposure_by_join(
        views=load_views(data_seed),
        target_count_records=load_saved_target_token_counts(data_seed),
    )
    reparsed_counts = compute_weighted_exposure_by_reparsing_targets(
        serialized_by_method,
        map_info=map_info,
        pair_A=pair_A,
    )
    assert_integer_numerators_and_denominators_equal(
        joined_counts,
        reparsed_counts,
    )
```

## 16. 为什么这套方法真正回应了 reviewer

它不再假设一个 API/tool call 就是一项给定 procedure：

- base atoms 只用于保证 Bash 不被切坏；
- procedure type 是可变长度、train-only、数据诱导的 repeated motif；
- 同一命令可能有多个 posterior-supported partitions；
- 主 sampling 在监督量完全相同的情况下只改变 boundary placement；
- 测试时模型看不到任何 partition，也可以生成与参考不同的 Bash action；
- 结果按环境输出/状态而不是 memory sequence exact match 评分。

TapMem 的基础思想没有改变：

- memory 仍是 tokenized procedural memory；
- EOC 仍显式标记当前 memory-controlled span 完成；
- TCRA 仍只在 memory decision boundary 修正 procedure-token logits；
- backbone 仍冻结；
- 改变的只是 procedure inventory 和训练序列不再由干净 atomic tool calls 直接给出。

如果 sampled setting 的 TapMem 相对 TokMem 提升且通过 primary CI，同时词典审计、
view entropy 和人工 disagreement 显示实际存在非退化多边界，那么这就是对两条评审
意见的直接证据。如果任何一环不成立，应按第 13 节缩小结论。

## 17. 其他场景的定位

| 场景 | 能补充什么 | 为什么不替代主实验 |
|---|---|---|
| [AppWorld](https://aclanthology.org/2024.acl-long.850/) | 开放代码、多 app、state-based tests；可对 Python AST statement motifs 做同样 induction | train solutions 少、环境和 1B/8B 成本高 |
| [APPS](https://arxiv.org/abs/2105.09938) | 非 agent 的跨场景证据；对 Python AST statement/basic-block motifs 训练，pass@1 评测 | 没有 execution-feedback agent loop |
| [SWE-Gym](https://github.com/SWE-Gym/SWE-Gym) / SWE-bench | 真实软件工程中定位、编辑、测试阶段高度模糊 | 小冻结模型地板和容器成本风险高 |
| [WebLINX](https://github.com/McGill-NLP/weblinx) / Mind2Web | 网页轨迹阶段边界模糊 | 原评测偏 next-action，底层操作仍较原子，最终状态评价不统一 |

若 InterCode 工程不可用，APPS 是合理非-agent fallback，但只能声称“不局限于
tool calling”；不能用它回答 interactive-agent 部分。AppWorld 是最有价值的后续
跨域确认，但 rebuttal 主证据优先采用成本更可控、可执行评分明确的 InterCode-Bash。

## 18. 建议论文表述

只有 primary 和 construct checks 都支持时，建议写：

> We additionally evaluate TapMem on InterCode-Bash, where an agent freely writes and
> executes shell programs with environment feedback and no procedure labels or segment
> boundaries are provided. We induce a vocabulary of variable-length Bash motifs solely
> from decontaminated NL2Bash training programs using a syntax-constrained unigram model.
> To isolate boundary placement ambiguity from the amount of transition supervision, we
> sample alternative segmentations conditioned on the MAP number of procedures, while
> sharing the exact sampled schedule across TokMem, EOC-only, and TapMem. At test time,
> no parser, segmentation, memory ID, or reference command is supplied; success is measured
> by the benchmark's output- and file-state-based reward.

随后只填入真实结果和 CI。禁止写“发现真实边界”“TapMem 精确边缘化 boundary”或
“已经泛化到任意开放世界 agent”。

## 19. 主要资料

- Taku Kudo. [Subword Regularization: Improving Neural Network Translation Models with
  Multiple Subword Candidates](https://aclanthology.org/P18-1007/), ACL 2018。
  提供 unigram segmentation、EM、likelihood-based pruning 和 posterior sampling 的
  直接方法基础；本文将基本单位从字符片段换成 syntax-safe Bash atoms，并额外显式
  参数化 stop probability。
- [SentencePiece: A simple and language independent subword tokenizer and
  detokenizer](https://aclanthology.org/D18-2012/)，用于核对 unigram vocabulary
  induction 的工程路线。
- [bashlex 官方仓库](https://github.com/idank/bashlex)。它是 GNU Bash parser 的
  Python port，不执行命令并输出完整 AST；官方也明确列出 arithmetic/复杂 parameter
  expansion 等限制。
- [NL2Bash 官方仓库](https://github.com/TellinaTool/nl2bash)，提供 raw parallel
  corpus、filter/split scripts 和 utility grammar。
- [InterCode 论文](https://papers.neurips.cc/paper_files/paper/2023/file/4b175d846fb008d540d233c188379ff9-Paper-Datasets_and_Benchmarks.pdf)
  与 [官方仓库](https://github.com/princeton-nlp/intercode)，提供 200 个 Bash
  interactive tasks、Docker environment 和 output/file-system reward。
