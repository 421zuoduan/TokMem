# Rebuttal 大纲

## 当前最高优先级

**当前最重要的是以下两项，优先级高于其他未完成实验：**

1. **与 tool-calling 方法对比**：在统一的数据和评测口径下，将 TokMem/TapMem 与现有 tool-calling 方法进行对比。
2. **Memory token 增量更新**：在已训练的 memory bank 中新增 memory token，实现增量更新，并评估新旧 memory token 的路由效果及对原有能力的影响。

## 实验

方法本身:

1. [x] 对比实验: 不用额外的 adapter, memory token 解码时直接约束在 memory bank 里, 对比有 adapter 的效果
2. **[不做]** 对比实验: adapter logits 和 lm head logits 直接相加的实验, 不对 adapter logits 做额外操作
3. [x] 对比实验: AR loss 不影响 adapter, adapter 只受到 routing loss 更新

实验设置

1. [x] 使用合成数据集: 已完成 TaskBench 和 TrajectBench 实验；**[不做]** 后续不再扩展 $\tau$-bench、StableToolBench 和 Berkeley Function Calling Leaderboard
2. [x] 泛化性实验: 在训练上只跑过 4call 的模型在 10call 上测试
3. [x] EOC 边界与作用验证
    1. [x] 在自由生成结果上计算 EOC-only 的 \<EOC\> F1, 证明显式边界可以被准确预测
    2. [x] EOC 主表只保留 `EOC F1` 和 `Exact EOC Count`，复用已有 EOC-only 自由生成结果；主表汇报 Llama-1B/8B，Llama-3B 不进入主表
    3. [x] 将 EOC-only 加入错误类型与 later-step mismatch 分析, 证明 EOC 不仅可预测, 而且能改善后续 procedure transition
4. [ ] 效果提升来源实验: 无 adaptation 效果提升主要原因是 transition aware, 而非参数量增加
    1. [x] 汇报下 procedure 格式错误的比例, 证明去掉格式错误的 procedure, 我们的方法依然减少了 transition error;
    2. [ ] 考虑到现有统计方法在格式错误时会判定参数生成错误, 实际存在格式错误但参数生成正确的情况, 如果上面的结果无法说明, 拿大模型逐个 procedure 分析再汇报一个结果
5. [x] TokMem 的实验里, procedure 顺序错了但 set 一样仍会算是对的, 统计下这类错误的比例
6. [x] 消融实验: adapter 使用 linear 和 mlp 两种结构的效果对比
7. [ ] memory 更新: 在已经训练好的 memory bank 基础上, 加入新的 memory token, 看原有 memory token 路由准确率的变化, 看 TCRA 更新是否只需要新加列
8. [ ] 混合回答设置实验: 将数据集改为 procedure 与普通文本混合的设置，使样本回答可包含 procedure 和普通文本，而非所有回答都只由 procedure 组成。
9.  [ ] 对比方法

额外分析:

1. [x] 错误类型与 transition 分析：按下文最终口径对 TokMem 和 TapMem 重新计算六类样本级互斥错误，以及两个 later-step mismatch 指标；EOC 边界实验额外复用相同口径评测 EOC-only。
2. [ ] 修正效果对比: adapter 修正 memory token logits 前后词表分布的熵对比和修正准确率对比


无法回应的质疑:

1. [ ] adapter 输出空间是 memory token + eoc token 的词表空间, 效果会不会更好
2. [ ] 参数量不匹配问题

## 论文写作与表述问题

本节集中记录论文写作中需要澄清或修正的表述问题，并与补充实验任务分开管理。修订时需要核对正文、公式、伪代码和实际实现是否一致，同时确认问题是否影响方法理解、实验设置或结果解释。

1. [ ] 修正论文伪代码的表述错误。
   - token 表示错位, 现在伪代码的表示是当前 token 预测生成当前 token 的 embedding
2. [ ] 摘要里 "an LLM" 表述错误
3. [ ] Explicit Termination Signals 出现的太少了, 摘要里说 ETS 和 TCRA, 但是正文里总是说 EOC 和 TCRA
4. [ ] 把 y 定义成多个 procedure 相邻的形式有些狭隘了, procedure 不一定是一种互斥或原子化的行为, 所以 y 和 boundary 的表示都有可能被质疑

## Rebuttal 补充实验的最终口径

本节记录 rebuttal 补充实验已经确认的对照组、checkpoint 选择与指标定义。总体原则是：如果项目中存在 `paper_tapmem.pdf` 表格对应数据的 checkpoint，就优先直接使用该 checkpoint 做补充评测；只有对应 checkpoint 缺失时才考虑重新训练，或者不汇报该模型规模。

### 1. 4-call checkpoint 在 10-call test 上的长度泛化

早期 `compositional/rebuttal/results/train4_checkpoint_eval_10calls/` 中使用错误对照组的结果已经删除。最终结果保存在 `compositional/rebuttal/results/train4_checkpoint_eval_10calls_final/`，比较的是论文表中的 TokMem 和 TapMem。

最终对照组固定为：

- TokMem：无 adaptation 的 `tokmem` checkpoint。
- TapMem：无 adaptation 的 `tokmem_eoc_logit_bias` checkpoint。

Llama-1B、Llama-3B 和 Llama-8B 均已使用论文表 checkpoint 完成重新评测，不需重新训练。TokMem 的 1B/8B checkpoint 来自 `results/compositional/all_methods/`；TokMem 的 3B checkpoint 来自论文最终使用的 `compositional/runs/tokmem_llama_3b_4calls_seed42_3x_20260425_214658_*`；TapMem 的 1B/3B/8B checkpoint 来自 `results/compositional/paper_compositional_head_8gpu/`。

该实验使用只在 4-call 数据上训练的上述 checkpoint，直接在 tools 51-100 的 10-call test split 上生成。主表汇报以下指标：

- **Tool F1**
  - 含义：预测的 procedure tools 与 gold tools 的集合式重合程度，与论文的 Tool Selection F1 口径保持一致。
  - 计算：先根据预测和 gold tool 的重合计算 precision 和 recall，再计算
    $F1_{tool}=2P_{tool}R_{tool}/(P_{tool}+R_{tool})$，最后对所有样本取平均。
- **Argument F1**
  - 含义：预测 function calls 与 gold function calls 在参数生成上的匹配程度，与论文的 Arguments Generation F1 口径保持一致。
  - 计算：复用项目现有的 `compare_function_calls_advanced(..., ignore_order=True)`，对每个样本计算 function-call/argument F1，再对样本取平均。
- **Tool Sequence Exact**
  - 含义：完整 procedure tool 序列是否连同顺序完全正确。
  - 计算：对样本 $i$ 计算 $\mathbf{1}[\hat T_i=T_i]$，要求长度相同且每个位置的 tool 相同，再对全部样本取平均。
- **Call Exact**
  - 含义：整个 function-call 预测是否完全正确。
  - 计算：复用现有 function-call evaluator 的 exact match 结果，对每个样本记为 0/1，再对全部样本取平均。

`parse error rate` 保持当前口径，不在这轮 rebuttal 整理中额外修改或扩展。

### 2. EOC 边界预测与作用验证

EOC 的作用是显式标记一个 tool-controlled span 的结束，并把该位置作为下一次 procedure routing 的统一决策边界。对 EOC 的验证分成两个相互补充的部分：EOC boundary accuracy 用来证明显式边界可以被可靠预测；TokMem vs EOC-only 的错误类型与 later-step mismatch 用来证明这个边界确实改善了后续 procedure transition。EOC F1 较高本身只表示模型学会了生成 `<EOC>`，不能单独证明 EOC 对任务有效。本实验不涉及 TapMem 或 logit bias/TCRA。

#### 2.1 EOC boundary accuracy

边界预测实验使用 4-call APIGen test 上的自由生成结果，而不是 teacher forcing。EOC F1 只对 EOC-only 统计；原始 TokMem 没有 `<EOC>` token，对应指标记为不适用，不将缺少 EOC 计为预测错误。最终主表直接复用 `compositional/rebuttal/results/eoc_boundary_accuracy/` 中已有的 EOC-only 自由生成结果：Llama-1B 和 Llama-8B 各汇总 5 个 trial，每个 trial 500 个样本；Llama-3B 不进入主表。

EOC boundary accuracy 表只保留两个指标：

- **EOC F1**
  - 含义：模型在自由生成中对 procedure 结束边界的综合预测质量，同时惩罚缺失、多余和不合法的 EOC。
  - 计算：一个 `<EOC>` 在已打开的 tool span 之后出现，且是该 span 的第一个 `<EOC>` 时，记为一次 valid closure。对样本 $i$，令 $J_i$ 为 gold procedure 数，$V_i$ 为 valid closure 数，$E_i$ 为实际生成的 EOC 总数，则
    $TP_i=\min(V_i,J_i)$，$FP_i=\max(E_i-TP_i,0)$，$FN_i=\max(J_i-TP_i,0)$。先在全部样本上累加 TP/FP/FN，再计算 precision、recall 和
    $F1_{EOC}=2P_{EOC}R_{EOC}/(P_{EOC}+R_{EOC})$。
  - 说明：这是 generated EOC boundary F1，衡量自由生成序列中有效边界的数量质量，不是 teacher-forced 的精确 token-position F1。
- **Exact EOC Count**
  - 含义：生成的 EOC 数量是否与 gold procedure 数完全相同。
  - 计算：对样本 $i$ 计算 $\mathbf{1}[E_i=J_i]$，再对全部样本取平均。

EOC precision、EOC recall、missing/extra EOC 和 `Malformed Boundary Rate` 不放入 rebuttal 主表。`Malformed Boundary Rate` 原本表示至少出现一次结构异常的样本比例，异常包括前一个 tool span 还没有 EOC 就生成下一个 tool token、没有打开 tool span 时生成 EOC，以及同一 span 内重复生成 EOC。该指标只保留为内部调试信息。

#### 2.2 EOC usefulness

EOC 是否有用不再另外定义一个让 TokMem 模拟 EOC token 的指标，而是复用第 3 节的错误类型与 later-step mismatch 口径。核心对比是 TokMem vs EOC-only，主指标是 `Later-step mismatch rate (first-correct)`：第一个 procedure 之前没有 EOC，只在第一个 procedure 已经预测正确的样本中检查后续位置，可以更直接地隔离 EOC 对跨 procedure transition 的作用。

辅助证据包括 `Later-step mismatch rate (all)`、`Later-only routing error`、`Length error` 和 `Tool Sequence Exact`。如果 EOC-only 具有较高 EOC F1，且相对 TokMem 降低 later-step mismatch、later-only routing error 和 length error，同时提高 tool-sequence exact，才构成“EOC 可被准确预测且确实有用”的完整证据链。这组 TokMem vs EOC-only 比较属于 EOC 边界实验的结果，正文应放在本节；它只复用第 3 节的错误类型与 later-step mismatch 定义。

### 3. 错误类型与 transition 分析

错误分析主体比较 TokMem 和 TapMem。第一组是样本级互斥分类，用来判断 TapMem 主要抑制了哪些错误类型；第二组是 later-step mismatch，专门判断 TapMem 是否改善了后续 procedure transition。两组指标分开汇报，不把 later-step mismatch 当作第七个互斥类别。第 2 节的 EOC 边界实验复用这里的定义比较 TokMem 和 EOC-only，但 EOC-specific 的结果与归因放在第 2 节汇报。

#### 3.1 样本级六类互斥结果

对样本 $i$，记 gold procedure tool 序列为 $T_i$，预测序列为 $\hat T_i$。按下列顺序判定，每个样本只进入一类：

1. **Correct**
   - 含义：procedure tool 序列和所有对应参数都完全正确。
   - 计算：$\hat T_i=T_i$，且现有 function-call evaluator 判定整个调用结果 exact match。
2. **Argument-only error**
   - 含义：procedure tool 序列完全正确，但至少一个 function call 的参数不正确。
   - 计算：$\hat T_i=T_i$，但 function-call exact match 为 false。
3. **Length error**
   - 含义：预测 procedure 数与 gold 不同，合并 missing/early stop 和 extra/over-generation。
   - 计算：$|\hat T_i|\ne|T_i|$。子类可在附录中按 $|\hat T_i|<|T_i|$ 和 $|\hat T_i|>|T_i|$ 分开统计。
4. **Order-only error**
   - 含义：预测包含了正确的 procedure tools 及正确的出现次数，但执行顺序错误。这就是 rebuttal 清单中“procedure 顺序错了但 tool set 一样”的严格实现。
   - 计算：$|\hat T_i|=|T_i|$，$\operatorname{Multiset}(\hat T_i)=\operatorname{Multiset}(T_i)$，但 $\hat T_i\ne T_i$。使用 multiset 而不是普通 set，以保留同一 tool 重复出现的次数。
5. **Initial-involved routing error**
   - 含义：procedure 数相同，错误不是单纯的顺序交换，并且错误涉及第一个 procedure。该类中允许后续位置也同时错误，因此不命名为 `Wrong first tool`。
   - 计算：$|\hat T_i|=|T_i|$，$\operatorname{Multiset}(\hat T_i)\ne\operatorname{Multiset}(T_i)$，且 $\hat T_{i,1}\ne T_{i,1}$。
6. **Later-only routing error**
   - 含义：procedure 数相同且第一个 procedure 正确，但后续至少一个 procedure 路由错误，且不属于 order-only error。
   - 计算：$|\hat T_i|=|T_i|$，$\operatorname{Multiset}(\hat T_i)\ne\operatorname{Multiset}(T_i)$，$\hat T_{i,1}=T_{i,1}$，且 $\hat T_i\ne T_i$。

对完全无法恢复 procedure tool 的输出，将预测 tool 序列视为空序列，因而归入 `Length error`；如果 tool 序列可恢复但参数不可解析，当 tool 序列完全正确时归入 `Argument-only error`。不再单独汇报 `Format error`。

对于类别 $c$，汇报

$$
p_c=\frac{N_c}{N}.
$$

由于六类互斥且覆盖全部样本，所以 $\sum_c p_c=1$。同时汇报 TapMem 相对 TokMem 的绝对变化：

$$
\Delta_c=p_c^{\mathrm{TapMem}}-p_c^{\mathrm{TokMem}}.
$$

对 `Correct`，$\Delta_c>0$ 表示改善；对五个错误类，$\Delta_c<0$ 表示 TapMem 抑制了该类错误。EOC 边界实验中的 TokMem vs EOC-only 使用相同的差值方向解释，但具体结果在第 2 节汇报。

#### 3.2 Later-step mismatch

Later-step mismatch 不与上述六类概率混合，它独立用来验证 TapMem 是否改善了从当前 procedure 切换到后续 procedure 的能力。它以 gold 序列中第二个及之后的 procedure 位置为统计单位，不使用互斥样本类别的优先级。

- **Later-step mismatch rate (all)**
  - 含义：在所有多 procedure 样本中，忽略第一个 initial decision，后续 gold procedure 位置上的平均失配率。即使第一个 tool 错误，也继续按位置检查后续 procedure，避免后续错误被 `Initial-involved routing error` 吸收。
  - 计算：对样本 $i$ 的 gold 序列长度 $J_i\ge2$，从 $j=2$ 到 $J_i$ 逐位比较 $\hat T_{i,j}$ 与 $T_{i,j}$。错工具和由于提前停止导致的缺失都记为 mismatch；gold 长度之外的额外预测不进入该指标的分母：

    $$
    \operatorname{LMR}_{all}=
    \frac{\sum_i\sum_{j=2}^{J_i}\mathbf{1}[\hat T_{i,j}\ne T_{i,j}]}
    {\sum_i(J_i-1)}.
    $$

- **Later-step mismatch rate (first-correct)**
  - 含义：在第一个 procedure 已经预测正确的样本中，后续 gold procedure 位置上的平均失配率。该指标隔离 initial routing 失败，直接检查模型在正确进入第一个 procedure 之后能否继续正确切换。
  - 计算：只保留满足 $\hat T_{i,1}=T_{i,1}$ 的样本，然后按与 `all` 相同的规则计算：

    $$
    \operatorname{LMR}_{first}=
    \frac{\sum_{i:\hat T_{i,1}=T_{i,1}}\sum_{j=2}^{J_i}\mathbf{1}[\hat T_{i,j}\ne T_{i,j}]}
    {\sum_{i:\hat T_{i,1}=T_{i,1}}(J_i-1)}.
    $$

在错误分析主表中汇报 TokMem、TapMem 的两个 rate，以及 $\Delta=\mathrm{TapMem}-\mathrm{TokMem}$。两个 rate 越低越好，$\Delta<0$ 表示 TapMem 改善了后续 procedure transition。EOC 边界实验额外按相同定义比较 TokMem 和 EOC-only，其中 `Later-step mismatch rate (first-correct)` 是验证 EOC usefulness 的主指标；该结果放在第 2 节。

### 4. TaskBench DailyLife

TaskBench 已经满足 rebuttal 需求，不再扩展 batch size、logit-bias scale 或 routing-loss weight 搜索。每个模型规模内 TokMem 和 TapMem 使用相同学习率和相同 seed，seed 42 重复运行至少 3 次。最终汇报：

- Llama-1B：学习率 `1e-2`。
- Llama-8B：学习率 `5e-3`。

两个模型规模上，TapMem 的 Tool F1 和 Argument F1 都高于 TokMem。主表只汇报以下 F1 指标：

- **Tool F1**：预测 tools 与 gold tools 的 F1；对每个样本计算后取平均。
- **Argument F1**：通过 `compare_function_calls_advanced(..., ignore_order=True)` 计算 function-call/argument F1；对样本取平均。

### 5. TRAJECT-Bench Sequential

TRAJECT-Bench 也已经满足 rebuttal 需求。TokMem 和 TapMem 最终只使用学习率搜索结果，每个模型规模内使用相同学习率和 seed 42，重复运行 3 次。ICL 严格对齐 TokMem upstream 的无训练推理与评测：prompt 提供候选工具文档和静态 arguments JSON 示例，模型只生成逐行 arguments JSON；Tool F1 沿用 upstream 的 argument-key proxy 实现。最终汇报：

- Llama-1B：学习率 `8e-3`。
- Llama-8B：学习率 `7e-3`。

两个模型规模上，TapMem 的 Tool F1 和 Argument F1 都高于 TokMem 和 ICL。主表只汇报 Tool F1 和 Argument F1；TokMem/TapMem 的指标含义与上述 TaskBench 定义相同，ICL 的 Tool F1 则沿用 TokMem upstream 的公开实现。不再使用早期单次运行、schema-inferred compositional ICL 或显式工具名 Direct Prompting 的结果作为最终 ICL 数据。

| Model | Method | LR | Tool F1 | Argument F1 |
| --- | --- | ---: | ---: | ---: |
| Llama-1B | ICL | — | 0.0556* | 0.0000 |
| Llama-1B | TokMem | `8e-3` | 0.5864 | 0.2085 |
| Llama-1B | TapMem | `8e-3` | **0.7863** | **0.4210** |
| Llama-8B | ICL | — | 0.5785* | 0.2227 |
| Llama-8B | TokMem | `7e-3` | 0.6847 | 0.3077 |
| Llama-8B | TapMem | `7e-3` | **0.8534** | **0.5565** |

Llama-8B 的 `7e-3` 结果来自 2026-07-24 的补充学习率搜索。相对旧的共同学习率 `8e-3`，TapMem 的 Tool F1 / Argument F1 分别提高 `0.0322 / 0.0523`，TapMem 相对 TokMem 的差值扩大到 `0.1687 / 0.2488`。TokMem 的 Argument F1 提高 `0.0231`，Tool F1 小幅变化 `-0.0044`。

\* ICL 的 Tool F1 严格复现 TokMem upstream evaluator：它把生成 arguments 的键名作为工具名计算 F1，属于 argument-key proxy，不是真实工具选择 F1。这里保留该实现是为了与 TokMem 原仓库的 ICL baseline 对齐。

## 已完成的 Rebuttal 实验

### 已有数据或 checkpoint，已按最终口径落实

1. [x] 使用无 adaptation 的 TokMem 和 TapMem 论文 checkpoint，重跑 4-call checkpoint 在 10-call test 上的评测。
2. [x] 完成 EOC 边界实验：复用已有 EOC-only 自由生成结果，boundary accuracy 主表只汇报 Llama-1B/8B 的 `EOC F1` 和 `Exact EOC Count`；TokMem 和 EOC-only 的对齐 prediction、六类错误与 later-step mismatch 已完成；3B 不进入主表。
3. [x] 从 TokMem 和 TapMem prediction 重新计算错误分析主表的六类样本级互斥指标。
4. [x] 从 TokMem 和 TapMem prediction 计算错误分析主表的 `Later-step mismatch rate (all)` 和 `Later-step mismatch rate (first-correct)`。
5. [x] 按已选学习率汇总 TaskBench 1B/8B 最终表。
6. [x] 按已选学习率汇总 TRAJECT-Bench 1B/8B 最终表。
7. [x] 完成 Llama-1B/8B APIGen 4-call 的 routing-only adapter 对照：训练和推理时保留 logit train-add，但阻断 AR loss 对 adapter 的梯度。

### Llama-1B APIGen 4-call：routing-only adapter

三种方法使用相同的 tools 51–100 数据集。下表均为 seed 42 的 3 次独立运行均值，括号内为样本标准差；只保留 rebuttal 使用的两个 F1 指标。

| Method | Tool F1 | Arguments F1 |
| --- | ---: | ---: |
| TokMem | 0.8087 (0.0023) | 0.6594 (0.0101) |
| TapMem | **0.8969 (0.0084)** | **0.7426 (0.0188)** |
| TapMem，adapter 仅由 routing loss 更新 | 0.8805 (0.0109) | 0.7169 (0.0067) |

### Llama-8B APIGen 4-call：routing-only adapter

三种方法使用相同的 tools 51–100 数据集。下表均为 seed 42 的 3 次运行均值，括号内为样本标准差；只保留 rebuttal 使用的两个 F1 指标。

| Method | Tool F1 | Arguments F1 |
| --- | ---: | ---: |
| TokMem | 0.8413 (0.0000) | 0.7121 (0.0000) |
| TapMem | **0.8666 (0.0000)** | 0.7473 (0.0000) |
| TapMem，adapter 仅由 routing loss 更新 | 0.8621 (0.0058) | **0.7616 (0.0015)** |

结果解释：routing loss 只优化分类器自身的 $-\log p(y\mid h)$，但实际解码依赖 adapter prior 与 LM logits 的融合结果 $z_{\mathrm{final}}$。纯分类损失会推动 adapter 给出更高置信度；当测试上下文中的分类判断错误时，较大的 bias 可能覆盖 LM 原本正确的工具排序。相比之下，TapMem 中来自 AR loss 的梯度会使 adapter 学习针对当前 LM logits 的残差修正，并同时校准其置信度与影响强度。因此，routing-only adapter 即使更接近纯分类器，其融合后的 Tool F1 仍可能略低于 TapMem。当前差值仅为 $-0.0045$，小于 routing-only 结果的运行标准差 $0.0058$，该机制解释仍需结合 adapter-only routing accuracy、融合前后 correct-to-wrong / wrong-to-correct 翻转和 bias scale/temperature 校准实验进一步验证。

## Rebuttal 实验进度

1. [x] 去掉额外 routing adapter，memory-token 解码时直接约束在 memory bank 中，与 TapMem 对比。
   - Llama-1B 与 Llama-8B 均已完成三组评测：`TokMem + bank constraint` 使用完整词表归一化后的 memory-bank 总概率 `>= 0.5` 触发；`EOC-only + bank constraint` 在 assistant-start 和实际生成的 EOC 后触发；`TapMem + bank constraint` 在相同显式边界先融合 TCRA bias，再约束候选。三组候选均为 memory tokens + `tokenizer.eos_token_id`；EOC-only/TapMem 的 `0.5` 阈值仅记录诊断，不控制触发。

### Llama-1B APIGen 4-call：memory-bank constraint

tools 51–100、500 个 4-call 测试样本，greedy decoding，测试 batch size 8。下表均为 seed 42 的 3 个论文 checkpoint 的均值，括号内为样本标准差。

| Method | Tool F1 | Arguments F1 |
| --- | ---: | ---: |
| TokMem | 0.8087 (0.0023) | 0.6594 (0.0101) |
| TokMem + bank constraint | 0.8112 (0.0034) | 0.6624 (0.0070) |
| EOC-only | 0.8509 (0.0066) | 0.7126 (0.0113) |
| EOC-only + bank constraint | 0.8522 (0.0052) | 0.7157 (0.0138) |
| TapMem | **0.8969 (0.0084)** | **0.7426 (0.0188)** |
| TapMem + bank constraint | 0.8950 (0.0075) | 0.7392 (0.0167) |

TokMem + constraint 平均每个样本触发 3.0487 次，changed-trigger rate 为 0.0007；EOC-only + constraint 为 3.4893/0.0010；TapMem + constraint 为 3.5413/0.0004。TapMem 加约束后的 Tool/Arguments F1 相对原方法变化 `-0.0019/-0.0034`，小于运行间标准差。说明 greedy 解码在这些触发位置原本几乎总会把 memory token 或 EOS 作为融合分布的 top-1，hard constraint 对 TapMem 也没有额外增益。

### Llama-8B APIGen 4-call：memory-bank constraint

tools 51–100、500 个 4-call 测试样本，greedy decoding，测试 batch size 1。下表按现有 3 个 trial 条目取均值，括号内为样本标准差。

| Method | Tool F1 | Arguments F1 |
| --- | ---: | ---: |
| TokMem | 0.8413 (0.0000) | 0.7121 (0.0000) |
| TokMem + bank constraint | 0.8436 (0.0000) | 0.7109 (0.0000) |
| EOC-only | 0.8578 (0.0000) | 0.7375 (0.0000) |
| EOC-only + bank constraint | 0.8614 (0.0000) | 0.7428 (0.0000) |
| TapMem | **0.8666 (0.0000)** | 0.7473 (0.0000) |
| TapMem + bank constraint | 0.8652 (0.0000) | **0.7515 (0.0000)** |

TokMem + constraint 平均每个样本触发 2.7620 次，changed-trigger rate 为 0.0007；EOC-only + constraint 为 3.5540/0.0011；TapMem + constraint 为 3.5900/0.0022。相对无约束基线，TokMem 的 Tool/Arguments F1 变化为 `+0.0023/-0.0012`，EOC-only 为 `+0.0037/+0.0053`，TapMem 为 `-0.0014/+0.0042`。所有变化都很小且没有一致方向，说明 hard constraint 不是主要性能来源。

注意：Llama-8B 三种 constrained 方法各自的 3 个 trial 条目均具有完全相同的 checkpoint fingerprint，所以相应结果相同且标准差为 0；这里按用户要求计算了均值，但不能将其视为 3 个独立训练重复。

2. **[不做]** adapter logits 与 LM-head logits 直接相加、不对 adapter logits 做额外处理的对比。
3. [ ] 参数量匹配的控制实验，验证无 adaptation 时的改善主要来自 transition-aware 机制，而不是参数量增加。
   - 使用 AR-only head 作为参数量匹配对照：保留与 TapMem 完全相同的 linear adapter、logit train-add 和推理时 logit fusion，将 routing-loss weight 设为 0，使 adapter 仅由正常 AR loss 更新。AR-only head 与 TapMem 参数量一致，两者的性能差异用于隔离显式 routing supervision 的贡献。
4. [x] Linear adapter 与 MLP adapter 的结构消融。
5. [ ] adapter 修正前后 memory-token logit 分布的熵和路由准确率对比。
6. [ ] 在已训练 memory bank 中加入新 memory token，评估原有 memory token 的路由准确率变化。
7. [ ] 将 adapter 输出空间限制为 memory tokens + EOS token 的对比。
8. **[不做]** 原始外部 benchmark 计划：$\tau$-bench、StableToolBench 和 BFCL 实验后续不再考虑。
9. [ ] 混合回答设置实验：将数据集改为 procedure 与普通文本混合的设置，使样本回答可包含 procedure 和普通文本，而非所有回答都只由 procedure 组成。
