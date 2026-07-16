# Rebuttal 大纲

## 实验

方法本身:

1. [ ] 对比实验: 不用额外的 adapter, memory token 解码时直接约束在 memory bank 里, 对比有 adapter 的效果
2. [ ] 对比实验: adapter logits 和 lm head logits 直接相加的实验, 不对 adapter logits 做额外操作
3. [ ] 对比实验: AR loss 不影响 adapter, adapter 只受到 routing loss 更新

实验设置

1. [x] 使用合成数据集: 尝试在 TaskBench, APIGen, $\tau$-bench, StableToolBench, Berkeley Function Calling Leaderboard 上进行实验
2. [x] 泛化性实验: 在训练上只跑过 4call 的模型在 10call 上测试
3. [x] EOC 边界预测准确率实验: 补一个 \<EOC\> token 的 F1 分数, 证明边界预测准确
4. [ ] 效果提升来源实验: 无 adaptation 效果提升主要原因是 transition aware, 而非参数量增加
    1. [x] 汇报下 procedure 格式错误的比例, 证明去掉格式错误的 procedure, 我们的方法依然减少了 transition error;
    2. [ ] 考虑到现有统计方法在格式错误时会判定参数生成错误, 实际存在格式错误但参数生成正确的情况, 如果上面的结果无法说明, 拿大模型逐个 procedure 分析再汇报一个结果
5. [ ] TokMem 的实验里, procedure 顺序错了但 set 一样仍会算是对的, 统计下这类错误的比例
6. [ ] 消融实验: adapter 使用 linear 和 mlp 两种结构的效果对比
7. [ ] 修正效果对比: adapter 修正 memory token logits 前后词表分布的熵对比和修正准确率对比
8. [ ] memory 更新: 在已经训练好的 memory bank 基础上, 加入新的 memory token, 看原有 memory token 路由准确率的变化

额外分析:

1. [x] 错误类型分析: 对比 TapMem 和 TokMem 在不同错误类型上的降低比例, 证明 TapMem 主要减少的是 transition-related errors, 比如 wrong eoc, extra eoc, missing procedure, extra procedure, correct tool but wrong arguments,
2. [ ]


无法回应的质疑:

1. [ ] adapter 输出空间是 memory token + eoc token 的词表空间, 效果会不会更好

## Rebuttal 补充实验的最终口径

本节记录 rebuttal 补充实验已经确认的对照组、checkpoint 选择与指标定义。总体原则是：如果项目中存在 `paper_tapmem.pdf` 表格对应数据的 checkpoint，就优先直接使用该 checkpoint 做补充评测；只有对应 checkpoint 缺失时才考虑重新训练，或者不汇报该模型规模。

### 1. 4-call checkpoint 在 10-call test 上的长度泛化

当前 `compositional/rebuttal/results/train4_checkpoint_eval_10calls/` 中的结果不能作为最终结果，因为它实际比较的是无 adaptation 的 TapMem 和有 adaptation 的 TapMem，而不是论文表中的 TokMem 和 TapMem。

最终对照组固定为：

- TokMem：无 adaptation 的 `tokmem` checkpoint。
- TapMem：无 adaptation 的 `tokmem_eoc_logit_bias` checkpoint。

Llama-1B、Llama-3B 和 Llama-8B 的论文表 checkpoint 都已经存在，因此本实验只需重新评测，不需重新训练。TokMem 的 1B/8B checkpoint 来自 `results/compositional/all_methods/`；TokMem 的 3B checkpoint 来自论文最终使用的 `compositional/runs/tokmem_llama_3b_4calls_seed42_3x_20260425_214658_*`；TapMem 的 1B/3B/8B checkpoint 来自 `results/compositional/paper_compositional_head_8gpu/`。

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

### 2. EOC 边界预测

EOC 实验使用 4-call APIGen test 上的自由生成结果，而不是 teacher forcing。对于 TapMem，1B 和 8B 已经使用论文表对应的 `tokmem_eoc_logit_bias` checkpoint，不需重跑；3B 可以用已存在的论文表 checkpoint 补评测，也可以不汇报 3B。原始 TokMem 没有 `<EOC>` token，因此不将 TokMem 作为 EOC F1 的必要对照。

主表只保留两个指标：

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

### 3. 错误类型与 transition 分析

错误分析分成两组独立结果。第一组是样本级互斥分类，用来判断 TapMem 主要抑制了哪些错误类型；第二组是 later-step mismatch，专门判断 TapMem 是否改善了后续 procedure transition。两组指标分开汇报，不把 later-step mismatch 当作第七个互斥类别。

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

由于六类互斥且覆盖全部样本，所以 $\sum_c p_c=1$。同时汇报 TapMem 相对 TokMem 的绝对变化

$$
\Delta_c=p_c^{\mathrm{TapMem}}-p_c^{\mathrm{TokMem}}.
$$

对 `Correct`，$\Delta_c>0$ 表示改善；对五个错误类，$\Delta_c<0$ 表示 TapMem 抑制了该类错误。

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

在 later-step mismatch 表中汇报 TokMem、TapMem 的两个 rate，以及 $\Delta=\mathrm{TapMem}-\mathrm{TokMem}$。两个 rate 越低越好，$\Delta<0$ 表示 TapMem 改善了后续 procedure transition。

### 4. TaskBench DailyLife

TaskBench 已经满足 rebuttal 需求，不再扩展 batch size、logit-bias scale 或 routing-loss weight 搜索。每个模型规模内 TokMem 和 TapMem 使用相同学习率和相同 seed，seed 42 重复运行至少 3 次。最终汇报：

- Llama-1B：学习率 `1e-2`。
- Llama-8B：学习率 `5e-3`。

两个模型规模上，TapMem 的 Tool F1、Argument F1 和 Routing Accuracy 都高于 TokMem，Transition Error 都低于 TokMem。主表指标定义如下：

- **Tool F1**：预测 tools 与 gold tools 的 F1；对每个样本计算后取平均。
- **Argument F1**：通过 `compare_function_calls_advanced(..., ignore_order=True)` 计算 function-call/argument F1；对样本取平均。
- **Routing Accuracy**：预测 tool 序列与 gold tool 序列完全相同的样本比例，即 $N^{-1}\sum_i\mathbf{1}[\hat T_i=T_i]$。
- **Rouge-L**：预测 function-call 序列与 gold function-call 序列的 Rouge-L；对样本取平均。
- **Transition Error**：与 TapMem 论文的 token-level transition error 口径一致。对 gold 序列从第二个 procedure 开始逐位比较，错工具或缺失记为错误，额外预测不进入分母，最后计算错误位置数除以 gold 后续 procedure 位置总数。

### 5. TRAJECT-Bench Sequential

TRAJECT-Bench 也已经满足 rebuttal 需求。最终只使用学习率搜索结果，每个模型规模内 TokMem 和 TapMem 使用相同学习率和 seed 42，重复运行 3 次。最终汇报：

- Llama-1B：学习率 `8e-3`。
- Llama-8B：学习率 `8e-3`。

两个模型规模上，TapMem 的 Tool F1、Argument F1、Routing Accuracy 和 Rouge-L 都高于 TokMem，Transition Error 都低于 TokMem。Tool F1、Argument F1、Routing Accuracy、Rouge-L 和 Transition Error 的含义与上述 TaskBench 定义相同，不再使用早期单次运行的归档结果作为主表数据。

## 尚未完成的 Rebuttal 实验

### 已有数据或 checkpoint，但还需按最终口径落实

1. [ ] 使用无 adaptation 的 TokMem 和 TapMem 论文 checkpoint，重跑 4-call checkpoint 在 10-call test 上的评测。
2. [ ] 将 EOC 主表精简为 `EOC F1` 和 `Exact EOC Count`；3B 选择使用论文 checkpoint 补评测或不汇报。
3. [ ] 从现有 prediction 重新计算六类样本级互斥错误指标。
4. [ ] 从现有 prediction 计算 `Later-step mismatch rate (all)` 和 `Later-step mismatch rate (first-correct)`。
5. [ ] 按已选学习率汇总 TaskBench 1B/8B 最终表。
6. [ ] 按已选学习率汇总 TRAJECT-Bench 1B/8B 最终表。

### 尚未开展的实验

1. [ ] 去掉额外 routing adapter，memory-token 解码时直接约束在 memory bank 中，与 TapMem 对比。
2. [ ] adapter logits 与 LM-head logits 直接相加、不对 adapter logits 做额外处理的对比。
3. [ ] 让 AR loss 不更新 adapter，adapter 只接收 routing loss 的对比。
4. [ ] 参数量匹配的控制实验，验证无 adaptation 时的改善主要来自 transition-aware 机制，而不是参数量增加。
5. [ ] Linear adapter 与 MLP adapter 的结构消融。
6. [ ] adapter 修正前后 memory-token logit 分布的熵和路由准确率对比。
7. [ ] 在已训练 memory bank 中加入新 memory token，评估原有 memory token 的路由准确率变化。
8. [ ] 将 adapter 输出空间限制为 memory tokens + EOC token 的对比。
9. [ ] 如果仍保留原始外部 benchmark 计划，补充 $\tau$-bench、StableToolBench 和 BFCL 实验。
