# TapMem 新工具冷启动方案：不再训练，直接生成新参数

> 本文是一份方法设计稿。目前还没有实现，也没有实验结果。

## 1. 我们到底要解决什么问题

TapMem 已经在 $K$ 个旧工具上训练完成。现在给它一个训练时没见过的新工具，我们只知道这个工具的：

- 名称；
- 功能说明；
- 参数名、参数类型和必填项。

我们不能拿新工具的数据继续训练，也不能更新原模型。要做的是根据旧工具已经学到的参数，直接为新工具补出：

1. 一个 memory token 的 embedding，记作 $e_\star$；
2. TCRA 的一行新参数，记作 $(w_\star,\beta_\star)$；
3. 新工具、参数行和 token ID 之间的对应关系。

接入完成后，模型需要同时做到两件事：

- 该用新工具时，能够选中它；
- 选中以后，能够生成参数名和取值都正确的 JSON。

这里说的“不再训练”有严格含义：

- 不使用任何与新工具有关的训练样本，包括用户问题、标准调用和完整调用过程；
- 不计算损失，不做反向传播，也不启动优化器；
- 主干模型、旧工具 embedding、旧 TCRA 和 `<EOC>` 全部冻结；
- 只能读取工具说明，让冻结模型做前向计算，再对旧参数做检索、加权平均和缩放。

“训练时没见过”只针对 TapMem 的任务训练。它不表示主干模型在预训练阶段从未见过这个工具名或相关概念。

## 2. 先分清 embedding、hidden state 和 TCRA 参数

$e_i$、$h_b$ 和 $w_i$ 都是 $d$ 维向量，但作用完全不同；$\beta_i$ 只是一个标量。

| 记号 | 含义 | 是否随请求变化 | 是否保存在模型中 |
| --- | --- | --- | --- |
| $e_i$ | 工具 $i$ 的 memory token embedding | 否 | 是 |
| $h_b$ | 模型在第 $b$ 个工具选择位置算出的 hidden state | 是 | 否 |
| $w_i,\beta_i$ | TCRA 判断工具 $i$ 时使用的权重和偏置 | 否 | 是 |

在一次推理中，模型根据当前上下文得到：

$$
h_b=H_\theta(q,y_{<b})[-1].
$$

第一次选工具时，$h_b$ 来自格式化后 prompt 的最后一个有效位置，也就是正式推理中用于预测第一个工具 token 的位置。连续调用工具时，后面的 $h_b$ 来自已经生成的 `<EOC>` 位置。不能使用“即将生成的工具 token”的 hidden state，因为此时那个 token 还不存在。

TapMem 用 $h_b$ 和工具 embedding 做点积，得到工具 token 的语言模型分数：

$$
\ell^{\mathrm{LM}}_{b,i}=h_b^\top e_i.
$$

TCRA 则另外计算：

$$
z_{b,i}=w_i^\top h_b+\beta_i.
$$

二者最关键的区别是：

- $h_b$ 是当前请求临时产生的状态；
- $e_i$ 是长期绑定在工具 token 上的模型参数；
- $w_i,\beta_i$ 是 TCRA 的长期参数。

TapMem 当前让工具 token 的输入 embedding 和输出 embedding 共用同一组参数（coupled）。因此，这个向量既参与选择工具，也会在工具被选中后作为下一步输入，继续影响 JSON 参数生成。冷启动不能只补 TCRA；只把工具选对、却生成不出正确参数，仍然算失败。

每接入一个新工具，需要新增 $2d+1$ 个参数：$e_\star$ 和 $w_\star$ 各有 $d$ 个数，再加一个偏置 $\beta_\star$。

## 3. 整体思路：先找参照，再分别生成两类参数

方案分成“建立旧工具参照库”和“接入新工具”两步。

一句话概括：新工具自己的文字说明给出 embedding 的起点，相似旧工具提供训练后形成的校正量；固定提示产生的 hidden state 给出 TCRA 的起点，相似旧工具再提供 TCRA 校正量。两类参数分别计算，不把 hidden state 当成 embedding。

### 3.1 建立旧工具参照库

这一步只做一次。对每个旧工具 $i$，保存：

- 统一格式的工具说明；
- 已经训练好的 $e_i,w_i,\beta_i$；
- 由工具名称、功能和参数得到的文本向量 $g_i$；
- 冻结模型读完固定提示后，在两类工具选择位置得到的平均 hidden state $p_i^{\mathrm{first}}$ 和 $p_i^{\mathrm{next}}$。

#### 3.1.1 怎样得到文本向量 $g_i$

$g_i$ 只由工具说明和主干模型原有的词表 embedding 算出，不需要让模型做前向计算。

第一步，先把工具说明拆成四个字段，不能直接把整段 JSON 拼起来以后统一截断：

1. 工具名；
2. 功能说明；
3. 参数名；
4. 参数类型，以及该参数是必填还是可选。

例如 `weighted_average` 会被整理成：

```text
工具名：
weighted_average
weighted average

功能说明：
Calculate the weighted average of a list of values.

参数名：
values
weights

参数类型和是否必填：
values: list[float], required
weights: list[float], required
```

工具名和参数名同时保留原始写法与拆词后的写法，例如同时保留 `weighted_average` 和 `weighted average`。这样既保留 API 中的准确名字，也让模型看到容易理解的普通单词。工具名、参数名和参数类型不能截断；功能说明过长时，首版最多保留前 $128$ 个普通文本 token。

第二步，使用 TapMem checkpoint 对应的原 tokenizer 分别编码四个字段，并设置 `add_special_tokens=False`。这里要去掉 BOS、EOS、PAD、聊天模板控制符、memory token 和 `<EOC>`，只保留普通文本 token。

第三步，从冻结主干模型的原始输入 embedding 表中查出这些 token 的向量。把这张表记作：

$$
E_{\mathrm{vocab}}\in\mathbb R^{V\times d}.
$$

字段 $f$ 的 token 集合记作 $T_i^{(f)}$，先在字段内部求平均：

$$
m_i^{(f)}
=
\frac{1}{\lvert T_i^{(f)}\rvert}
\sum_{t\in T_i^{(f)}}E_{\mathrm{vocab}}[t].
$$

参数名和参数类型需要多做一层平均：先对每个参数单独求平均，再在所有参数之间求平均。这样，一个名字很长的参数不会因为 token 更多而占据过大权重。

最后合成：

$$
g_i
=
0.30m_i^{(\mathrm{name})}
+0.25m_i^{(\mathrm{desc})}
+0.35m_i^{(\mathrm{arg})}
+0.10m_i^{(\mathrm{type})}.
$$

如果某个字段为空，就去掉该项，并把其余权重重新归一化到和为 $1$。保存未经单位化的 $g_i$，供第 4.1 节构造新 embedding；计算余弦相似度时，再临时使用：

$$
\widehat g_i
=
\frac{g_i}{\lVert g_i\rVert_2+\epsilon}.
$$

$g_i$ 与工具 embedding $e_i$ 的维度都是 $d$，原因只是它们都来自同一个模型的 embedding 坐标系。$g_i$ 是工具说明中普通词向量的平均结果，$e_i$ 则是 TapMem 训练出来并绑定到 memory token 上的参数，二者不能混为一谈。

#### 3.1.2 怎样得到平均 hidden state $p_i$

$p_i^{\mathrm{first}}$ 和 $p_i^{\mathrm{next}}$ 必须让冻结的 TapMem 做前向计算。它们分别模拟“第一次选工具”和“在 `<EOC>` 后继续选工具”两种情况。

前向时使用 `eval()` 和 `torch.inference_mode()`，所有参数保持冻结。读取的是最后一层中、真正送入 LM head 和 TCRA 的那份 hidden state，不读取中间层，也不对整个提示的所有位置做平均。

下面的 $\operatorname{Norm}$ 表示单位化：

$$
\operatorname{Norm}(v)
=
\frac{v}{\lVert v\rVert_2+\epsilon}.
$$

**第一次选工具。** 准备 $M=4$ 个固定提示模板。模板只是换一种说法，内容都只来自工具说明，不由另一个模型生成。例如：

```text
<user>
Task name: {工具名}
Task description: {功能说明}
Required inputs: {参数名、参数类型和是否必填}
Choose the appropriate tool and produce its call.
<assistant>
```

实际编码时必须使用 compositional 正式推理所用的同一套聊天模板。设第 $m$ 个模板编码后为 $x_{i,m}^{\mathrm{first}}$，工具 token 即将生成的位置为 $b_{i,m}^{\mathrm{first}}$，则读取：

$$
h_{i,m}^{\mathrm{first}}
=
H_\theta
\left(x_{i,m}^{\mathrm{first}}\right)
\left[b_{i,m}^{\mathrm{first}}\right].
$$

$b_{i,m}^{\mathrm{first}}$ 就是正式推理中用来计算下一个 token 分数的位置，通常是格式化后 prompt 的最后一个非 PAD token。这里还没有生成任何工具 token。

把四个模板得到的 hidden state 分别缩放到单位长度，再求平均并再次单位化：

$$
p_i^{\mathrm{first}}
=
\operatorname{Norm}
\left(
\frac{1}{M}
\sum_{m=1}^{M}
\operatorname{Norm}
\left(h_{i,m}^{\mathrm{first}}\right)
\right).
$$

**在 `<EOC>` 后继续选工具。** 这种情况还需要在目标工具之前放一个已经存在的旧工具调用。为避免结果只适合某一个旧工具，预先从旧工具库中固定选择 $A=4$ 个锚点工具。首版可以先选最接近旧工具文本向量中心的工具，再逐个选择与已有锚点最远的工具，直到得到四个锚点。整个锚点集合在看到新工具之前就固定下来。

每个锚点只需要一个符合其参数类型的占位调用，不需要真实训练样本，也不执行工具。例如：

- 整数填 `1`，浮点数填 `1.0`；
- 字符串填 `"x"`，布尔值填 `true`；
- 列表放入一个对应类型的元素；
- 必填参数全部保留，可选参数省略。

用户消息同时描述锚点任务和目标任务：

```text
<user>
First complete: {锚点工具的功能说明和参数}.
Then complete: {目标工具的名称、功能说明和参数}.
<assistant>
```

assistant 前缀则放入：

```text
<锚点工具的 memory token><占位 JSON><EOC>
```

前向只运行到 `<EOC>`，不追加目标工具 token。设第 $a$ 个锚点和第 $m$ 个模板组成的输入为 $x_{i,a,m}^{\mathrm{next}}$，则读取 `<EOC>` 位置上、用于预测下一个 token 的 hidden state：

$$
h_{i,a,m}^{\mathrm{next}}
=
H_\theta
\left(x_{i,a,m}^{\mathrm{next}}\right)
\left[\operatorname{pos}(\mathrm{EOC})\right].
$$

然后在所有锚点和模板上平均：

$$
p_i^{\mathrm{next}}
=
\operatorname{Norm}
\left(
\frac{1}{AM}
\sum_{a=1}^{A}
\sum_{m=1}^{M}
\operatorname{Norm}
\left(h_{i,a,m}^{\mathrm{next}}\right)
\right).
$$

如果目标旧工具本身恰好是某个锚点，就跳过这个锚点，并使用预先排在下一位的候补锚点。批量前向时，必须根据 attention mask 找到每条输入的真实末尾或 `<EOC>` 位置，不能直接读取补齐后的最后一列。

新工具使用完全相同的字段处理、提示模板、锚点工具和取值位置，得到 $g_\star$、$p_\star^{\mathrm{first}}$ 和 $p_\star^{\mathrm{next}}$。整个过程只把新工具说明写进普通文本，既不需要新工具 token，也不会读取“新工具 token 生成以后”的 hidden state。

三者的用途如下：

- $g_i$：普通词 embedding 的加权平均，用于文本相似度和新工具 embedding 的起点；
- $p_i^{\mathrm{first}}$、$p_i^{\mathrm{next}}$：固定提示在选工具位置产生的平均 hidden state，用于 hidden-state 相似度和 TCRA 参数的起点；
- 它们都只是构造新参数时使用的材料，不是新增的模型参数，也不参与接入后的正常推理。

### 3.2 接入新工具

新工具到来后，按同样办法得到 $g_\star$、$p_\star^{\mathrm{first}}$ 和 $p_\star^{\mathrm{next}}$，再从旧工具中找出最相近的 $k$ 个。相似度同时考虑：

- 工具说明是否接近；
- 模型在两种选工具场景下的反应是否接近，也就是相应 hidden state 是否接近。

首版相似度可以直接写成：

$$
s_i
=0.5\cos(g_\star,g_i)
+0.25\cos(p_\star^{\mathrm{first}},p_i^{\mathrm{first}})
+0.25\cos(p_\star^{\mathrm{next}},p_i^{\mathrm{next}}).
$$

取分数最高的 4 个旧工具，再对这 4 个分数做 softmax，得到权重 $a_i$，并令 $\sum_i a_i=1$。例如，新工具是 `weighted_average`，参照工具可能包括 `average` 和 `calculate_grade`。新工具自己的说明负责提供 `weights` 这类新增信息，旧工具则提供 TapMem 已经学会的工具选择和 JSON 生成习惯。

同时计算一个可信度 $c_\star\in[0,1]$。新工具与旧工具越接近，$c_\star$ 越高。具体做法是先在旧工具上轮流模拟冷启动，记录每次能找到的最高相似度；再把新工具的最高相似度放到这个旧工具分布中，最低一成对应 0，最高一成对应 1，中间线性换算。softmax 的温度系数和可信度阈值也只能用这套旧工具模拟确定，不能查看新工具测试结果后再调。

## 4. 新参数具体怎么生成

### 4.1 生成新工具的 embedding

先按第 3.1.1 节的流程，从新工具说明得到 $g_\star$。这一步只查主干模型原有的词表 embedding，不运行模型前向，也不使用 memory-token embedding。

旧工具的 $g_i$ 与训练好的 $e_i$ 虽然都在 $d$ 维空间中，但均值和长度通常不同。先用旧工具统计量做一次固定的平移和缩放：

$$
B_E(g)=\bar e+
\frac{r_e}{r_g+\epsilon}(g-\bar g).
$$

其中，$\bar g,\bar e$ 是旧工具两类向量的均值；$r_g$ 是所有 $\lVert g_i-\bar g\rVert_2$ 的中位数，$r_e$ 同理。这里只是按旧工具的统计量做平移和缩放，不需要拟合任何新参数。

接着看每个旧工具在训练后相对这个初始位置发生了什么变化：

$$
d_i^E=e_i-B_E(g_i).
$$

新工具的 embedding 定义为：

$$
\widetilde e_\star
=
B_E(g_\star)
+c_\star\sum_i a_i d_i^E.
$$

直观上，第一项来自新工具自己的文字说明，第二项借用了相似旧工具在 TapMem 训练中形成的校正量。最后把 $\lVert\widetilde e_\star-\bar e\rVert_2$ 限制在旧工具对应距离的第 5 到第 95 百分位之间，得到 $e_\star$，避免新 token 的语言模型分数异常大或异常小。

这一整步只在 embedding 空间里运算，没有把 hidden state 加到 $e_\star$ 上。

### 4.2 生成 TCRA 的新参数

TCRA 走另一条路。把第 3.1.2 节得到的两类平均 hidden state 按一半对一半合并：

$$
p_i
=
\operatorname{Norm}
\left(
0.5p_i^{\mathrm{first}}
+0.5p_i^{\mathrm{next}}
\right).
$$

新工具的 $p_\star$ 也用相同办法得到。这些向量只用来构造分类器参数。

同样先对齐两组旧向量：一组是合成后的 $p_i$，另一组是已经训练好的 TCRA 权重。

$$
B_W(p)=\bar w+
\frac{r_w}{r_p+\epsilon}(p-\bar p).
$$

这里的均值和中位距离，与 embedding 部分使用相同的算法计算。

再计算旧工具 TCRA 权重相对这个初始值的校正量：

$$
d_i^W=w_i-B_W(p_i).
$$

新工具的 TCRA 权重和偏置为：

$$
\widetilde w_\star
=
B_W(p_\star)
+c_\star\sum_i a_i d_i^W,
$$

$$
\beta_\star
=
\bar\beta+
c_\star\sum_i a_i(\beta_i-\bar\beta).
$$

最后把 $\lVert\widetilde w_\star-\bar w\rVert_2$ 限制在旧 TCRA 权重对应距离的第 5 到第 95 百分位之间，得到 $w_\star$。

这里的关系要再次说清楚：

- $p_\star$ 是从若干次前向计算中临时得到的平均 hidden state；
- 真正写入模型的是 $w_\star,\beta_\star$；
- $p_\star$ 不会被当作 $e_\star$，也不会成为一个新的 memory token。

## 5. 新 TCRA 不能直接和旧工具一起做 softmax

旧 TCRA 的分数只在 $K$ 个旧工具之间归一化。若把新工具直接塞进同一个 softmax，分母就会改变。这样旧参数虽然没动，旧工具得到的修正值也会变。

为了不破坏旧工具，旧分支始终只用原来的 $K$ 个工具作为基准：

$$
A_{\mathrm{old}}(h_b)
=
\operatorname{logsumexp}(z_{b,1:K})-\log K.
$$

旧工具继续使用原来的修正：

$$
\Delta_{b,i}
=
\alpha\bigl(z_{b,i}-A_{\mathrm{old}}(h_b)\bigr).
$$

新工具使用同一个旧工具基准，但乘上接入可信度：

$$
\Delta_{b,\star}
=
\alpha c_\star
\bigl(z_{b,\star}-A_{\mathrm{old}}(h_b)\bigr).
$$

这样，只要到这个选择位置为止的上下文相同、算出的 $h_b$ 也相同，接入新工具前后，旧工具的 TCRA 修正值就完全相同。

这套计算不是 $K+M$ 个工具上的联合概率。一次接入多个新工具时，它们最终通过“语言模型分数加 TCRA 修正”互相竞争，因此必须专门测试新工具之间的混淆。

还要注意，$c_\star=0$ 只会关闭新工具的 TCRA 修正，语言模型仍有可能选中它。可信度过低时，应直接屏蔽这个 memory token，并退回到把工具说明放进上下文的普通调用方式。

## 6. 代码需要怎样配合

### 6.1 不要移动原来的 `<EOC>`

保存 checkpoint 时，要把旧工具、新工具和 `<EOC>` 分开记：

```text
E_old, E_new, EOC
W_old, beta_old
W_new, beta_new
```

再用一份明确的登记表保存：

```text
工具名 -> 旧/新工具 -> 参数行 -> reserved token ID
```

接入新工具时，优先启用一个预留但尚未使用的 token ID（reserved token ID）。不能把旧 `<EOC>` 的位置改成新工具，也不能再假设“工具编号、参数行号和 token ID 永远相同”。

### 6.2 旧分支和新分支分开算

checkpoint 必须记录旧工具数量 $K$。推理时，旧 TCRA 仍按 $K$ 类计算，新 TCRA 按上一节的公式单独计算。不能先把新旧参数拼起来，再调用现有的统一 `log_softmax`。

一个工具冷启动得到的参数默认不能继续充当后续新工具的参照。否则多轮接入后，早期误差会不断传下去。参照库只使用真正训练过的旧工具。

### 6.3 推理流程

每次到达工具选择位置时：

1. 根据当前上下文算出动态的 $h_b$；
2. 用 $h_b$ 与所有工具 embedding 计算语言模型分数；
3. 分别计算旧 TCRA 和新 TCRA 的修正；
4. 把修正加到各自的 memory-token 分数上；
5. 选择工具 token；
6. 如果选中新工具，固定的 $e_\star$ 会作为下一步输入，继续生成 JSON 参数；
7. 参数结束后仍使用原来的 `<EOC>`。

建立参照库和接入工具时都应使用 `eval()` 和 `torch.inference_mode()`。最终 checkpoint 还应保存工具顺序、token ID、$K$、构造参数、可信度，以及所用 tokenizer 和文本编码方式的标识。

## 7. 实验必须真正模拟“训练时没见过”

主实验直接使用现有的 Llama-1B TokMem 和 TapMem checkpoint。它们只训练过
tools 51–100，因此这 50 个工具作为旧工具。新工具从 tools 1–50 中选择，
固定名单保存在：

```text
compositional/cold_start_selected_tools_20.json
```

这 20 个工具没有对应的 memory-token embedding，也没有对应的 TCRA 参数行，
因此是训练时真正不存在的工具。接入时只能读取工具说明；新参数生成完成后，
才允许读取测试请求和正确答案。

在 tools 51–100 内部再做 40/10 划分可以作为补充实验，但不作为本轮主实验。
不能简单地在一个训练过全部 50 个工具的 checkpoint 中遮住 10 行，因为这些
工具已经参与过联合训练。

旧工具内部可以轮流做一次“模拟冷启动”：每次暂时拿走一个旧工具，假装它是新工具，再用其余旧工具重建它。这样可以确定 $k$、字段权重和可信度阈值。但这只能用于选设置，不能当作冷启动效果的主要证据，因为这些旧工具原本参加过联合训练。

### 7.1 至少需要的对照

- 随机初始化新 embedding，且不给新工具 TCRA 修正；
- 只用工具说明生成 embedding；
- 直接复制最相近旧工具的参数；
- 按说明相似度平均旧工具参数；
- 只生成新 embedding，不增加 TCRA；
- 本文完整方案；
- 直接改成 $K+M$ 类 softmax；
- 把新工具说明放进上下文；
- 允许使用新工具样本训练的结果，作为参考上限。

### 7.2 不能只看工具有没有选对

至少报告：

- Tool F1 和整条工具序列完全正确的比例；
- 新工具各自的 precision、recall 和 F1；
- Arguments F1、完整调用完全正确的比例和 JSON 解析失败率；
- 参数名集合完全正确的比例；
- 接入前后旧工具的性能变化；
- 旧工具请求被误分给新工具的比例；
- 第一次选工具与 `<EOC>` 后选工具的准确率；
- 一次接入 1、5、10、20 个新工具时的混淆情况；
- 单个工具的接入时间、参数量和显存开销。

主测试集沿用仓库原来的 XLAM 合成方法，但由独立脚本生成，不修改
`xlam_datasets.py`：

```text
compositional/synthesize_cold_start_mixed_test.py
```

默认生成 500 条样本。双工具样本为 200 条，全部是一个新工具和一个旧工具；
三工具样本为 200 条，其中两种新旧组成各 100 条；四工具样本为 100 条，
组成如下：

| 新工具数 | 旧工具数 | 样本数 |
| ---: | ---: | ---: |
| 2 | 2 | 34 |
| 1 | 3 | 33 |
| 3 | 1 | 33 |

工具顺序随机打乱，因此测试集中会自然出现“旧工具到新工具”“新工具到旧工具”
和“新工具到新工具”等切换。只有 Tool F1 与 Arguments F1 都提升，同时旧工具
性能基本不下降，才能说这个方案缓解了冷启动问题。

## 8. 方案的边界和建议顺序

这个方案依赖两个前提：

- 主干模型能够理解新工具的自然语言说明；
- 新工具能从旧工具中找到语义或参数结构相近的参照。

如果工具说明含糊、参数名是无意义编码、新旧工具表面相似但行为相反，或者一次加入许多几乎相同的新工具，效果可能很差。此外，TapMem 只用一个工具 embedding 同时负责“选中工具”和“帮助生成参数”。工具说明很长、参数很多时，一个向量可能装不下所有信息。

APIGen 实验只能说明模型会不会选工具、能不能生成 JSON，不能证明它会正确执行真实 API、处理真实返回值，也不能证明它能在工具调用失败后恢复。

建议按下面的顺序实现：

1. 先固定工具登记表、token ID 和 `<EOC>`，完成 checkpoint 扩容；
2. 先做“相似旧参数加权平均”这个最简单的无需训练基线；
3. 再加入第 4 节的 embedding 构造；
4. 最后加入 TCRA 新行，以及保持旧工具 TCRA 修正值不变的计算方式；
5. 在 tools 51–100 checkpoint 上一次接入选定的 20 个新工具，同时检查
   Tool F1 和 Arguments F1。

第一轮实验只需要回答两个问题：新工具能不能被选中，以及选中以后能不能生成正确参数。两个问题都通过，方案才算真正解决了一部分冷启动问题。
