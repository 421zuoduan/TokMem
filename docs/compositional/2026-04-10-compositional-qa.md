# Compositional 问答记录

这份文档专门记录本轮关于 `compositional/` 数据构造、TokMem 训练格式与论文设定的问答，回答尽量简短。

## Q1. 现在 scripts 里 comp 训练是使用 `4calls` 进行训练吗？

是。

- 当前 launcher 会生成并读取 `..._4calls.json`
- `4calls` 的意思是每条样本最多 `4` 个 function calls，不是每条都恰好 `4` 个 calls

## Q2. 当前选取的 50 个 benchmark tools，是否满足每个 tool 都有 `100` 个训练样本和 `10` 个测试样本？

不满足。

- 如果看当前主 benchmark 使用的 `51-100` 这组 tools，它们都达不到 `110` 条单工具原始样本
- 如果看 `1-50` 这组，从原始样本量上可以满足 `100+10`
- 但当前脚本本身也没有按 `100/10` 这个规则生成数据

## Q3. 第二组工具抽取逻辑有什么问题，导致最终数据不符合 `100 train + 10 test`？

主要有四个问题：

1. 选 tool 时只按频次排名切片 `51-100`，没有先过滤 `>=110` 样本量的 tool。
2. `max_samples_per_tool` 被脚本固定成 `50`，先天不可能得到 `100+10`。
3. 之后是按全局比例切分，不是按每个 tool 固定切出 `100/10`。
4. 测试集默认只保留合成的 multi-tool compositional 样本，不是每个 tool 各拿 10 条独立 test。

## Q4. 论文里 compositional 的 adaptation 阶段和正式训练阶段分别用到多少数据量？

论文口径是：

- adaptation phase：`50` 个 held-out auxiliary tools，`5,000` 个样本
- 正式 compositional benchmark：`50` 个 benchmark tools，`5,000` 个训练 queries，`500` 个测试 queries

附录还补充：

- 正式 benchmark 先对 `50` 个 tools 每个收集 `50` 个 query-call pairs
- 所以底层先有 `2,500` 个原始单工具样本
- 再据此合成最终的 `5,000 / 500` compositional queries

## Q5. `held-out` 是什么意思？

这里的意思是：

- 这批工具被单独留出来
- 不和正式 benchmark 的那 `50` 个 tools 重叠
- 只用于 adaptation，不参与后面的主评测

它的重点是“与正式评测集隔离”，不是“模型永远没见过任何相关模式”。

## Q6. adaptation 和正式 compositional 训练阶段，训练样本数量有区别吗？

按论文口径，训练样本数量基本没有区别：

- adaptation：`5,000` 个训练样本
- 正式 compositional 训练：`5,000` 个训练 queries

主要区别在于：

- adaptation 用的是 held-out auxiliary tools
- 正式阶段用的是 benchmark tools

## Q7. 原始数据是什么样子的？作者是怎么构建 compositional 训练数据的？

原始 APIGen / XLAM 一行数据主要有三个字段：

- `query`：自然语言问题
- `answers`：gold tool calls，形如 `{"name": ..., "arguments": ...}`
- `tools`：候选工具定义，含 `name / description / parameters`

作者/代码构造 compositional 数据的大致流程是：

1. 先从原始数据里抽单工具 procedure 样本
2. 再把多个单工具样本拼成一个多步骤 query
3. 最后把它们转成 TokMem 训练用的 `tool token + 参数 JSON` 序列

## Q8. 原始数据里本来就有 `<memory token> + response1 + <memory token> + response2` 这种格式吗？

没有。

原始数据只有：

- 自然语言 `query`
- 结构化的 `answers`
- 候选 `tools`

`memory token` / `tool token` 是 TokMem 在模型侧额外引入的表示，不是原始数据自带的。

## Q9. APIGen 高频工具是怎么抽取的？

核心思想是只看 `answers`，不看 `tools`。

具体规则：

1. 遍历原始样本
2. 解析 `answers`
3. 如果该样本里真实被调用的 tool “有且仅有一个唯一 tool”，则给这个 tool 计数 `+1`
4. 统计完后按频次排序
5. 排除 `search`
6. 取 `1-50` 或 `51-100`

注意：

- 这里是按“样本数”计数，不是按“调用次数”计数
- 同一个 tool 在一条样本里调两次，这条样本仍只记一次频次

## Q10. `tools` 字段的作用是什么？

它不是“告诉模型可以调用哪些模型”，而是“告诉求解器当前有哪些候选工具，以及每个工具的 schema”。

在这个 repo 里主要有两个用途：

- 从原始数据中抽取工具描述和参数定义
- 给 ICL / RAG baseline 拼 `Available Tools` prompt

TokMem 训练本身并不是逐条样本把整份 `tools` 列表喂进模型。

## Q11. 原始工具调用没有给一个现实世界可能出现的问题吗？

不是。

原始数据是有自然语言 `query` 的，只是很多 query 带有 benchmark / research-oriented 风格，不一定像真实产品里的用户请求那么自然。

所以更准确的说法是：

- 有现实世界式的自然语言问题
- 但整体仍是研究型数据集，不完全等于真实使用场景

## Q12. TokMem 训练时，模型是怎么学会“输出 tool token”的？

核心机制是“把正确 tool token 当作监督目标来预测”。

具体做法：

1. 每个 tool name 先映射到一个专属 reserved special token
2. 训练样本被构造成：
   `<user> query <assistant> [tool_token_1] {args_1} [tool_token_2] {args_2} ...`
3. loss 只监督后面的 `tool token + 参数 JSON`，不监督用户输入部分

所以模型学到的是：

- 面对某类 query
- 下一步应该先输出哪个 tool token
- 再输出该 tool 对应的参数

## Q13. 组合 compositional query 用的单工具样本池，是不是按 Q9 的规则得到的？

是。

- 先按 Q9 的规则抽出“唯一 tool”样本
- 再拆成：
  - 单次调用的单工具样本
  - 同一 tool 多次调用的单工具样本

它们共同构成后续 compositional synthesis 的原料池。

## Q14. 合成后的 compositional query 里，多个 tool 可能完全不相干，这会不会有问题？

会有这个问题。

当前实现主要是随机选多个单工具 query，再用 `Also / Lastly` 之类连接词拼起来，所以很多组合只是“并列多个小任务”，不一定存在真实依赖关系。

## Q15. 作者有没有考虑多个 tool 之间组合的合理性？

没有明显做严格约束。

作者考虑了：

- 每个 tool invocation 是 atomic procedure
- compositional query 需要多个 procedures
- 样本上限和数据泄漏控制

作者没有明显处理：

- 语义一致性
- 实体一致性
- 前后依赖
- tool A 输出作为 tool B 输入
- 执行可行性

论文的限制部分也承认目前是 research-oriented setting。

## Q16. 这种做法合理吗？如果不够合理，可以怎么改进？

要分目标看。

如果目标是：

- 测试 memory recall
- 测试模型能否连续输出多个 tool token 和参数

那是合理的，因为它是一个受控 benchmark。

如果目标是：

- 测真实 agent 多步骤任务能力
- 测真实工具链依赖

那就不够合理。

可改进方向：

- 要求多个 tool 共享同一实体或主题
- 强制后一个 tool 依赖前一个 tool 的输出
- 用 teacher LLM 合成更连贯的 query
- 执行工具或 mock 工具，过滤掉不可执行组合

## Q17. 展示一条完整的合成样本。

下面是一条当前 repo 里的真实合成训练样本：

```json
{
  "user_input": "Determine the integral of e^x from -1 to 1 using 15000 subdivisions. Also, determine if the year 1900 is a leap year. Lastly, is 192.168.1.1 a valid ip address for a local network?",
  "tools": [
    "trapezoidal_integration",
    "is_leap_year",
    "is_valid_ip_address"
  ],
  "function_calls": [
    "{\"func\": \"np.exp(x)\", \"a\": -1, \"b\": 1, \"n\": 15000}",
    "{\"year\": 1900}",
    "{\"ip\": \"192.168.1.1\"}"
  ]
}
```

它在训练时会被线性化成近似下面这种序列：

```text
<user> Determine the integral ... <assistant>
[tool_token_for_trapezoidal_integration] {"func":"np.exp(x)","a":-1,"b":1,"n":15000}
[tool_token_for_is_leap_year] {"year":1900}
[tool_token_for_is_valid_ip_address] {"ip":"192.168.1.1"}
<eot>
```

## Q18. 如何判断模型选对了 tool？

做法是：

1. 从模型输出序列里找所有 tool token
2. 把 tool token 映射回 tool name
3. 再和 gold tool names 做比较

指标上主要是 `Tool Prediction F1`。

## Q19. 如何判断 tool 的调用参数是正确的？

做法是把预测调用和 gold 调用都解析成结构化表示，再做标准化比较。

指标上主要是 `Argument Generation F1`。

为了容忍表面写法不同，repo 会把调用归一化成 AST / 规范化 JSON 后再算分。

## Q20. 如何判断 tool 的执行结果是正确的？

当前这套 repo / 论文并不真正执行工具。

所以它能判断的是：

- tool 选没选对
- 参数写没写对

但它不能直接判断：

- API 真执行后返回了什么
- 返回值是否正确
- 后续步骤是否真的消费了前一步输出

如果要判断 execution correctness，就需要补真实工具执行或 mock executor。

## Q21. 单 tool 样本池里的单次调用样本和同一 tool 的多次调用样本，后续处理有什么不同？

有，主要差别在数据整理阶段和后续合成阶段。

- 在高频 tool 计数阶段，两者都算“单 tool 样本”，每条样本只给该 tool 记一次频次
- 在整理阶段：
  - 单次调用样本进入 `single_samples`
  - 同一 tool 多次调用样本进入 `multi_call_samples`
- 在训练集里：
  - 单次调用样本会变成 `tools=[tool]`、`function_calls=[一组参数]`
  - 多次调用样本会变成 `tools=[tool, tool, ...]`、`function_calls=[多组参数]`
- 在测试集里：
  - 两者都不会原样保留
  - 它们只作为后续合成 multi-tool compositional 样本的原料
- 在合成 multi-tool 样本时：
  - 单次调用样本每次只贡献 1 次调用
  - 多次调用样本可能让同一个 tool 一次性贡献多次调用

## Q22. 之前说正式训练数据可能不满足要求，是不是没有考虑后续合成数据？

是，必须区分两个口径。

### 口径 1：原始单 tool 原料是否满足每个 tool 有 `100 train + 10 test`

这个口径下，前面的结论成立：

- `1-50` 这组原始单工具样本量够
- `51-100` 这组原始单工具样本量不够
- 而且脚本还把 `max_samples_per_tool` 限成了 `50`

### 口径 2：最终生成后的训练/测试集里，每个 tool 是否至少出现 `100/10` 次

这个口径下，当前最终数据是满足的，因为有后续合成数据把覆盖次数补上来了。

也就是说：

- 如果看原始单 tool 原料，正式 benchmark 那组不满足 `100+10`
- 如果看最终合成后的训练/测试集覆盖次数，当前数据里每个 tool 的出现次数已经超过 `100/10`

所以之后讨论“满足不满足要求”时，必须先说清楚问的是：

- `raw single-tool support`
- 还是 `final synthesized dataset coverage`

## Q23. 测试数据一共有多少条？都是合成的 compositional 数据吗？

- 每套 test 是 `500` 条
- `tools1-50` 一套 `500`
- `tools51-100` 一套 `500`
- 当前 repo 里两套 test 都是`纯合成的 multi-tool compositional 数据`
- 不会把原始单 tool 样本原样放进 test

## Q24. 当前筛出来的高频 tools 有什么问题？需要联网/外部 API 的 tool 怎么处理？

主要问题有：

- 高频定义很窄，只看 `answers` 里“恰好只有一个唯一 tool”的样本频次
- 只排除了名字叫 `search` 的 tool，不是把所有外部 API / 外部世界 tool 都排掉
- `51-100` 这组原始单 tool 频次不足以支撑每个 tool 原生 `100 train + 10 test`
- 当前脚本还把每个 tool 的原始样本先截到 `50`

对于 `whois`、`whole_foods_order`、`directions_between_2_locations` 这类 tool，这套 benchmark / repo 只要求：

- 选对 tool
- 填对 argument

它`不要求真正联网搜索或真实执行工具`。

## Q25. `whois`、`whole_foods_order`、`directions_between_2_locations` 的具体样本是什么样？

原始 APIGen 里可参考这些真实样本：

- `whois`：`id=52402`
  - query: `Fetch WhoIS lookup information for localbusiness.com, localbusiness.ca, and localbusiness.co.uk.`
  - arguments: 多次调用，每次只填 `domain`
- `whole_foods_order`：`id=19879`
  - query: `Place a medium-sized order at the Whole Foods store located in San Francisco ...`
  - arguments: `store_location`、`item_list`、`order_size`
- `directions_between_2_locations`：`id=31468`
  - query: `What are the directions from Lagos ... to Abuja ... in kilometers?`
  - arguments: `start_lat`、`start_lon`、`end_lat`、`end_lon`、`distance_unit`

这些例子都说明：监督目标主要是`参数填写正确`，不是执行结果验证。

## Q26. 在一个 tool 的 arguments 生成完后，模型怎么知道该生成下一个 tool token？

不是靠显式规则判断，而是靠自回归 next-token 学习：

- 训练序列本来就是 `[tool_1][args_1][tool_2][args_2]...[tool_k][args_k][eot]`
- 所以在 `args_1` 结尾位置，gold 下一个 token 就是 `tool_2`
- 在最后一个 `args_k` 结尾位置，gold 下一个 token 就是 `<eot>`

模型学到的是这种序列模式，不是显式解析 JSON 完整性后再切换。

## Q27. 训练和测试时的输入输出 token 形式分别是什么？

训练时：

```text
输入 = <user prompt>[tool_1][args_1][tool_2][args_2]...[tool_k][args_k]<eot>
监督 = 用户部分 label 为 -100，只监督后半段 [tool][args]...[eot]
```

测试时：

```text
输入 = <user prompt>
输出 = 模型自回归生成 [tool_1][args_1][tool_2][args_2]...[tool_k][args_k]<eot>
```

其中 `[tool_i]` 是某个 tool 对应的一个 reserved special token，`[args_i]` 是参数 JSON 的 token 序列。

## Q28. `args_1` 和 `tool_2` 中间有额外的结束标志吗？

没有。

当前格式就是直接：

```text
[tool_1][args_1][tool_2][args_2]...[tool_k][args_k]<eot>
```

所以边界是隐式的：

- 非最后一个 call：`args_i` 后面直接接下一个 `tool token`
- 最后一个 call：`args_k` 后面直接接 `<eot>`

## Q29. 论文和代码里分别怎么判断模型有效性？用了什么指标？

论文在 compositional setting 里主要看两个指标：

- `Tool Prediction F1`
- `Argument Generation F1`

并且会先把 gold / prediction 归一化成 AST 再比较。

当前 repo 除了这两个核心思路外，还额外输出：

- `Exact Match Accuracy`
- `Tool Prediction Accuracy`
- `Average F1 / Precision / Recall`
- `Average Tool F1 / Precision / Recall`
- `Parse Error Rate`
- 按 `1/2/3/4 call` 分桶的 breakdown

它们共同点是：`都不真正执行工具`，主要比较 tool 和 arguments。

## Q30. adaptation 在 TokMem 的训练过程中有什么作用？

`adaptation` 的作用是一个 compositional warm-up。

- 它先用一组 `held-out` 的辅助 tools，让 backbone 学会 `[tool][args][tool][args]...` 这种生成结构
- 然后再进入正式 benchmark 的 memory acquisition

所以它更像：

- 先教模型“怎么交错生成 tool token 和参数”
- 再教模型“正式那批 tools 的具体记忆”

在当前 repo 的 TokMem launcher 里，它大致对应第一轮 `1-50:1`；后面 `51-100:3` 更接近正式训练阶段。
