# Compositional Data And QA

## 说明

这份文档整理自 docs/compositional/ 下多份按日期命名的草稿、设计和计划文档，按主题合并，便于后续集中查阅。

## 来源文档

- 2026-04-09-compositional-xlam-apigen-dataset-notes.md
- 2026-04-10-compositional-qa.md

---

## 原文：2026-04-09-compositional-xlam-apigen-dataset-notes.md

## Compositional 使用的原始数据集与 100 Tools 抽取说明

### 1. 先说结论

- `compositional/` 当前使用的原始数据集是本地的 `datasets/xlam-function-calling-60k`，也就是 XLAM / APIGen 的 function-calling 数据。
- 论文正文真正作为 compositional benchmark 评测的是 50 个 tools；仓库为了 TokMem 的 adaptation phase，又额外准备了另外 50 个互不重叠的 tools，所以源码和脚本里会看到总共 `100 tools`。
- 这 `100 tools` 不是随机抽的，而是从原始数据的 `train` split 中，按“单工具样本里的真实调用频次”降序选出来的 top-100，并且显式排除了 `search`。
- 当前仓库默认把这 100 个 tools 划成两组：
  - `1-50`：adaptation / auxiliary tools
  - `51-100`：主 benchmark tools

### 2. 数据与脚本入口

#### 2.1 原始数据入口

- 原始数据目录：`datasets/xlam-function-calling-60k/data/`
- 数据集 README：`datasets/xlam-function-calling-60k/README.md`
- 处理脚本：`compositional/xlam_datasets.py`

#### 2.2 生成后的 compositional 数据

默认脚本会把处理结果写到 `compositional/data/`：

- `tool_descriptions_tools1-50.json`
- `tool_descriptions_tools51-100.json`
- `training/function_calling_train_tools1-50_4calls.json`
- `training/function_calling_train_tools51-100_4calls.json`
- `test/function_calling_test_tools1-50_4calls.json`
- `test/function_calling_test_tools51-100_4calls.json`

文件名里的 `Ncalls` 由 `train_max_function_calls` / `test_max_function_calls` 决定，所以同一套脚本也可以生成 `..._10calls.json` 这类文件。

#### 2.3 维护脚本里的固定参数

`scripts/download/prepare_xlam_dataset.sh` 和 `scripts/compositional/llama_1b/*.sh` 使用的是同一组核心参数：

```bash
python xlam_datasets.py \
  --top_k 1-50 或 51-100 \
  --max_samples_per_tool 50 \
  --train_size 5000 \
  --test_size 500 \
  --train_max_function_calls 4 \
  --test_max_function_calls 4 \
  --train_multi_tool_ratios 0.5,0.5 \
  --test_multi_tool_ratios 0.5,0.5
```

这组参数基本对应论文附录里的设置：每个 tool 保留 50 个 query-call pairs，并把最终 train / test 规模控制在 `5000 / 500`。

当前 `xlam_datasets.py` 的 ratio 语义是：

- `train_multi_tool_ratios` / `test_multi_tool_ratios` 接受任意长度的逗号分隔比例串
- 如果 ratio 串长度是 `k`，它对应 `2-tool, 3-tool, ..., (k+1)-tool`
- 现有 maintained launcher 继续使用 `0.5,0.5`，也就是 `2-tool` 和 `3-tool`
- ratio 之和需要接近 `1.0`
- 超出当前 split 的 `max_function_calls` 或可用单工具池上限的 tool-count bucket 会被丢弃，剩余 bucket 按原权重重归一化

### 3. 原始 XLAM / APIGen 数据长什么样

根据 `datasets/xlam-function-calling-60k/README.md`，默认配置的主字段如下：

| 字段 | parquet 中类型 | 含义 | compositional 是否直接用到 |
| --- | --- | --- | --- |
| `id` | `int64` | 样本 ID | 会保留用于中间处理 |
| `query` | `string` | 自然语言用户请求 | 会直接用作后续 `user_input` 的来源 |
| `answers` | `string` | JSON 字符串，表示 gold tool calls | 是，源码用它来判断“真实调用了哪些 tools” |
| `tools` | `string` | JSON 字符串，表示该样本附带的候选工具定义 | 是，源码用它来抽取 tool description / parameter schema |

数据集 README 还给了一个 `sft_format` 配置，比默认配置额外多两个字段：

| 字段 | 含义 | compositional 是否使用 |
| --- | --- | --- |
| `prompt` | SFT prompt | 否 |
| `response` | SFT response | 否 |

默认 split 规模如下：

| split | 样本数 |
| --- | --- |
| `train` | 59,000 |
| `validation` | 500 |
| `test` | 500 |

#### 3.1 `answers` 里已经标注了什么

`answers` 是“真实调用记录”，每个元素至少有：

| 子字段 | 含义 |
| --- | --- |
| `name` | 实际调用的 tool 名称 |
| `arguments` | 该次调用的参数字典 |

#### 3.2 `tools` 里已经标注了什么

`tools` 是“候选工具定义列表”，每个 tool 一般包含：

| 子字段 | 含义 |
| --- | --- |
| `name` | tool 名称 |
| `description` | tool 功能描述 |
| `parameters` | 参数 schema |

而 `parameters` 里每个参数通常又带：

| 子字段 | 含义 |
| --- | --- |
| `description` | 参数语义说明 |
| `type` | 参数类型 |
| `default` | 默认值，若原数据里提供 |

#### 3.3 一个容易误解但很关键的点

源码在统计 tool 频次时看的是 `answers`，不是 `tools`。

原因是原始样本的 `tools` 往往是“候选工具列表”，里面可能包含没有真正被调用的其他工具；只有 `answers` 才表示该条样本里真正执行了哪些 tool calls。这个细节直接决定了“100 tools 是怎么抽出来的”。

### 4. 原始样本展示

下面都来自本地原始 parquet。

#### 4.1 单工具、单次调用

```json
{
  "id": 8249,
  "query": "I'm looking for the next greater element for each number in this list: 15, 25, 35, 45, 55.",
  "answers": [
    {
      "name": "find_next_greater_element",
      "arguments": {
        "nums": [15, 25, 35, 45, 55]
      }
    }
  ],
  "tools": [
    {
      "name": "find_next_greater_element",
      "description": "Finds the next greater element for each element in a list.",
      "parameters": {
        "nums": {
          "description": "The list of numbers.",
          "type": "List[int]"
        }
      }
    }
  ]
}
```

#### 4.2 单工具、同一 tool 多次调用

```json
{
  "id": 27762,
  "query": "What is the time zone of London and Tokyo at the current moment?",
  "answers": [
    {
      "name": "timezone_for_location",
      "arguments": {
        "location": "London",
        "area": "Europe"
      }
    },
    {
      "name": "timezone_for_location",
      "arguments": {
        "location": "Tokyo",
        "area": "Asia"
      }
    }
  ]
}
```

这类样本在论文和源码里都被视为“非 compositional 的单工具使用”，因为虽然有多次 call，但都属于同一个 tool。

#### 4.3 一条原始样本里含多个不同 tools

```json
{
  "id": 53263,
  "query": "Update the endpoint with info1 as 'update1', info2 as 'update2', and info3 as 'update3'. Also, find the blank label sheet brands for A4 format.",
  "answers": [
    {
      "name": "update",
      "arguments": {
        "info3": "update3",
        "info1": "update1",
        "info2": "update2"
      }
    },
    {
      "name": "label_template_brands",
      "arguments": {
        "format": "A4"
      }
    }
  ]
}
```

这类样本说明原始 XLAM / APIGen 本身就包含 multi-tool query，但 `compositional/xlam_datasets.py` 在“选 tool”阶段不会直接拿它们做频次统计，而是优先从单工具样本里挑可复用的 atomic building blocks。

### 5. 处理后给 compositional 训练/评测用的数据长什么样

生成后的 JSON 样本统一是下面这种格式：

| 字段 | 含义 |
| --- | --- |
| `user_input` | 供模型直接读取的输入文本 |
| `tools` | gold tool 名称序列；如果同一 tool 被调用两次，会重复出现两次 |
| `function_calls` | 与 `tools` 一一对应的参数 JSON 字符串 |

#### 5.1 训练集样本

来自 `compositional/data/training/function_calling_train_tools51-100_4calls.json`：

```json
{
  "user_input": "Provide me with the route information, including distance, duration, and steps, from the starting point (lat: 30.0444, lon: 31.2357) to the ending point (lat: 29.9792, lon: 31.1344) in kilometers.",
  "tools": [
    "directions_between_2_locations"
  ],
  "function_calls": [
    "{\"start_lat\": 30.0444, \"end_lon\": 31.1344, \"start_lon\": 31.2357, \"end_lat\": 29.9792, \"distance_unit\": \"km\"}"
  ]
}
```

#### 5.2 测试集样本

来自 `compositional/data/test/function_calling_test_tools51-100_4calls.json`：

```json
{
  "user_input": "\"I have a project where I need to determine the population growth for a small town that currently has 5000 residents. Could you tell me what the population would be in 5 years with a growth rate of 1.0%? Also, if the same town had a population of 3000 five years ago, what would have been its projected population today?\" Plus, get the whois information for google.com Lastly, calculate the greatest common divisor of the numbers 84 and 252.",
  "tools": [
    "project_population",
    "project_population",
    "whois",
    "greatest_common_divisor"
  ],
  "function_calls": [
    "{\"current_pop\": 5000, \"num_years\": 5, \"annual_growth\": 1.0}",
    "{\"current_pop\": 3000, \"num_years\": 5}",
    "{\"domain\": \"google.com\"}",
    "{\"a\": 84, \"b\": 252}"
  ]
}
```

可以看出，仓库最终训练/评测用的是“线性化后的 gold tool sequence + 参数序列”。

### 5.3 当前仓库里既有合成数据的实际统计

下面这些数字都基于当前仓库里已经生成好的数据文件，以及当前 `xlam_datasets.py` 的实现口径：

- `1-tool` 指 `len(set(sample["tools"])) == 1`
- `n-call` 指 `len(sample["function_calls"]) == n`
- 测试集是否包含 `1-tool` 样本由生成逻辑决定，当前实现中测试集跳过了 single-tool 样本，只生成 multi-tool 样本

#### 5.3.1 `51-100` 这组既有数据

对应文件：

- `compositional/data/training/function_calling_train_tools51-100_4calls.json`
- `compositional/data/test/function_calling_test_tools51-100_4calls.json`

`51-100` 训练集与测试集的 `1-tool` 数量如下：

| split | `1-tool` 样本数 | 总样本数 |
| --- | ---: | ---: |
| train | 2243 | 5000 |
| test | 0 | 500 |
| train + test | 2243 | 5500 |

`51-100` 测试集的 `unique tools` 分布如下：

| `unique tools` 数量 | 样本数 |
| --- | ---: |
| 2-tools | 250 |
| 3-tools | 250 |

`51-100` 测试集的 `function_calls` 数量分布如下：

| `function_calls` 数量 | 样本数 |
| --- | ---: |
| 2-call | 229 |
| 3-call | 233 |
| 4-call | 38 |

`51-100` 训练集中，单工具样本内部的 call 数分布如下：

| 单工具样本的 `function_calls` 数量 | 样本数 |
| --- | ---: |
| 1-call | 1904 |
| 2-call | 283 |
| 3-call | 50 |
| 4-call | 6 |

整个 `51-100` 训练集的 `function_calls` 分布如下：

| `function_calls` 数量 | 样本数 |
| --- | ---: |
| 1-call | 1904 |
| 2-call | 1434 |
| 3-call | 1350 |
| 4-call | 312 |

#### 5.3.2 `1-100` 这组既有数据

对应文件：

- `compositional/data/training/function_calling_train_tools1-100_4calls.json`
- `compositional/data/test/function_calling_test_tools1-100_4calls.json`

`1-100` 训练集与测试集的 `1-tool` 数量如下：

| split | `1-tool` 样本数 | 总样本数 |
| --- | ---: | ---: |
| train | 4483 | 5000 |
| test | 0 | 500 |
| train + test | 4483 | 5500 |

`1-100` 训练集的 `function_calls` 分布如下：

| `function_calls` 数量 | 样本数 |
| --- | ---: |
| 1-call | 3734 |
| 2-call | 809 |
| 3-call | 388 |
| 4-call | 69 |

`1-100` 训练集的 `unique tools` 分布如下：

| `unique tools` 数量 | 样本数 |
| --- | ---: |
| 1-tool | 4483 |
| 2-tool | 258 |
| 3-tool | 259 |

`1-100` 训练集中，单工具样本内部的 call 数分布如下：

| 单工具样本的 `function_calls` 数量 | 样本数 |
| --- | ---: |
| 1-call | 3734 |
| 2-call | 596 |
| 3-call | 134 |
| 4-call | 19 |

#### 5.3.3 为什么测试集没有 `1-tool` 样本

当前实现中，`synthesize_multi_tool_data()` 在 `split_name == "test"` 时跳过了 single-tool 样本的加入；训练集才会先加入全部 single-tool 样本，再补 multi-tool 样本。对应源码逻辑如下：

- `test` 分支直接跳过 single-tool 样本
- `training` 分支先加入 single-call single-tool 样本
- `training` 分支再加入同一个 tool 多次调用的 single-tool 样本
- 最后再根据 `train_multi_tool_ratios` / `test_multi_tool_ratios` 合成 multi-tool 样本

所以在当前既有数据里：

- train 中可以同时看到 `1-tool`、`2-tool`、`3-tool`
- test 中只会看到 `2-tool` 和 `3-tool`

更一般地说，当前实现遵循下面的 split 语义：

- training split 始终先保留 single-tool 样本，再按 ratio 配置补 multi-tool 样本
- test split 只生成 multi-tool 样本
- 当 ratio 串扩展到更多 bucket 时，train / test 会覆盖对应的 `2-tool..N-tool`

#### 5.3.4 原始 `1-100` 单工具池与 `max_samples_per_tool=80`

按当前本地原始 XLAM / APIGen 数据统计，`1-100` 这 100 个工具一共能抽到 `11695` 条 single-tool 样本，其中 `2036` 条是同一个 tool 的 multi-call 样本。

如果把 `max_samples_per_tool` 设为 `80`，覆盖情况如下：

- 有 `99` 个工具能拿到至少 `80` 条 single-tool 样本
- 有 `1` 个工具低于 `80`
- 这个工具是 `is_anagram_phrase`
- `is_anagram_phrase` 当前只有 `79` 条 single-tool 样本

所以对当前数据来说：

- `max_samples_per_tool=80` 基本可行
- 要求所有 `1-100` 工具完全统一上限时，`79` 是更稳的值

### 6. 论文和源码是如何从原始数据里抽出这 100 个 tools 的

#### 6.1 论文里的说法

论文附录 A.3 / A.4 给出的关键设定是：

1. benchmark 侧先从 APIGen 中采样 50 个 tools；
2. 每个 tool 保留 50 个 query-call pairs；
3. 同一 tool 的多次调用仍算“单工具”；
4. 再把不同 tools 的调用拼接成 multi-step queries；
5. 最终把合成数据控制在 `5000` 个训练样本、`500` 个测试样本；
6. 另外再准备一个额外的、与 benchmark tools 不重叠的 50-tool adaptation set，供 TokMem 先做 compositional adaptation。

也就是说：

- 论文的“主 benchmark”是 50 个 tools；
- 仓库里的“100 tools” = `50 个 adaptation tools + 50 个 benchmark tools`。

#### 6.2 源码里的实际落地步骤

`compositional/xlam_datasets.py` 的处理流程可以概括成下面几步：

1. 从 `datasets/xlam-function-calling-60k/data/*.parquet` 读入原始数据。
2. 逐条解析 `answers`，只看真实调用的 tool names。
3. 只保留“`answers` 里恰好只有 1 个唯一 tool”的样本来做 tool ranking。
   - 保留“同一个 tool 多次调用”的样本。
   - 丢弃“同一条样本里出现多个不同 tools”的样本。
4. 统计每个 tool 在这类单工具样本中的出现频次。
5. 显式排除 `search`。
6. 频次降序排序后：
   - `1-50` 取前 50 个 tools
   - `51-100` 取第 51 到第 100 个 tools
7. 对选中的每个 tool，最多保留 `50` 条原始 query-call pairs。
8. 按 `train_size / (train_size + test_size) = 5000 / 5500 ≈ 90.9%` 的比例，先对每个 tool 的原始样本做 train / test 拆分。
9. 在拆分后的单工具样本上继续合成 multi-tool queries：
   - 训练集：保留单工具样本，再补充 ratio 配置对应的 multi-tool 合成样本
   - 测试集：只保留 ratio 配置对应的 compositional multi-tool 样本
10. 每条最终样本最多允许 `train_max_function_calls` / `test_max_function_calls` 指定的 function calls；maintained launcher 当前使用 `4`。

#### 6.3 当前仓库里可以直接验证到的事实

从本地数据统计可得到：

| 统计项 | 数值 |
| --- | --- |
| 原始 train split 中的单工具样本总数 | `38,892` |
| 排名第 50 的 tool 频次 | `109` |
| 排名第 100 的 tool 频次 | `77` |
| `search` 在单工具样本中的频次 | `683` |
| `search` 是否会进入 top-100 | 不会，源码显式排除 |

这说明这 100 个 tools 都不是长尾极稀有工具，而是单工具样本里最常见的一批。

#### 6.4 当前生成结果的组成

以现成的 JSON 文件为例：

| 文件 | 样本数 | unique tool 数分布 | function call 数分布 |
| --- | --- | --- | --- |
| `train_tools1-50_4calls` | `5000` | `1:2240, 2:1380, 3:1380` | `1:1824, 2:1460, 3:1392, 4:324` |
| `test_tools1-50_4calls` | `500` | `2:250, 3:250` | `2:223, 3:243, 4:34` |
| `train_tools51-100_4calls` | `5000` | `1:2243, 2:1378, 3:1379` | `1:1904, 2:1434, 3:1350, 4:312` |
| `test_tools51-100_4calls` | `500` | `2:250, 3:250` | `2:229, 3:233, 4:38` |

这也说明当前 repo 默认落盘的 `test` 文件是一个更“纯”的 compositional evaluation：500 条测试样本全部都是 multi-tool。

### 7. 这 100 个 tools 有什么特别的地方

#### 7.1 它们不是随机的，而是高频且数据充足

- 这 100 个 tools 来自单工具样本频次 top-100。
- 第 100 名也有 `77` 条单工具样本，足够支撑“每个 tool 留 50 条原始 query-call pairs”的设定。

#### 7.2 它们是为“先学原子 procedure，再学组合”挑出来的

源码先看单工具样本，再拿这些 tool 去合成 multi-tool queries。说明它想要的是一批能稳定充当 atomic building blocks 的 procedures，而不是直接照抄原始 multi-tool rows。

这批 tool 的名字里能看到很明显的“原子化”特征，例如：

- 算法 / 数学类：`find_next_greater_element`、`binary_search`、`fibonacci`、`greatest_common_divisor`
- 统计 / 科学计算类：`chi_square_independence_test`、`calculate_standard_deviation`、`cosine_similarity`
- API / 业务动作类：`whois`、`whole_foods_order`、`place_safeway_order`、`get_order`

它们单独看都比较自洽，参数 schema 也比较清楚，所以很适合被拼成“先做 A，再做 B，最后做 C”的 compositional benchmark。

#### 7.3 两组 tools 是刻意拆开的

- `1-50` 不是主 benchmark，而是 adaptation / auxiliary set。
- `51-100` 才是当前 compositional README 里强调的 evaluation tools。

这种拆法的意义是：让 TokMem 先学会“如何在输出里交织 memory tokens 与 tool calls”，再迁移到一组从未见过的 benchmark tools 上。

#### 7.4 `search` 被排除，说明 benchmark 更偏 procedure memory 而不是泛化检索

源码明确把 `search` 从 ranked tools 中去掉了。

这里可以做一个合理推断：`search` 这类过于通用的工具会弱化 benchmark 对“具体 procedure 是否被记住、能否被组合”的考察，因此不适合作为这套 compositional setting 的核心 tool。这个动机在代码里是隐含的，不是论文里逐字写明的。

### 8. 一些代表性的 tool 名字

这里只列部分，完整名单见 `compositional/data/tool_descriptions_tools1-50.json` 和 `compositional/data/tool_descriptions_tools51-100.json`。

#### 8.1 排名前 1-10 的 tools

```text
calculate_investment_return
bacterial_growth
chi_square_independence_test
project_investment_growth
calculate_standard_deviation
min_meeting_rooms
linear_regression_prediction
cell_density
sort_numbers
euclidean_distance
```

#### 8.2 41-60 名附近的 tools

```text
average
batting_average
is_perfect_square
least_common_multiple
is_valid_email
predict_forest_area
get_order
find_kth_smallest_number
draw_cards
std_deviation
is_leap_year
final_velocity
find_equilibrium_index
cosine_similarity
get_pokemon_move_info
place_safeway_order
is_valid_palindrome
is_rotation
potential_energy
generate_random_string
```

#### 8.3 91-100 名的 tools

```text
calculate_factorial
find_first_non_repeating_char
find_longest_word
get_products_in_category
is_valid_sudoku
reverse_string
fibonacci
directions_between_2_locations
format_date
is_valid_parentheses
```

### 9. 与论文、README、脚本三者的关系

| 位置 | 关于 tools 的说法 | 如何理解 |
| --- | --- | --- |
| 论文附录 A.3 | 50 个 benchmark tools | 主评测集 |
| 论文附录 A.4 | 额外 50 个 adaptation tools | TokMem 的 held-out adaptation set |
| `compositional/README.md` | 1-50 adaptation，51-100 evaluation | 仓库层面的 100-tool 组织方式 |
| `scripts/compositional/llama_1b/tokmem_*.sh` | `51-100:1` | 当前维护的单轮 benchmark tools 训练 |
| `scripts/compositional/llama_1b/adaptation/run_compositional_tokmem_llama_1b.sh` | `1-50:1,51-100:3` | adaptation 旧实验入口，先 adaptation，再在 benchmark tools 上继续训练 |
| `scripts/compositional/llama_1b/adaptation/run_compositional_icl_llama_1b.sh` | 只加载 `51-100` | ICL baseline 旧实验入口，只在主 benchmark 上评测 |

#### 9.1 论文里 compositional adaptation 与正式训练各用了多少数据

论文里的数据量口径要分成两层看：

| 阶段 | tools 数量 | 样本量 | 说明 |
| --- | --- | --- | --- |
| adaptation phase | `50` 个 held-out auxiliary tools | `5,000` 个样本 | 用 LoRA 先让 backbone 学会 compositional memory token 的使用方式 |
| 正式 compositional benchmark | `50` 个 benchmark tools | `5,000` 个训练 queries + `500` 个测试 queries | 主文 3.3 里直接给出的最终训练/测试规模 |

同时，附录 A.3 还补充了正式 benchmark 的底层来源：

- 先对这 `50` 个正式 tools 中的每个 tool 收集 `50` 个 query-call pairs；
- 因此先得到 `50 × 50 = 2,500` 个原始单工具样本；
- 再基于这些样本合成 multi-tool queries，形成最终的 `5,000` 个训练 queries 和 `500` 个测试 queries。

换句话说：

- `adaptation` 阶段的数据量是 `50 tools + 5,000 samples`；
- `正式训练` 阶段的最终训练集规模是 `5,000`，但它的底层原始 atomic 样本池其实是 `2,500`。

### 10. 可以直接记住的最短版本

- 原始数据：XLAM / APIGen function-calling dataset。
- 原始主字段：`id / query / answers / tools`。
- tool ranking 依据：`answers` 中单工具真实调用频次，而不是 `tools` 候选列表。
- 100 tools 的来源：top-100 高频单工具 procedures，排除 `search`。
- 100 tools 的分工：前 50 做 adaptation，后 50 做主 benchmark。
- 论文里的 compositional 数据量：adaptation 是 `50 tools + 5,000 samples`，正式 benchmark 是 `50 tools + 5,000 train + 500 test`。
- `held-out` 的意思：这批辅助 tools 是单独留出来做 adaptation 的，不和正式 benchmark tools 重叠。
- 这些 tools 的特别之处：高频、参数结构清晰、单个 procedure 自洽、适合被拼成 multi-step compositional queries。

---

## 原文：2026-04-10-compositional-qa.md

## Compositional 问答记录

这份文档专门记录本轮关于 `compositional/` 数据构造、TokMem 训练格式与论文设定的问答，回答尽量简短。

### Q1. 现在 scripts 里 comp 训练是使用 `4calls` 进行训练吗？

是。

- 当前 maintained launcher 会生成并读取 `..._4calls.json`
- `4calls` 的意思是每条样本最多 `4` 个 function calls
- 数据生成器本身已经支持其他 `Ncalls` 文件名，例如 `..._10calls.json`

### Q2. 当前选取的 50 个 benchmark tools，是否满足每个 tool 都有 `100` 个训练样本和 `10` 个测试样本？

不满足。

- 如果看当前主 benchmark 使用的 `51-100` 这组 tools，它们都达不到 `110` 条单工具原始样本
- 如果看 `1-50` 这组，从原始样本量上可以满足 `100+10`
- 但当前脚本本身也没有按 `100/10` 这个规则生成数据

### Q3. 第二组工具抽取逻辑有什么问题，导致最终数据不符合 `100 train + 10 test`？

主要有四个问题：

1. 选 tool 时只按频次排名切片 `51-100`，没有先过滤 `>=110` 样本量的 tool。
2. `max_samples_per_tool` 被脚本固定成 `50`，先天不可能得到 `100+10`。
3. 之后是按全局比例切分，不是按每个 tool 固定切出 `100/10`。
4. 测试集默认只保留合成的 multi-tool compositional 样本，不是每个 tool 各拿 10 条独立 test。

### Q4. 论文里 compositional 的 adaptation 阶段和正式训练阶段分别用到多少数据量？

论文口径是：

- adaptation phase：`50` 个 held-out auxiliary tools，`5,000` 个样本
- 正式 compositional benchmark：`50` 个 benchmark tools，`5,000` 个训练 queries，`500` 个测试 queries

附录还补充：

- 正式 benchmark 先对 `50` 个 tools 每个收集 `50` 个 query-call pairs
- 所以底层先有 `2,500` 个原始单工具样本
- 再据此合成最终的 `5,000 / 500` compositional queries

### Q5. `held-out` 是什么意思？

这里的意思是：

- 这批工具被单独留出来
- 不和正式 benchmark 的那 `50` 个 tools 重叠
- 只用于 adaptation，不参与后面的主评测

它的重点是“与正式评测集隔离”，不是“模型永远没见过任何相关模式”。

### Q6. adaptation 和正式 compositional 训练阶段，训练样本数量有区别吗？

按论文口径，训练样本数量基本没有区别：

- adaptation：`5,000` 个训练样本
- 正式 compositional 训练：`5,000` 个训练 queries

主要区别在于：

- adaptation 用的是 held-out auxiliary tools
- 正式阶段用的是 benchmark tools

### Q7. 原始数据是什么样子的？作者是怎么构建 compositional 训练数据的？

原始 APIGen / XLAM 一行数据主要有三个字段：

- `query`：自然语言问题
- `answers`：gold tool calls，形如 `{"name": ..., "arguments": ...}`
- `tools`：候选工具定义，含 `name / description / parameters`

作者/代码构造 compositional 数据的大致流程是：

1. 先从原始数据里抽单工具 procedure 样本
2. 再把多个单工具样本拼成一个多步骤 query
3. 最后把它们转成 TokMem 训练用的 `tool token + 参数 JSON` 序列

### Q8. 原始数据里本来就有 `<memory token> + response1 + <memory token> + response2` 这种格式吗？

没有。

原始数据只有：

- 自然语言 `query`
- 结构化的 `answers`
- 候选 `tools`

`memory token` / `tool token` 是 TokMem 在模型侧额外引入的表示，不是原始数据自带的。

### Q9. APIGen 高频工具是怎么抽取的？

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

### Q10. `tools` 字段的作用是什么？

它不是“告诉模型可以调用哪些模型”，而是“告诉求解器当前有哪些候选工具，以及每个工具的 schema”。

在这个 repo 里主要有两个用途：

- 从原始数据中抽取工具描述和参数定义
- 给 ICL / RAG baseline 拼 `Available Tools` prompt

TokMem 训练本身并不是逐条样本把整份 `tools` 列表喂进模型。

### Q11. 原始工具调用没有给一个现实世界可能出现的问题吗？

不是。

原始数据是有自然语言 `query` 的，只是很多 query 带有 benchmark / research-oriented 风格，不一定像真实产品里的用户请求那么自然。

所以更准确的说法是：

- 有现实世界式的自然语言问题
- 但整体仍是研究型数据集，不完全等于真实使用场景

### Q12. TokMem 训练时，模型是怎么学会“输出 tool token”的？

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

### Q13. 组合 compositional query 用的单工具样本池，是不是按 Q9 的规则得到的？

是。

- 先按 Q9 的规则抽出“唯一 tool”样本
- 再拆成：
  - 单次调用的单工具样本
  - 同一 tool 多次调用的单工具样本

它们共同构成后续 compositional synthesis 的原料池。

### Q14. 合成后的 compositional query 里，多个 tool 可能完全不相干，这会不会有问题？

会有这个问题。

当前实现主要是随机选多个单工具 query，再用 `Also / Lastly` 之类连接词拼起来，所以很多组合只是“并列多个小任务”，不一定存在真实依赖关系。

### Q15. 作者有没有考虑多个 tool 之间组合的合理性？

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

### Q16. 这种做法合理吗？如果不够合理，可以怎么改进？

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

### Q17. 展示一条完整的合成样本。

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

### Q18. 如何判断模型选对了 tool？

做法是：

1. 从模型输出序列里找所有 tool token
2. 把 tool token 映射回 tool name
3. 再和 gold tool names 做比较

指标上主要是 `Tool Prediction F1`。

### Q19. 如何判断 tool 的调用参数是正确的？

做法是把预测调用和 gold 调用都解析成结构化表示，再做标准化比较。

指标上主要是 `Argument Generation F1`。

为了容忍表面写法不同，repo 会把调用归一化成 AST / 规范化 JSON 后再算分。

### Q20. 如何判断 tool 的执行结果是正确的？

当前这套 repo / 论文并不真正执行工具。

所以它能判断的是：

- tool 选没选对
- 参数写没写对

但它不能直接判断：

- API 真执行后返回了什么
- 返回值是否正确
- 后续步骤是否真的消费了前一步输出

如果要判断 execution correctness，就需要补真实工具执行或 mock executor。

### Q21. 单 tool 样本池里的单次调用样本和同一 tool 的多次调用样本，后续处理有什么不同？

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

### Q22. 之前说正式训练数据可能不满足要求，是不是没有考虑后续合成数据？

是，必须区分两个口径。

#### 口径 1：原始单 tool 原料是否满足每个 tool 有 `100 train + 10 test`

这个口径下，前面的结论成立：

- `1-50` 这组原始单工具样本量够
- `51-100` 这组原始单工具样本量不够
- 而且脚本还把 `max_samples_per_tool` 限成了 `50`

#### 口径 2：最终生成后的训练/测试集里，每个 tool 是否至少出现 `100/10` 次

这个口径下，当前最终数据是满足的，因为有后续合成数据把覆盖次数补上来了。

也就是说：

- 如果看原始单 tool 原料，正式 benchmark 那组不满足 `100+10`
- 如果看最终合成后的训练/测试集覆盖次数，当前数据里每个 tool 的出现次数已经超过 `100/10`

所以之后讨论“满足不满足要求”时，必须先说清楚问的是：

- `raw single-tool support`
- 还是 `final synthesized dataset coverage`

### Q23. 测试数据一共有多少条？都是合成的 compositional 数据吗？

- 每套 test 是 `500` 条
- `tools1-50` 一套 `500`
- `tools51-100` 一套 `500`
- 当前 repo 里两套 test 都是`纯合成的 multi-tool compositional 数据`
- 不会把原始单 tool 样本原样放进 test

### Q24. 当前筛出来的高频 tools 有什么问题？需要联网/外部 API 的 tool 怎么处理？

主要问题有：

- 高频定义很窄，只看 `answers` 里“恰好只有一个唯一 tool”的样本频次
- 只排除了名字叫 `search` 的 tool，不是把所有外部 API / 外部世界 tool 都排掉
- `51-100` 这组原始单 tool 频次不足以支撑每个 tool 原生 `100 train + 10 test`
- 当前脚本还把每个 tool 的原始样本先截到 `50`

对于 `whois`、`whole_foods_order`、`directions_between_2_locations` 这类 tool，这套 benchmark / repo 只要求：

- 选对 tool
- 填对 argument

它`不要求真正联网搜索或真实执行工具`。

### Q25. `whois`、`whole_foods_order`、`directions_between_2_locations` 的具体样本是什么样？

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

### Q26. 在一个 tool 的 arguments 生成完后，模型怎么知道该生成下一个 tool token？

不是靠显式规则判断，而是靠自回归 next-token 学习：

- 训练序列本来就是 `[tool_1][args_1][tool_2][args_2]...[tool_k][args_k][eot]`
- 所以在 `args_1` 结尾位置，gold 下一个 token 就是 `tool_2`
- 在最后一个 `args_k` 结尾位置，gold 下一个 token 就是 `<eot>`

模型学到的是这种序列模式，不是显式解析 JSON 完整性后再切换。

### Q27. 训练和测试时的输入输出 token 形式分别是什么？

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

### Q28. `args_1` 和 `tool_2` 中间有额外的结束标志吗？

没有。

当前格式就是直接：

```text
[tool_1][args_1][tool_2][args_2]...[tool_k][args_k]<eot>
```

所以边界是隐式的：

- 非最后一个 call：`args_i` 后面直接接下一个 `tool token`
- 最后一个 call：`args_k` 后面直接接 `<eot>`

### Q29. 论文和代码里分别怎么判断模型有效性？用了什么指标？

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

`Parse Error Rate` 在当前代码里定义为：

- `parse_errors / total_examples`
- 其中 `parse_errors` 累加的是 output calls 的解析失败次数
- 这个值表示每个样本平均会产生多少次 parse error
- 数值可以大于 `1`
- 这个统计口径与论文保持一致

当前 repo 里，`parse` 实际分两层：

### 1. 先把生成序列切成多条 `function_calls`

模型生成结束后，`compositional/model.py::_parse_generated_sequences` 会先把“用户输入之后新生成的 token”拿出来，再做这几步：

- 去掉 `eos` 之后的内容
- 扫描整段生成，找出所有 tool token 的位置
- 对每个 tool token，取它后面的参数 span
- 开启 `use_eoc` 时，优先截到下一个 `eoc`
- 没找到 `eoc` 时，退回到下一个 tool token；如果后面也没有 tool token，就截到序列尾部
- 把每一段参数 token decode 成字符串，放进 `function_calls`

所以这一层只是在做 span 切分。此时得到的是：

- `predicted_tools`
- `function_calls`

其中 `function_calls` 还是字符串列表，还没有做 JSON 解析。

### 2. 再把每条 `function_call` 字符串解析成结构化对象

评测阶段，`compositional/eval.py::parse_function_call` 会对每条 call 依次尝试：

1. 直接 `json.loads(text)`
2. 如果字符串里有 `{ ... }`，就截出最外层第一段对象子串再 `json.loads`
3. 如果像 `func_name(arg1=..., arg2=...)`，就按简单函数调用格式解析
4. 前面都失败时，返回：

```python
{"raw_text": text, "parse_error": True}
```

所以当前 `parse error` 的直接含义是：这条输出 call 没能成功转成结构化对象。

### 3. parse error 会怎样影响各个指标？

这部分最容易混，当前 repo 里至少有三套不同口径。

#### 3.1 `Tool F1`

`Tool F1` 只看 `predicted_tools` 和 `expected_tools`。

- 它不依赖 `function_calls` 是否 parse 成功
- 只要 tool token 已经被切出来，tool 名预测对了，这条样本就会给 `Tool F1` 加分

所以完全可能出现：

- 某条 call 的参数串已经 parse error
- 但 `Tool F1` 仍然很高

#### 3.2 表里的 `Arguments F1`

README 和汇总表里的 `Arguments F1`，当前实际对应的是评测结果中的 `avg_f1_score`，也就是 `Average F1 Score (Function Calls)`。

它的比较单位是“整条 call 的规范化结果”，不是参数 token 级别的部分匹配。

如果某条预测 call parse 失败：

- 这条 call 仍然会参与 `avg_f1_score`
- 参与时会以 `raw_text` 这一整条坏字符串进入比较
- 它通常和 gold call 对不上
- 效果上通常等价于：
  - 少一个 `TP`
  - 多一个 `FP`
  - 对应 gold 再多一个 `FN`

所以 parse 失败时，这条 `function call` 里的 arguments 会整体失分；当前实现里没有“虽然 JSON 坏了，但前半段几个参数还能拿部分分”的 token-level credit。

#### 3.3 `arguments_accuracy`

当前评测还会单独打印一个 `Arguments Accuracy`。

它和表里的 `Arguments F1` 不是同一个指标。它会先抽取 `(tool_name, normalized_args)` 对，再看有多少 gold pair 被命中。

这里如果某条 call parse error：

- 这条 call 不会产出有效的 `(tool, args)` pair
- 对应 gold arguments 匹配不上
- 效果上也是整条 arguments 失分

#### 3.4 `Parse Error Rate`

`Parse Error Rate` 单独累计每条 output call 的 parse 失败次数，再除以 `total_examples`。

所以：

- 它的统计粒度是 call-level
- 同一个样本里如果坏了两条 call，就会累加两次
- 因此这个值可以大于 `1`

### 4. 具体样本：一条 call 提前被 `eoc` 截断时，F1 怎么算？

测试集中有这类 4-call 样本：

```python
tools = [
  "project_population",
  "project_population",
  "whois",
  "greatest_common_divisor",
]

target_calls = [
  "{\"current_pop\": 5000, \"num_years\": 5, \"annual_growth\": 1.0}",
  "{\"current_pop\": 3000, \"num_years\": 5}",
  "{\"domain\": \"google.com\"}",
  "{\"a\": 84, \"b\": 252}",
]
```

假设模型输出里第一条被提前截断：

```python
predicted_calls = [
  "{\"current_pop\": 5000, \"num_years\": 5",
  "{\"current_pop\": 3000, \"num_years\": 5}",
  "{\"domain\": \"google.com\"}",
  "{\"a\": 84, \"b\": 252}",
]
```

这时：

- 第 1 条 parse 失败，变成 `{"raw_text": "...", "parse_error": True}`
- 后 3 条都能正常 parse

对表里的 `Arguments F1` 而言，可以把它理解成：

- 匹配成功的 call 有 `3` 条
- 预测总条数是 `4`
- gold 总条数也是 `4`

于是：

- `precision = 3 / 4 = 0.75`
- `recall = 3 / 4 = 0.75`
- `F1 = 0.75`

同时：

- `parse_errors["outputs"] = 1`

如果这条样本的 tool 顺序和 tool 名都对，那么它还可能同时表现为：

- `Tool F1` 很高
- 表里的 `Arguments F1` 下降
- `Parse Error Rate` 上升

这正是当前 `eoc` 相关实验里经常出现的组合。

### 5. 如果生成工具的顺序变化了，会有影响吗？

当前实现里，顺序影响比直觉中小。

#### 5.1 `Tool F1`

当前 `Tool F1` 用的是工具集合式比较，顺序不影响结果。

#### 5.2 表里的 `Arguments F1`

当前 `compare_function_calls_advanced(..., ignore_order=True)` 会按无序多集合思路比较，所以只要整条 call 本身一致，顺序变化通常不影响 `avg_f1_score`。

#### 5.3 `Exact Match Accuracy`

当前 repo 里的 `Exact Match Accuracy` 在 compositional 评测里也复用了 `ignore_order=True` 的比较口径，所以纯粹的调用顺序变化通常也不会单独把它打成错误。

### 6. 一个容易忽略的小细节

当前 `avg_f1_score` 里面的 `calculate_f1_score` 用的是 `set(outputs)` 和 `set(targets)`。

这意味着：

- 顺序不敏感
- 完全相同的重复 call 会被折叠

所以如果以后要更严格地区分：

- 调用顺序
- 重复调用次数

就需要单独改这层比较逻辑，而不是只看现在的 `Arguments F1`。

### Q30. adaptation 在 TokMem 的训练过程中有什么作用？

`adaptation` 的作用是一个 compositional warm-up。

- 它先用一组 `held-out` 的辅助 tools，让 backbone 学会 `[tool][args][tool][args]...` 这种生成结构
- 然后再进入正式 benchmark 的 memory acquisition

所以它更像：

- 先教模型“怎么交错生成 tool token 和参数”
- 再教模型“正式那批 tools 的具体记忆”

在当前 repo 的 TokMem launcher 里，它大致对应第一轮 `1-50:1`；后面 `51-100:3` 更接近正式训练阶段。
