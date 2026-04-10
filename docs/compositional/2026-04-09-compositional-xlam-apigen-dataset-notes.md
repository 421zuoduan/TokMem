# Compositional 使用的原始数据集与 100 Tools 抽取说明

## 1. 先说结论

- `compositional/` 当前使用的原始数据集是本地的 `datasets/xlam-function-calling-60k`，也就是 XLAM / APIGen 的 function-calling 数据。
- 论文正文真正作为 compositional benchmark 评测的是 50 个 tools；仓库为了 TokMem 的 adaptation phase，又额外准备了另外 50 个互不重叠的 tools，所以源码和脚本里会看到总共 `100 tools`。
- 这 `100 tools` 不是随机抽的，而是从原始数据的 `train` split 中，按“单工具样本里的真实调用频次”降序选出来的 top-100，并且显式排除了 `search`。
- 当前仓库默认把这 100 个 tools 划成两组：
  - `1-50`：adaptation / auxiliary tools
  - `51-100`：主 benchmark tools

## 2. 数据与脚本入口

### 2.1 原始数据入口

- 原始数据目录：`datasets/xlam-function-calling-60k/data/`
- 数据集 README：`datasets/xlam-function-calling-60k/README.md`
- 处理脚本：`compositional/xlam_datasets.py`

### 2.2 生成后的 compositional 数据

默认脚本会把处理结果写到 `compositional/data/`：

- `tool_descriptions_tools1-50.json`
- `tool_descriptions_tools51-100.json`
- `training/function_calling_train_tools1-50_4calls.json`
- `training/function_calling_train_tools51-100_4calls.json`
- `test/function_calling_test_tools1-50_4calls.json`
- `test/function_calling_test_tools51-100_4calls.json`

### 2.3 维护脚本里的固定参数

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

## 3. 原始 XLAM / APIGen 数据长什么样

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

### 3.1 `answers` 里已经标注了什么

`answers` 是“真实调用记录”，每个元素至少有：

| 子字段 | 含义 |
| --- | --- |
| `name` | 实际调用的 tool 名称 |
| `arguments` | 该次调用的参数字典 |

### 3.2 `tools` 里已经标注了什么

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

### 3.3 一个容易误解但很关键的点

源码在统计 tool 频次时看的是 `answers`，不是 `tools`。

原因是原始样本的 `tools` 往往是“候选工具列表”，里面可能包含没有真正被调用的其他工具；只有 `answers` 才表示该条样本里真正执行了哪些 tool calls。这个细节直接决定了“100 tools 是怎么抽出来的”。

## 4. 原始样本展示

下面都来自本地原始 parquet。

### 4.1 单工具、单次调用

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

### 4.2 单工具、同一 tool 多次调用

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

### 4.3 一条原始样本里含多个不同 tools

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

## 5. 处理后给 compositional 训练/评测用的数据长什么样

生成后的 JSON 样本统一是下面这种格式：

| 字段 | 含义 |
| --- | --- |
| `user_input` | 供模型直接读取的输入文本 |
| `tools` | gold tool 名称序列；如果同一 tool 被调用两次，会重复出现两次 |
| `function_calls` | 与 `tools` 一一对应的参数 JSON 字符串 |

### 5.1 训练集样本

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

### 5.2 测试集样本

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

## 6. 论文和源码是如何从原始数据里抽出这 100 个 tools 的

### 6.1 论文里的说法

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

### 6.2 源码里的实际落地步骤

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
   - 训练集：保留单工具样本，再补充 2-tool / 3-tool 合成样本
   - 测试集：在当前仓库默认实现下，只保留 2-tool / 3-tool compositional 样本
10. 每条最终样本最多允许 `4` 个 function calls。

### 6.3 当前仓库里可以直接验证到的事实

从本地数据统计可得到：

| 统计项 | 数值 |
| --- | --- |
| 原始 train split 中的单工具样本总数 | `38,892` |
| 排名第 50 的 tool 频次 | `109` |
| 排名第 100 的 tool 频次 | `77` |
| `search` 在单工具样本中的频次 | `683` |
| `search` 是否会进入 top-100 | 不会，源码显式排除 |

这说明这 100 个 tools 都不是长尾极稀有工具，而是单工具样本里最常见的一批。

### 6.4 当前生成结果的组成

以现成的 JSON 文件为例：

| 文件 | 样本数 | unique tool 数分布 | function call 数分布 |
| --- | --- | --- | --- |
| `train_tools1-50_4calls` | `5000` | `1:2240, 2:1380, 3:1380` | `1:1824, 2:1460, 3:1392, 4:324` |
| `test_tools1-50_4calls` | `500` | `2:250, 3:250` | `2:223, 3:243, 4:34` |
| `train_tools51-100_4calls` | `5000` | `1:2243, 2:1378, 3:1379` | `1:1904, 2:1434, 3:1350, 4:312` |
| `test_tools51-100_4calls` | `500` | `2:250, 3:250` | `2:229, 3:233, 4:38` |

这也说明当前 repo 默认落盘的 `test` 文件是一个更“纯”的 compositional evaluation：500 条测试样本全部都是 multi-tool。

## 7. 这 100 个 tools 有什么特别的地方

### 7.1 它们不是随机的，而是高频且数据充足

- 这 100 个 tools 来自单工具样本频次 top-100。
- 第 100 名也有 `77` 条单工具样本，足够支撑“每个 tool 留 50 条原始 query-call pairs”的设定。

### 7.2 它们是为“先学原子 procedure，再学组合”挑出来的

源码先看单工具样本，再拿这些 tool 去合成 multi-tool queries。说明它想要的是一批能稳定充当 atomic building blocks 的 procedures，而不是直接照抄原始 multi-tool rows。

这批 tool 的名字里能看到很明显的“原子化”特征，例如：

- 算法 / 数学类：`find_next_greater_element`、`binary_search`、`fibonacci`、`greatest_common_divisor`
- 统计 / 科学计算类：`chi_square_independence_test`、`calculate_standard_deviation`、`cosine_similarity`
- API / 业务动作类：`whois`、`whole_foods_order`、`place_safeway_order`、`get_order`

它们单独看都比较自洽，参数 schema 也比较清楚，所以很适合被拼成“先做 A，再做 B，最后做 C”的 compositional benchmark。

### 7.3 两组 tools 是刻意拆开的

- `1-50` 不是主 benchmark，而是 adaptation / auxiliary set。
- `51-100` 才是当前 compositional README 里强调的 evaluation tools。

这种拆法的意义是：让 TokMem 先学会“如何在输出里交织 memory tokens 与 tool calls”，再迁移到一组从未见过的 benchmark tools 上。

### 7.4 `search` 被排除，说明 benchmark 更偏 procedure memory 而不是泛化检索

源码明确把 `search` 从 ranked tools 中去掉了。

这里可以做一个合理推断：`search` 这类过于通用的工具会弱化 benchmark 对“具体 procedure 是否被记住、能否被组合”的考察，因此不适合作为这套 compositional setting 的核心 tool。这个动机在代码里是隐含的，不是论文里逐字写明的。

## 8. 一些代表性的 tool 名字

这里只列部分，完整名单见 `compositional/data/tool_descriptions_tools1-50.json` 和 `compositional/data/tool_descriptions_tools51-100.json`。

### 8.1 排名前 1-10 的 tools

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

### 8.2 41-60 名附近的 tools

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

### 8.3 91-100 名的 tools

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

## 9. 与论文、README、脚本三者的关系

| 位置 | 关于 tools 的说法 | 如何理解 |
| --- | --- | --- |
| 论文附录 A.3 | 50 个 benchmark tools | 主评测集 |
| 论文附录 A.4 | 额外 50 个 adaptation tools | TokMem 的 held-out adaptation set |
| `compositional/README.md` | 1-50 adaptation，51-100 evaluation | 仓库层面的 100-tool 组织方式 |
| `run_compositional_tokmem_llama_1b.sh` | `1-50:1,51-100:3` | 先 adaptation，再在 benchmark tools 上继续训练 |
| `run_compositional_icl_llama_1b.sh` | 只加载 `51-100` | ICL baseline 只在主 benchmark 上评测 |

### 9.1 论文里 compositional adaptation 与正式训练各用了多少数据

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

## 10. 可以直接记住的最短版本

- 原始数据：XLAM / APIGen function-calling dataset。
- 原始主字段：`id / query / answers / tools`。
- tool ranking 依据：`answers` 中单工具真实调用频次，而不是 `tools` 候选列表。
- 100 tools 的来源：top-100 高频单工具 procedures，排除 `search`。
- 100 tools 的分工：前 50 做 adaptation，后 50 做主 benchmark。
- 论文里的 compositional 数据量：adaptation 是 `50 tools + 5,000 samples`，正式 benchmark 是 `50 tools + 5,000 train + 500 test`。
- `held-out` 的意思：这批辅助 tools 是单独留出来做 adaptation 的，不和正式 benchmark tools 重叠。
- 这些 tools 的特别之处：高频、参数结构清晰、单个 procedure 自洽、适合被拼成 multi-step compositional queries。
