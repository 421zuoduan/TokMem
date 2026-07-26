# TapMem / TokMem 的 InterCode-Bash 实验

这个目录包含一套独立的训练与测试代码，用来回答：

> procedure 不是人工给定的原子工具，而且一条 Bash 程序有多种合理切法时，
> TapMem 是否仍然有效？

训练数据使用 NL2Bash，最终测试使用 InterCode-Bash 正式发布的 200 题。测试时
TokMem/TapMem 只看到自然语言任务和前几轮真实执行结果，不会看到参考命令或正确
procedure 切分；Base+ToolDesc 还看到一份对所有测试题都相同的 TRAIN-only 文字目录。

本目录只读取 `../datasets/nl2bash` 和 `../datasets/intercode`。所有生成的数据、
checkpoint、日志和 Docker 构建记录都必须写在本目录下；I/O 代码会拒绝把实验输出
写到其他目录。NL2Bash 的旧解析器会在导入目录生成缓存，因此代码会先把所需模块复制
到本目录的私有 staging，再从 staging 导入，不会改动下载的数据集。

## 1. 方法没有改成另一个模型

训练方法只比较：

| 方法 | procedure memory | EOC | TCRA / logit bias | memory bank 概率门 |
|---|---:|---:|---:|---:|
| `tokmem` | 是 | 否 | 否 | 否 |
| `tapmem` | 是 | 是 | 是 | 是 |

两者都冻结基础语言模型并训练 procedure embedding；TapMem 另外训练 EOC embedding
和原论文中的线性 TCRA head。procedure 不再被假定为铺满整段回答。训练目标保留
原始 Bash 文本中 procedure 以外的前缀、连接符、空白和注释。例如 TokMem 可以是：

```text
普通前缀 <memory_1> Bash片段1 | <memory_2> Bash片段2 普通后缀
```

TapMem 只在每个 procedure 的实际结尾加 EOC：

```text
普通前缀 <memory_1> Bash片段1 <EOC> | <memory_2> Bash片段2 <EOC> 普通后缀
```

训练时，每个真实 memory token 都提供一次 TCRA 路由监督；它前面可以是回复开头、
普通文本或 EOC，不再要求 memory 紧挨着 EOC。推理时 TapMem 维护一个简单状态：
选中 memory 后进入 procedure，生成 EOC 后退出；普通文本不会被误当成 procedure。

只有 TapMem 使用 memory bank 概率门。在 procedure 外，先从**没有加 TCRA 的全词表
logits**计算全部 memory token 的 softmax 概率总和。总和达到默认阈值 `0.5` 时，
应用 TCRA 后只在 memory bank 内选择；未达到时仍在全词表中选择。低于阈值但原始
top-1 已经是 memory 时，TapMem 仍按原 TCRA 做软重排，但不屏蔽普通 token。在
procedure 内不计算这道门，也不应用 TCRA。TokMem 始终使用原始全词表 greedy
解码，不使用 EOC、TCRA 或这道概率门。

此外评测一个不训练的 `Base+ToolDesc` 基线：同一个原始基础模型在 system prompt
中获得一份固定的文字工具目录，不使用 memory token。目录只由去泄漏后的 TRAIN
生成，包含全部 procedure 结构和每个原子工具的一条用法示例；所有测试题看到完全
相同的目录，不按 query 检索或截断，因此它不是 RAG。

## 2. procedure 边界为什么是模糊的

`bashlex` 只把一条简单命令或平坦管道拆成不会破坏 Bash 语法的基本单元。例如：

```bash
find /testbed -type f -print0 | sort -z | xargs -r0 md5sum | md5sum
```

得到：

```text
find → sort → xargs → md5sum
```

它可以切成 `[find+sort] [xargs+md5sum]`，也可以切成
`[find] [sort+xargs] [md5sum]`。`bashlex` 不判断哪一种更像人的步骤。

代码从训练语料统计跨命令模板重复出现的任意长度相邻组合，再用 unigram 概率模型对
一条命令的所有完整切法一起计算概率。用于比较边界的普通样本固定 procedure 数量，
只从相同数量的合法切法中抽边界。因此这些样本的 MAP 与 sampled 视图中，普通 Bash
token 和 procedure 调用数完全相同，真正变化的只有边界放在哪里。

这里的 MAP 不是另一种模型，而是固定边界对照：对每条命令永远采用概率最高的完整
切分。sampled 则保持 MAP 的 procedure 数量不变，按后验概率抽取另一种合法切分。
比较两者可以判断 TapMem 的收益是否真的与模糊边界有关。

另外有一小组 TRAIN-only coverage anchor，专门防止“词典里有 token，训练目标里却
一次也没出现”。它们在 MAP 和 sampled 设置中完全相同，不参与两种边界设置的差异；
只有某个 procedure 在任何固定数量切法中都不可能出现时，anchor 才使用同一训练命令
上的合法变数量切法。TokMem 和 TapMem 的模型结构没有因此改变。

## 3. procedure 超过 reserved token 数量

247 不再是 procedure 数量上限，只是“不需要扩词表”的分界点：

1. 先使用 tokenizer 原生的 reserved token，并固定留一枚给 EOC；
2. procedure 更多时，用 `tokenizer.add_special_tokens` 加入新的控制 token；
3. 扩展基础模型输入、输出词表，使新增 token ID 合法；
4. 真正训练的仍是 wrapper 中单独保存的 procedure memory embedding；
5. 原生和新增 procedure 的全部 embedding 一次性做两两正交初始化。

严格两两正交要求 `K <= hidden_size`。如果 procedure 数超过隐藏维度，代码会直接
报错，不会静默截断或声称仍然正交。EOC 不计入 K，也不加入 procedure 的 Gram 矩阵。

## 4. 环境

```bash
source /home/shilong/anaconda3/etc/profile.d/conda.sh
conda activate tokmem
```

训练和纯 CPU 测试所需的主要包已经在 `tokmem` 环境中。真实 InterCode 评测还需要
Docker Python SDK、Gymnasium 和可用的 Docker daemon：

```bash
pip install -r compositional_intercode_bash/requirements-intercode.txt
```

所有命令都从仓库根目录 `/data/shilong/tokmem` 执行。

### 4.1 没有管理员权限时的本地 Docker

机器没有系统 Docker 权限、也不能安装 `newuidmap/newgidmap` 时，可以使用本目录已经
下载的 Docker 26.1.3 静态程序和“单 UID”调试环境。整个过程使用当前普通用户，不调用
`sudo`，不启动系统服务，也不开放 TCP 端口。先检查前提：

```bash
compositional_intercode_bash/single_uid_rootless_runtime.sh doctor
```

在一个单独终端中以前台方式启动：

```bash
compositional_intercode_bash/single_uid_rootless_runtime.sh foreground
```

另一个终端设置客户端地址并检查四个镜像：

```bash
eval "$(compositional_intercode_bash/single_uid_rootless_runtime.sh env)"
compositional_intercode_bash/single_uid_rootless_runtime.sh status
compositional_intercode_bash/single_uid_rootless_runtime.sh smoke
```

普通交互式终端也可以用 `start` 放到后台，并用 `stop` 停止。受控任务执行器可能在命令
返回时统一清理后台子进程，此时应使用上面的 `foreground`。运行目录只允许当前用户
访问；容器不能连接宿主机的回环服务。`status` 必须显示 `rootless`、`vfs` 和
`default_runtime=singleuid`。

这个方案只把宿主当前 UID/GID 映射为容器里的 root，不提供额外宿主权限。因为它无法
表示容器里的 `_apt`、`adm` 等其他所有者，所以实现中做了三项明确的兼容处理：

- 导入基础镜像时，把层内文件所有者压成容器 root；
- 安装软件包时，不执行无法表示的 `chown/chgrp`，官方四个文件系统 setup 脚本本身
  仍保持不变；
- 启动容器时只把 `devpts` 的 `gid=5` 改为可表示的 `gid=0`。

因此它适合验证“镜像能否启动、模型能否执行命令、官方 reward 能否返回”的工程链路，
不能替代论文最终结果所需的标准 Docker 复核。当前宿主也没有可委派的 cgroup v2，
不能可靠限制模型命令的 CPU、内存和进程数；评测必须串行运行。`vfs` 会占用较多磁盘，
`doctor` 会打印剩余空间。正式 200 题结果应在支持标准 rootless Docker
（具有真实 subordinate-ID 映射和 cgroup v2）或隔离的 rootful Docker 机器上复跑。

## 5. 构造训练数据

### 5.1 下载并核验论文发布的 NL2Bash 划分

本地已下载时，这一步会核验哈希后直接跳过：

```bash
python -m compositional_intercode_bash.download_data
```

如果 Hugging Face 主站连接失败：

```bash
python -m compositional_intercode_bash.download_data \
  --endpoint https://hf-mirror.com
```

代码固定使用数据仓库 revision
`f92a7f891996d1d1230f25294a2a1c1c8b0cb1a4`，并分别核验三份 Parquet 的
SHA-256。然后执行：

```bash
python -m compositional_intercode_bash.prepare_data sources \
  --artifact-dir compositional_intercode_bash/artifacts/default
```

它硬检查：

```text
raw = 12,607
filtered = 9,305
train/dev/test = 8,090 / 609 / 606
InterCode fs1/fs2/fs3/fs4 = 60 / 53 / 60 / 27
```

`source_manifest.json` 同时绑定两份物化 JSONL、三份发布 Parquet、四份 InterCode
任务文件和原始 `all.nl/all.cm` 的哈希。任务必须是严格连续且不重复的官方 200 个
ID；空文件、截断文件或改过的 manifest 都不能进入来源匹配。

当前仓库中的旧 `filter_data.py` 在现代依赖下会得到 11,546 条，而不是论文发布的
9,305 条，因此不能作为默认数据。若要审计这个版本差异，可以显式运行
`--split-source legacy-replay`；数量不符时程序会停止，绝不会把 11,546 条伪装成
论文划分。

### 5.2 建来源组并去除 InterCode 泄漏

正式构造直接运行：

```bash
python -m compositional_intercode_bash.prepare_data provenance \
  --artifact-dir compositional_intercode_bash/artifacts/default \
  --primary
```

来源候选使用下面几条明确规则，不做“看起来相似”的无限扩张：

- 两种命令模板表示必须整体相同；出现超过 64 次的通用模板不用于连边；
- 结构序列相同或只差一步时，双方都必须至少有两个命令；
- 结构候选的指令词集合 Jaccard 不低于 0.25；
- 每道题最多保留 512 条候选，超过就停止并要求检查规则。

固定数据上，149 题能自动唯一确认来源，另外 51 题存在多个合理候选。正式实验不再
要求人工裁决，也不会把这 51 题当成“无泄漏”：对官方 200 题检索到的全部候选
source group 一律从 NL2Bash TRAIN/DEV 中删除。删除发生在重新划分训练集和开发集
之前，因此候选来源也不会进入用于选择 K 的开发集。

`provenance.jsonl` 保存每道题的全部候选、命中理由、文字重合度以及实际删除的
source group。当前固定数据共删除 1,153 个来源组，涉及发布 TRAIN/DEV 的
1,317/8,699 行，即 15.14%；最终保留 5,816 条派生训练样本和 619 条派生开发样本。
`--primary` 硬检查官方 200 题完整、候选并集与实际删除集合完全相同。若删除比例超过
50% 或剩余比例低于 50%，程序会停止。`--exploratory` 使用相同的保守删除规则，但
产物不能冒充正式实验。

仍可通过 `--adjudications` 保存人工分析结果，但它只作为审计信息，不会把任何候选
样本放回训练集。

### 5.3 学 procedure 词典并选择 K

Llama-3.2-1B 使用：

```bash
python -m compositional_intercode_bash.prepare_data procedures \
  --artifact-dir compositional_intercode_bash/artifacts/default \
  --hidden-size 2048 \
  --primary
```

流程是：

1. 只接受一个简单命令或由简单命令组成的平坦管道；
2. 保存原字符范围和基本文本块，任何切分删除控制 token 后都逐字还原原命令；
3. singleton 全部保留；多单元组合要求至少跨两个模板组出现，并至少一次只是更长命令
   的一部分；不会额外塞入训练集中从未出现的占位 procedure；
4. 用 log-space forward/backward 和 EM 学每个组合的概率；完全相同的基本单元序列
   先合并并累加权重，数学目标不变，但删减评分更快；
5. 用三种固定初始值分别删减候选；每轮按“删除后训练似然下降”排序，批量删除当前
   可删 procedure 的 20%，并强制保留 `K=247` 和最小词典两个落点，不暴力遍历每个 K；
6. 对每个候选 K 先用训练集计算 A/R 组数量；正式实验至少要求 20 个模糊模板组、
   长管道模板组的 5% 和 100 个普通模板组，不能稳定形成
   `2 个模糊组 + 2 个普通组` 的 K 不进入选择；
7. 在 NL2Bash 开发集上比较剩余容量，用成对 one-standard-error 规则选择最小的
   足够好 K。

K 可以大于 247，只要不超过 `--hidden-size`。最终词典写到
`artifacts/default/procedure_lexicon.json`。词典哈希覆盖 procedure 顺序、概率、
停止概率和必保留标记；只要其中一项变化，旧切分就不能继续使用。

`--primary` 会核对上一步确实删除了官方 200 题的全部候选来源组，并核对来源文件
哈希。当前正式数据有 330 个候选 procedure；训练边界门槛允许 `K=292` 和 `K=330`
进入开发集比较，one-standard-error 规则最终选择 `K=292`。其中前 247 个使用原生
reserved token，另外 45 个使用新增 special token。

`procedure_report.json` 不只绑定最终词典，还绑定 atoms、候选表、容量选择、
三条删减路径、所有 K 的网格目录和训练边界门禁。因此不能事后改 K 曲线或替换某个
中间模型而继续训练。

### 5.4 生成固定边界或模糊边界训练表

```bash
python -m compositional_intercode_bash.prepare_data views \
  --artifact-dir compositional_intercode_bash/artifacts/default \
  --setting sampled \
  --data-seed 1729 \
  --epochs 100 \
  --minimum-procedure-exposures 10 \
  --primary
```

第二套预注册边界表只需把 seed 改为 `7919`。固定最高概率切分使用
`--setting map`。`--epochs 100` 是最少轮数，不是强行截断：代码先只用派生 TRAIN
建立“合法切分会产生哪些 procedure 正例”的表，再用确定性集合覆盖挑展示命令。
若 100 轮不足以让每个 procedure 至少出现 10 次，会自动增加 epoch。每个有效 batch
仍是 2 个边界不确定来源组和 2 个普通来源组，同一模板组在一个 epoch 内最多出现
一次。

coverage anchor 只取自 TRAIN，不看开发集和 InterCode；MAP、两个 sampled seed 以及
TokMem/TapMem 共用相同 anchor。最终 `.coverage.json` 分别报告原生 procedure 和新增
procedure 的正例次数，任何一项少于 10 次都会停止。sampled 主设置还会在真正随机抽
边界的 `boundary_sample` 子集上检查换边界比例不低于 10%，coverage anchor 不能替
它把这个比例抬高。

当前 `K=292` 的三张正式训练表都是 100 个调度 epoch、5,200 条展示和 1,300 次参数
更新；sampled-1729 和 sampled-7919 在真正边界样本上的换边界率分别为 37.26% 和
36.37%。增加训练量时同时提高最少 procedure 正例次数，避免只重复高频工具。

每份 view 都有配套的 `.report.json` 和 `.coverage.json`。完整哈希链从原始来源、
候选来源删除、来源组、派生划分，一直覆盖到 procedure 选择证据、展示表和最终 view。
后续文件被改过、混入另一份词典，或者把探索产物冒充正式产物时，训练会在加载模型
前停止。

## 6. 训练

论文 compositional 主表使用
`Llama-3.2-1B-Instruct`、`Llama-3.2-3B-Instruct` 和
`Llama-3.1-8B-Instruct`。本实验沿用同一组基础模型；下面是 1B、sampled-1729
的一次运行示例：

```bash
VIEW=compositional_intercode_bash/artifacts/default/views.sampled.1729.jsonl
LEXICON=compositional_intercode_bash/artifacts/default/procedure_lexicon.json
MODEL=models/Llama-3.2-1B-Instruct

python -m compositional_intercode_bash.train \
  --method tokmem \
  --model-name "$MODEL" \
  --procedure-lexicon "$LEXICON" \
  --views "$VIEW" \
  --output-dir compositional_intercode_bash/runs/tokmem_d1729_m42 \
  --model-seed 42 \
  --batch-size 4 \
  --gradient-accumulation-steps 1

python -m compositional_intercode_bash.train \
  --method tapmem \
  --model-name "$MODEL" \
  --procedure-lexicon "$LEXICON" \
  --views "$VIEW" \
  --output-dir compositional_intercode_bash/runs/tapmem_d1729_m42 \
  --model-seed 42 \
  --expected-epochs 100 \
  --training-epochs 151 \
  --batch-size 4 \
  --gradient-accumulation-steps 1
```

现有训练表保持为 5200 条、100 个 source epoch，不会重新生成或改写。正式训练先按
原顺序完整跑完 source epoch 0–99，再按同样顺序复用 source epoch 0–50，因此
`--training-epochs 151` 一共执行 `151 × 13 = 1963` 次参数更新。`--expected-epochs
100` 只核对源文件，`--training-epochs` 才控制实际训练轮数；scheduler 也按 1963
次更新计算。checkpoint、summary 和逐步日志会同时记录 source epoch、实际训练 epoch
以及复用轮次。

训练前仍会统计 TRAIN 中有多少 template group 出现在这份 100 轮训练表里，并把未出现
的 group 写进完整性诊断。这个统计只用于说明数据覆盖情况，不再因为训练表没有枚举完
所有可用 group 而拒绝正式训练；procedure token 的正例覆盖检查仍然保留。
默认配置是学习率 `5e-3`、AdamW、10% warmup 后线性降到 0、最长 1024 token。
超长样本、非有限 loss 或非有限梯度都会使本次 run 失败，不会截断、跳步或回退到更早
checkpoint。

正式实验若使用 Hugging Face 模型名，`--model-revision` 必须是完整的 40 位 commit
SHA，`main` 或标签名会被拒绝。若 `MODEL` 是本地目录，可以省略 revision；程序会
计算整个目录的 SHA-256，并在训练结束和每次加载 checkpoint 时重新核对。程序还会
核对 view 内的 data seed、边界设置、procedure 数量和词典哈希，防止同 K 的旧 view
与新词典混用。

每次参数更新的有效 batch 强制为 4，才能保持固定的 `2 个边界模糊组 + 2 个普通组`。
8B 使用：

```text
--batch-size 1 --gradient-accumulation-steps 4
```

`4×1`、`2×2` 和 `1×4` 都按同一组 4 条样本中的全部监督 token 数归一化；TapMem
的路由损失另按该组的全部路由位置数归一化。因此改变 microbatch 只改变显存占用，
不会把“每条样本等权”和“每个 token 等权”混成两个训练目标。冻结的基础模型始终
保持 eval 模式，避免 dropout 让不同切法产生额外差异。

checkpoint 只保存可训练的 procedure/EOC embedding 和 TCRA head，同时保存扩展后的
tokenizer、procedure token 的准确名称和 ID、基础模型身份、完整上游产物哈希、
正例覆盖报告、主实验状态及训练配置。若只是调试非正式数据，必须显式增加
`--allow-exploratory-views`；这类 checkpoint 会标记为非正式，不能用于论文主结果。

### 6.1 只用 TRAIN 选择 TapMem 路由头学习率

路由头学习率预实验不使用 TEST，也不直接拿“训练前 20 个 epoch、后 5 个 epoch”
比较，因为相同命令模板会跨 epoch 重复。先按 `template_group_id` 固定留出 10% 的
A 组和 R 组，再从完全不含留出组的完整 `2A+2R` batch 中取出与 20 个源 schedule
epoch 等量的更新。选 batch 时先用确定性的贪心覆盖全部 K 个 procedure 类别，再按
源 schedule 顺序补足更新数；若某组是某个 procedure 唯一的训练来源，它不会被留出。
这样避免把“某个路由类别从未训练过”误判成学习率问题：

```bash
SPLIT=compositional_intercode_bash/artifacts/default/route_lr_split.json

python -m compositional_intercode_bash.route_probe_split \
  --views "$VIEW" \
  --output "$SPLIT" \
  --fit-source-epochs 20

python -m compositional_intercode_bash.train \
  --method tapmem \
  --model-name "$MODEL" \
  --procedure-lexicon "$LEXICON" \
  --views "$VIEW" \
  --route-probe-split "$SPLIT" \
  --routing-learning-rate 1e-3 \
  --output-dir compositional_intercode_bash/runs/route_lr_1e-3 \
  --batch-size 1 \
  --gradient-accumulation-steps 4

python -m compositional_intercode_bash.probe_checkpoint \
  --checkpoint compositional_intercode_bash/runs/route_lr_1e-3 \
  --views "$VIEW" \
  --split-manifest "$SPLIT" \
  --output compositional_intercode_bash/runs/route_lr_1e-3/probe.json
```

manifest 绑定完整 views 和 report 的哈希，fit/probe 的模板组交集必须为 0；probe
对每个留出组固定选一条，并分别报告 A/R。指标包括 AR NLL、路由 NLL，以及在 gold
状态为 procedure 外时：真实 memory 位置触发 `0.5` 概率门的召回率、普通 token
位置误触发率、触发后的 TCRA memory-bank top-1 正确率。这里普通 token 包括
procedure 之间的连接文本和回答结束 token。概率门阈值固定为 `0.5`，不参与调参。
用相同初始化分别跑 `5e-3`、`1e-3`、`5e-4` 三个路由头学习率，主要按留出组的路由
NLL 选择；接近时再比较 AR NLL 和 A/R 分项，不能换 split 追求更好的数值。

partial run 的 scheduler 仍按完整正式 schedule 的总更新数计算 warmup 和衰减，
checkpoint 会记录 split 哈希和两侧模板组，并强制 `formal_ready=false`。选好路由头
学习率后，正式 TapMem 必须重新初始化并去掉 `--route-probe-split`，在全部 TRAIN
views 上完整训练；不能接着使用 partial 权重。

### 6.2 Base+ToolDesc 不训练基线

这个基线不走 `train.py`，也没有 model seed 或 epoch。先创建只包含模型身份、
原始 tokenizer 和固定工具目录的评测产物：

```bash
python -m compositional_intercode_bash.base_text \
  --source-artifact-dir compositional_intercode_bash/artifacts/default \
  --output-dir compositional_intercode_bash/artifacts/base_tool_desc_llama1b \
  --model-name models/Llama-3.2-1B-Instruct
```

目录列出全部 292 个 procedure 结构，并为 138 个必保留原子工具提供一条确定性的
TRAIN-only 用法示例。它不使用完整训练指令、开发集、InterCode query/gold 或文件
系统内容，也不调用大模型生成说明。1B tokenizer 下目录为 2,813 token；代码设有
3,072-token 硬上限，超过就停止，不会针对测试 query 检索或裁剪。产物没有
`trainable.safetensors`，`trainable_parameter_count=0`。

准备好的比较矩阵是：

- Base+ToolDesc：每个 backbone 评测一次，不训练；
- TokMem、TapMem：分别使用 MAP、sampled-1729、sampled-7919；
- memory 方法沿用论文的 model seed 40、41、42。

因此每个 backbone 有 18 个 memory 训练 run 和 1 个无训练基线。sampled 是回应
“模糊边界”质疑的主结果，MAP 用作固定边界对照；目前只生成了正式数据和三个
backbone 的 Base+ToolDesc 产物，没有启动任何训练。

## 7. 构建四个 InterCode 文件系统

官方 Dockerfile 把文件系统版本写死为 1，不能用同一个镜像跑四组题。本目录提供参数化
Dockerfile：

```bash
BASE_IMAGE='ubuntu@sha256:填写64位镜像摘要'
python -m compositional_intercode_bash.build_images \
  --base-image "$BASE_IMAGE"
```

它构建：

```text
intercode-nl2bash-fs1
intercode-nl2bash-fs2
intercode-nl2bash-fs3
intercode-nl2bash-fs4
```

代码原样运行官方四个 setup 脚本，不修正其中的历史命令或额外安装工具。基础镜像必须
用 `sha256` 摘要固定，`ubuntu:latest` 这类会随时间变化的标签会被拒绝。构建日志和
镜像信息写到 `compositional_intercode_bash/artifacts/docker/`。四个镜像分别写入
`fs1` 到 `fs4` 标签；评测会检查标签和四个不同的 image ID，并直接按不可变 image ID
启动容器，不会在长跑中重新解析可能移动的 tag。

单 UID 调试环境使用已转换的本地基础镜像 ID，并明确打开兼容分支：

```bash
eval "$(compositional_intercode_bash/single_uid_rootless_runtime.sh env)"
export DOCKER_BUILDKIT=0
python -m compositional_intercode_bash.build_images \
  --base-image sha256:199bf54dfbcbd2dfd4415caed456eef4efdcfa9a38ec42117e549e2eb4034ddf \
  --single-uid-rootless
```

转换前的下载包始终保留，不会被覆盖。转换报告在
`artifacts/docker-download/single_uid_rewrite.json`，四镜像 ID 和标签在
`artifacts/docker/image_manifest.json`。

## 8. 正式评测

10 轮主结果：

```bash
python -m compositional_intercode_bash.evaluate \
  --checkpoint compositional_intercode_bash/runs/tapmem_d1729_m42 \
  --output-dir compositional_intercode_bash/evaluations/tapmem_d1729_m42 \
  --max-turns 10 \
  --resume
```

Base+ToolDesc 使用同一个评测入口：

```bash
python -m compositional_intercode_bash.evaluate \
  --checkpoint compositional_intercode_bash/artifacts/base_tool_desc_llama1b \
  --output-dir compositional_intercode_bash/evaluations/base_tool_desc_llama1b \
  --max-turns 10 \
  --resume
```

单轮结果把 `--max-turns` 改成 1。四组任务固定为 60、53、60、27，共 200 题。
评测默认只接受 `formal_ready=true` 的 memory checkpoint 或 base-text artifact；
调试产物必须显式增加 `--allow-exploratory-checkpoint`，结果不能作为论文主结果。
使用 `--limit` 或 `--fs-id` 可以调试部分题，但 summary 的
`evaluation_complete=false`、`paper_ready=false`；只有正式 checkpoint、完整
60/53/60/27 四组任务和正确镜像同时满足时，`paper_ready` 才为 true。
需要跨文件系统做固定诊断时，可重复传入精确任务编号，例如
`--task-id fs1:000 --task-id fs2:000 --task-id fs3:000 --task-id fs4:000`。
该选项不能与 `--limit` 或 `--fs-id` 同时使用，结果同样只属于诊断结果。

### 8.1 超长 shell 输出

模型命令先在容器中完整执行，随后直接在未被 wrapper 改写的容器状态上调用官方
`submit`；action 的 stdout 不参与文件系统子分数。评分完成后，stdout 才被处理成下一
轮模型看到的固定有界反馈：

- 不超过 2,048 UTF-8 bytes 时原样保留；
- 超过时保留前 896-byte 窗口和后 896-byte 窗口，中间写入原始字节数和
  SHA-256；
- 按 UTF-8 解码时若窗口正好切在多字节字符中间，只丢掉不完整的边界字节；
- episode 同时记录原始字符数、字节数、SHA-256、实际保留字节数和是否截断。

这个规则不依赖模型 tokenizer，Base+ToolDesc、TokMem 和 TapMem 得到相同的原始反馈
边界。`reward_info` 只用于日志，其中的超长字符串也按同一规则保存，不回喂模型。
若已经执行并评分的最新一轮在有界处理后仍放不进上下文，该题记录
`termination_reason=context_overflow` 并保留已有 reward；完整 summary 会报告
overflow、截断轮数、最大原始输出和历史删除次数，而且只要出现 overflow，
`paper_ready/formal_ready` 就是 false，不能静默当作完整论文结果。

### 8.2 正式分片

在具有 cgroup v2 的标准 rootless Docker 或隔离 rootful Docker 机器上，可以让多个
进程共同执行同一个完整 200 题 run。所有进程必须使用完全相同的 checkpoint、
依赖、镜像 ID、参数和输出目录，只改变 GPU 与 shard index：

```bash
CUDA_VISIBLE_DEVICES=<gpu> \
python -m compositional_intercode_bash.evaluate \
  --checkpoint compositional_intercode_bash/runs/tapmem_d1729_m42 \
  --output-dir compositional_intercode_bash/evaluations/tapmem_d1729_m42 \
  --max-turns 10 \
  --shard-count 4 \
  --shard-index <0到3> \
  --resume
```

分片按每个文件系统内部的 `local_index % shard_count` 分配。逻辑任务集始终是完整
60/53/60/27，因此各 episode 的 run key 与非分片正式评测一致；shard 参数只决定
当前进程实际执行哪些题，不进入 episode identity。不同 shard 向共享 episode 目录
写互不重叠的文件，并各自只生成 `shard_summary_*`，绝不提前生成正式 summary。

全部 shard 完成后，去掉两个 shard 参数，用相同命令和 `--resume` 再运行一次。程序
会要求 episode 文件集合恰好是官方 200 题，并逐题核对 run key、完整 run inputs、
任务、镜像、checkpoint、runner 和 runtime 身份；文件已经齐全时不会再次执行题目，
核验通过后才写 `summary_10_turn.json`。若缺题，普通 resume 会继续补齐后再做同样的
严格检查。`--task-id`、`--fs-id` 和 `--limit` 不能与正式分片混用。

当前 single-UID fallback 没有可委派的 cgroup v2，仍然必须串行，不能因为机器上有
多张 GPU 就在这个调试 daemon 中并发执行模型命令。评测会记录 Docker runtime，
并把这种环境的 `input_paper_ready/paper_ready/formal_ready` 强制设为 false。

官方 reset 只恢复 Git 纳管的工作树，并删除未跟踪但未被忽略的文件；它不会恢复被
Git 忽略的系统目录，也不会主动终止后台进程。这是官方 InterCode-Bash 的既有假设。
串行且不中断的完整 200 题最接近官方发布流程；若使用 shard 形成论文结果，应先在
标准 Docker 上确认相关任务不会通过 ignored 路径或后台进程影响后续题。

每轮流程与官方 n-turn runner 一致：

```text
生成命令 → 执行命令 → submit 评分
```

InterCode 的 `submit` 每次都会返回 `done=True`，但 Try-Again 协议仍在同一任务容器
累计修改并继续尝试，直到 `reward == 1.0` 或达到轮数上限。0.67 不是成功。

题间恢复方式与官方 InterCode-Bash 一致。每个评测进程在每个文件系统上创建一对
agent/evaluation 容器，同一文件系统的题目复用这对容器。每道新题开始时，
`BashEnv.reset(local_index)` 会在 agent 容器中执行：

```bash
git reset --hard
git clean -fd
```

evaluation 容器仍由官方 `get_reward()` 在每次 `submit` 评分前执行同样的重置。一个
文件系统的全部待执行题完成后，wrapper 才停止并删除这对临时容器和临时镜像标签。
官方 `close()` 本身只停止容器；这里在批次结束时额外删除随机命名的临时资源，不会
改变题间状态或评分。
不同文件系统、不同 evaluator 进程和并发 shard 仍使用不同的随机容器别名，不会共享
状态。Base+ToolDesc 的 data/model seed 记为空，因为它不训练。模型下一轮只看到：

- 原任务；
- 实际执行的 Bash action；对 TokMem/TapMem 会先精确删除 memory/EOC control ID；
- 该 action 的真实 observation 的上述有界首尾反馈和标量 reward。

wrapper 只在每个文件系统首次构造环境时关闭官方两次固定 3 秒等待，随后循环检查
两个容器在同一轮都处于 `running` 且能成功执行 `true`。题间不重复构造容器，只调用
上述官方 reset。reset 失败会立即中止当前 run 并清理该文件系统环境；已经完整写入的
episode 可由 `--resume` 继续复用。全量 resume 若某个文件系统没有缺题，就不会为它
启动容器。启动和生命周期策略都写入 episode identity，所有方法必须统一使用同一
版本。Docker SDK 的 API timeout 固定为 300 秒，避免 single-UID VFS 创建容器时被
默认 60 秒提前中断；readiness 的 20 秒仍只用于容器创建完成后的主动探测。

`reward_info` 中可能包含 gold execution output，只保存有界日志，绝不回喂模型。
`--resume` 还会核对 checkpoint/base artifact、tokenizer、文字工具目录文件哈希、
基础模型身份、推理设备和依赖版本、系统提示词、runner/checkpoint 代码、官方
InterCode 环境与评分源码、
Docker image ID、任务文件哈希、上下文长度和生成长度；任一项变化都会拒绝复用旧
episode。容器或临时镜像在文件系统批次结束时清理；清理失败会显式报错，不会静默
积累残留资源。

评测不替换官方的 `GIT_STATUS_SCRIPT`、`BashEnv.parse_status` 或
`BashEnv.get_reward`。正常题仍由官方 `git status --short`、官方三项 reward 和
`reward == 1.0` 成功条件直接给分。启动前还会核对 InterCode v1.0.1 固定提交
`7c311cce135bd306ff47549def69f3fb0944d35d` 的关键源码哈希，并确认运行时导入的
`bash_env.py`、`ic_env.py` 和 `utils.py` 就来自这份已核验源码。

只增加一个样本级例外：runner 在模型 action 后主动调用 `submit` 时，如果固定版本的
官方 `BashEnv.get_reward` 正在解析 agent 侧 `diff_agent`，并由固定版本的
`BashEnv.parse_status` 抛出 `IndexError`，才把该题标为失败并立即结束。这里只把
parser 失败的当前轮记 0；该题保留此前各轮由官方 scorer 已经给出的最高
reward，但 `success=false`。如果第一轮就失败，题目最高 reward 才是 0。评测保存生成
命令、轮次、实际待解析状态的字节数、SHA-256、有界预览、切分字段数和 traceback
函数链，然后关闭这对容器并从原镜像新建一对容器，再跑下一题，避免受损环境污染后续
样本。

evaluation/gold 侧的 parser 错误、parser 之外的 `IndexError`、其他异常以及模型直接
生成字面量 `submit` 时触发的异常，都不进入这个例外，仍按官方自身行为或原始异常处理。
普通 reward 小于 1 的失败题也只使用下一题的官方 reset，不重建容器。该口径由
`status_parse_failure_policy`、episode schema 和 runner code hash 共同绑定；summary
单独报告 `official_status_parse_error_episodes`。

## 9. CPU 测试

不需要 GPU 或 Docker：

```bash
python -m unittest discover \
  -s compositional_intercode_bash/tests \
  -t . \
  -v
```

测试覆盖：

- simple command、管道、`|&`、Unicode、尾部注释和嵌套 command substitution；
- 字符和普通 token ID 的完整还原；
- unigram forward/backward、EM、Viterbi 和固定数量 FFBS；
- 重复序列合并前后的似然、EM 统计量与删减结果一致；
- K 超过原生 reserved token 后不截断、扩词表和全体 procedure 正交；
- TRAIN-only coverage anchor 和每个原生/新增 procedure 的正例硬门禁；
- TokMem/TapMem 目标及 TCRA 的因果 shift；
- 全候选来源删除、完整数据血缘、主实验门禁、view ID 和 `2A+2R` 批次顺序；
- Base+ToolDesc 的 TRAIN-only 全目录、token 上限、零训练参数和贪心解码；
- `4×1`、`2×2`、`1×4` 的有效 batch 损失与梯度一致；
- 多 ID 回复结束符和精确 control-ID 删除；
- InterCode `done=True` 后继续重试、UTF-8 有界 observation、历史成对删除和
  context overflow 记录；
- 正常样本原样调用官方 scorer；只处理模型 action 后、agent 侧 `diff_agent` 解析中
  的官方 `parse_status` `IndexError`，失败轮记 0，题目保留此前官方
  `max_reward`、`success=false`，然后立即结束并重建容器；
- 正式分片覆盖且不重叠、200 题严格合并、共享 prompt 原子写入；
- 镜像标签、双容器同时 readiness、同一文件系统题间复用、逐题官方 reset、固定等待
  恢复、失败清理和 client 关闭。

真实 Docker reward、文件系统 reset 和 200 题执行需要在安装可选依赖且拥有 Docker
权限的机器上另行验证。
