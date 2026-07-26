# 普通用户 Docker、TapMem 训练和真实环境测试记录

日期：2026-07-26

## 结论

这次已经把完整工程链路跑通：

```text
普通用户启动 rootless Docker
→ 构建并启动 fs1、fs2、fs3、fs4
→ 训练 TapMem
→ 只保存新增参数
→ 重新加载 checkpoint
→ 模型生成 Bash 命令
→ 官方双容器执行和判分
→ 清理临时容器
```

没有使用 `sudo`，没有安装系统服务，也没有连接系统 Docker socket。最终四文件系统
诊断共 4 题，10 轮上限，严格成功 1 题，成功率 25%，平均连续分数 0.78。

这 4 题用于确认下载、启动、checkpoint 加载、命令执行和判分链路，不是完整 200 题
论文结果。完整论文结果仍需跑 60/53/60/27 共 200 题，并在标准 Docker 隔离机器上
复核。

## Docker 是怎么运行的

使用的是 Docker 26.1.3 官方静态程序：

- `docker-26.1.3.tgz`
  - SHA-256：`a50076d372d3bbe955664707af1a4ce4f5df6b2d896e68b12ecc74e724d1db31`
- `docker-rootless-extras-26.1.3.tgz`
  - SHA-256：`864852d0210582ef6618f8f3a1de786c31ec53581541769b0db05d71ebfda97b`

启动后的实际状态：

```text
server_version=26.1.3
security=rootless
storage_driver=vfs
default_runtime=singleuid
network=vpnkit
host loopback access=disabled
```

四个镜像都通过了启动、文件布局、常用命令、Git 初始状态和软件包状态检查：

| 文件系统 | 镜像 ID |
|---|---|
| fs1 | `sha256:b2d0dbf7ce34bd9511eda71df9668e8ad26032e33dd9f2648e960aabfe1fa037` |
| fs2 | `sha256:8ac3468fde929e3b0f7220198c2de11a6ee31d10bcd1369d946d339bf1851e67` |
| fs3 | `sha256:983f693c8b48fddb4f276f24c160f76ffc1b6e560d4c8cd1e6336c1311f70cd5` |
| fs4 | `sha256:3b5d69eef67890922ab65649fbbcb1128d1d9b045c3d96bcc99e69ddc8f442d5` |

还额外用 fs1 第 0 题的参考命令检查了官方完整判分流程。agent 容器和
evaluation 容器分别执行后，三部分得分均为 0.33，总 reward 为 1.0。这证明不是只
比较模型输出字符串，而是真正在两个隔离环境中执行并比较结果。

可复现启动入口：

```bash
compositional_intercode_bash/single_uid_rootless_runtime.sh doctor
compositional_intercode_bash/single_uid_rootless_runtime.sh foreground
```

另一个终端执行：

```bash
eval "$(compositional_intercode_bash/single_uid_rootless_runtime.sh env)"
compositional_intercode_bash/single_uid_rootless_runtime.sh smoke
```

单 UID 方案会压平容器文件所有者，并对软件包安装和 `devpts` 做有限兼容，因此只能
作为没有管理员权限时的工程验证环境。它没有给当前用户增加任何宿主权限，但也不等同
于标准 Docker 论文环境。

## 训练配置

基础模型：

```text
/data/shilong/tokmem/models/Llama-3.2-1B-Instruct
```

主要配置：

| 项目 | 值 |
|---|---:|
| 方法 | TapMem |
| procedure 数量 K | 292 |
| 数据设置 | sampled-1729 |
| 模型随机种子 | 42 |
| 调度 epoch | 100 |
| 训练展示数 | 5,200 |
| 参数更新次数 | 1,300 |
| batch size | 4 |
| 学习率 | 5e-3 |
| 数据类型 | bfloat16 |
| 基础模型状态 | 冻结、eval 模式 |
| 可训练参数数 | 1,198,372 |

训练完成时：

```text
final_loss=0.9284021374
overall_loss=4.1363852561
overall_ar_loss=2.8979176411
overall_route_loss=12.3846761500
formal_ready=true
```

## checkpoint 是否保存了整个模型

没有。`trainable.safetensors` 只有 2,397,096 字节，SHA-256 为：

```text
ab4ce1c40871351fd4d636cd8a1516064a4e9527f8597ac67d5086a9667e9090
```

文件中只有四个新增张量：

| 张量 | 形状 | 参数数 |
|---|---:|---:|
| `procedure_embeddings` | `[292, 2048]` | 598,016 |
| `eoc_embedding` | `[1, 2048]` | 2,048 |
| `routing_head.weight` | `[292, 2048]` | 598,016 |
| `routing_head.bias` | `[292]` | 292 |
| 合计 |  | 1,198,372 |

目录中没有 `model.safetensors`、`pytorch_model*.bin`、`.pt` 或 `.pth`。整个 run
目录约 21 MB，其中约 17.2 MB 是扩展后的 tokenizer，约 1.3 MB 是逐步训练日志。
测试加载时会从原路径重新读取冻结的基础模型，再叠加上述四个张量。

## 最终四文件系统测试

测试命令使用最终源码和训练后的 checkpoint：

```bash
CUDA_VISIBLE_DEVICES=3 \
DOCKER_HOST=unix:///tmp/tokmem-intercode-rootless-1002/docker.sock \
python -m compositional_intercode_bash.evaluate \
  --checkpoint compositional_intercode_bash/runs/tapmem_llama1b_sampled1729_seed42 \
  --output-dir compositional_intercode_bash/evaluations/tapmem_llama1b_sampled1729_seed42_diag4_10turn_final \
  --max-turns 10 \
  --task-id fs1:000 \
  --task-id fs2:000 \
  --task-id fs3:000 \
  --task-id fs4:000 \
  --dtype bfloat16 \
  --device cuda
```

结果：

| 任务 | 成功 | 最高 reward | 使用轮数 |
|---|---:|---:|---:|
| fs1:000 | 否 | 0.76 | 10 |
| fs2:000 | 否 | 0.69 | 10 |
| fs3:000 | 是 | 1.00 | 1 |
| fs4:000 | 否 | 0.67 | 10 |
| 总体 | 1 / 4 | 平均 0.78 | 平均 7.75 |

fs3:000 的任务是查找 `/workspace` 下 30 天内修改过的文件。模型第一轮生成：

```bash
find /workspace -type f -mtime -30
```

官方环境执行后直接返回 reward 1.0。fs1:000 中，模型根据第一轮路径报错继续修正，
最高分从 0.67 提升到 0.76，但去重逻辑仍不正确，所以不能算成功。0.67、0.69 或
0.76 都只是部分正确，严格成功只认 1.0。

最终一次测试耗时 255.04 秒。summary 的 SHA-256：

```text
7f6de54eaadc728f64c2260d6b4d0f8745d3ca62dab42de053eccbbf64fa540f
```

独立检查已经确认：

- 4 个 task ID 唯一；
- summary 的成功数和平均分能从四个 episode 重新算出；
- `turns_taken` 等于实际记录轮数；
- `max_reward` 等于各轮 reward 最大值；
- success 当且仅当至少一轮 reward 为 1.0；
- 每题结束后没有遗留 `tokmem-intercode-*` 容器或临时镜像标签。

## 验证

CPU 测试：

```text
Ran 67 tests in 19.379s
OK
```

真实环境检查：

```text
fs1 smoke-ok
fs2 smoke-ok
fs3 smoke-ok
fs4 smoke-ok
official gold reward=1.0
trained checkpoint: 1/4 success, mean reward=0.78
```

下一步若要形成审稿回复中的正式结论，应在标准隔离 Docker 环境中，对 TapMem、
TokMem 和 Base+ToolDesc 使用完全相同的完整 200 题、10-turn 设置评测；这 4 题只能
作为当前实现和运行环境已经打通的证据。

## 200 题串行尝试暴露的问题与修复

随后使用同一 TapMem checkpoint 启动了完整 200 题、10-turn 串行测试。旧 runner
完成 11 题后在 `fs1:011` 中止：某轮命令产生的 shell 输出经 tokenizer 编码后约为
4,631,749 token，而下一轮 prompt 预算只有 7,680 token。旧程序只会删除更早的完整
历史轮，不能处理“最新 observation 自身已经超长”，因此抛出
`ContextOverflowError`。这 11 题不是完整结果，也不能与修复后的 runner 混合。

新的统一协议是：

1. action 先完整执行，再在未被 wrapper 改写的容器状态上调用官方 `submit`；stdout
   不参与文件系统子分数；
2. 评分后才构造下一轮模型反馈，最多 2,048 UTF-8 bytes；
3. 超限时保留前后各 896-byte 窗口，并记录原始长度、SHA-256 和实际保留长度；
4. Base+ToolDesc、TokMem、TapMem 使用完全相同的字节边界，不按 tokenizer 分别裁剪；
5. summary 显式报告截断、历史删除和 context overflow；任何 overflow 都会使完整
   summary 的 `paper_ready/formal_ready=false`。

同时把官方每个容器固定等待 3 秒改为主动 readiness 探测。每个文件系统首次创建的
两个容器必须在同一轮均为 `running`，且能成功执行 `true`，之后才进入 reset。

后续核对官方源码后，题间恢复方式已经改成 InterCode-Bash 的原始方式：同一评测
进程中，每个文件系统只创建一对 agent/evaluation 容器；每道题开始时调用
`BashEnv.reset(local_index)`，由官方代码执行 `git reset --hard; git clean -fd`；
evaluation 容器在每次 `submit` 前由官方 `get_reward()` 重置。一个文件系统的待执行
题全部结束后才停止并删除容器。官方最后只停止容器；wrapper 额外删除批次使用的随机
临时容器和标签。reset、推理、评分或写盘失败时立即清理并报错，
`--resume` 只为仍有缺题的文件系统创建环境。

single-UID VFS 上一次真实两题测试还暴露出容器创建可能超过 Docker SDK 默认 60 秒。
当前 wrapper 将官方容器 client 的 API timeout 固定为 300 秒，并把它写入 runner
identity；容器创建完成后的双容器 readiness 仍使用独立的 20 秒轮询期限。

真实 reset 冒烟测试随后通过：

- 在 fs1 的 agent 容器中执行第一题 reset，创建
  `/tapmem_intercode_reset_probe`；
- `git status --short` 显示该文件为未跟踪文件；
- 在同一个容器 ID 中执行第二题 reset；
- 探针文件消失，`git status --short` 恢复为空；
- 测试结束后没有 `tokmem-intercode-*` 容器或临时镜像标签残留。

之后用训练完成的 TapMem checkpoint 对 `fs1:000` 和 `fs1:001` 做了真实一轮测试。
基础模型在 GPU 上加载，模型生成的命令进入同一对复用容器执行，再由官方 evaluation
容器运行标准答案并计算 reward。结果如下：

| 任务 | 模型命令 | 命令执行 | reward | 成功 |
|---|---|---:|---:|---:|
| `fs1:000` | `find /testbed/.java -type f -name "*.java" -exec md5sum {} \;` | 失败，路径不存在 | 0.67 | 否 |
| `fs1:001` | `du -s /testbed/dir2/subdir2 \| md5sum \| cut -c1-` | 成功 | 0.67 | 否 |

两题平均 reward 为 0.67，没有 observation 截断或 context overflow，总耗时
179.42 秒。0.67 来自文件状态两项各 0.33 和初始 0.01；输出答案不正确，所以严格
成功率为 0/2。这是执行链路和题间 reset 的小规模验证，不是方法效果结论。结果目录：

```text
compositional_intercode_bash/evaluations/
  tapmem_llama1b_sampled1729_seed42_fs1_two_task_reset_smoke_v5/
```

`summary_1_turn.json` 的 SHA-256 为：

```text
31ed1665cce5bc6db32fc3f7750f65e00b0558257201e51b4c296c873e6c7432
```

该结果只选了两题，而且 Docker runtime 是无 cgroup v2 的 single-UID fallback，
因此 `evaluation_complete=false`、`paper_ready=false`、`formal_ready=false`。

标准 Docker 环境还可以通过 `--shard-count/--shard-index` 对完整 200 题做进程级
分片。shard 只改变执行集合，episode identity 仍按完整 200 题计算；全部 shard 完成
后，由无 shard 的 `--resume` 严格核验 200 个 episode 再生成正式 summary。当前
single-UID fallback 没有可委派 cgroup v2，依然只允许串行验证，不能在这里启用并发
模型命令。

修复后的真实回归结果：

| 检查 | 结果 |
|---|---|
| 官方 fs1:000 参考命令 | reward 1.0，三项子分数均为 0.33 |
| `fs1:011` episode | 正常完成 10 轮，无 context overflow |
| `fs1:011` 最高 reward | 0.39 |
| 最大原始 observation | 16,630,596 UTF-8 bytes |
| 模型实际看到的有界反馈 | 1,969 UTF-8 bytes |
| 后续最大 prompt | 4,086 token |
| episode 文件大小 | 76 KB |
| 单题总耗时 | 135.56 秒 |
| 资源清理 | 无临时容器或临时镜像标签残留 |
| CPU 测试 | 当前 106 项全部通过 |

单题回归产物位于：

```text
compositional_intercode_bash/evaluations/
  tapmem_llama1b_sampled1729_seed42_fs1_011_observation_regression_v4/
```

这个 `runner_v4` 单题目录是在“每题新建容器”的旧生命周期下生成的，只保留作超长
输出回归证据。当前 runner protocol 已升级，run identity 也包含官方题间 reset
策略，因此旧 episode 不能与新的完整评测混用或 resume。

评测还会把 Docker server version、存储驱动、默认 runtime、cgroup 与 rootless 状态
写入 run identity。`default_runtime=singleuid`，或者 rootless daemon 没有 cgroup v2
时，`input_paper_ready/paper_ready/formal_ready` 都为 false；完整跑完也只能算工程
结果，不能因题目数达到 200 就被误标为论文正式结果。

## 8B 训练和带空格路径的官方评分崩溃（历史诊断）

Llama-3.1-8B-Instruct 的 TokMem 和 TapMem 已使用 sampled-1729、model seed 42 完成
训练。两者都是 100 个调度 epoch、5,200 个 microbatch、1,300 次参数更新；
`batch_size=1, gradient_accumulation_steps=4` 按完整四条样本的 token/route-site
总数归一化。checkpoint 仍只保存新增参数：

| 方法 | 新增参数 | `trainable.safetensors` |
|---|---:|---:|
| TokMem | 1,196,032 | 2,392,160 bytes |
| TapMem | 2,396,452 | 4,793,256 bytes |

第一次 6-shard 真实评测使用 GPU 1、3、6，每卡两个进程。TokMem shard 0 在
`fs1:015` 评分时触发了官方 parser 的既有缺陷。模型命令创建了带空格的路径：

```text
/testbed/dir3/subdir1/subsubdir1 not
```

Git 返回：

```text
?? "testbed/dir3/subdir1/subsubdir1 not"
```

官方 `parse_status` 对整段文字直接调用 `split()`，把这一行拆成三个字段，随后访问
不存在的第四个字段并抛出 `IndexError`。同一 checkpoint 和初始状态会确定性复现，
简单 resume 不能绕过。其余 shard 因 runner 必须统一而停止；已写出的旧 episode
只保留作诊断证据，不能进入修复后的汇总。

当时的 runner v6 曾把 Git 状态输入改成 `git status --porcelain=v1 -z` 并使用自定义
parser。虽然目的是绕过带空格路径触发的崩溃，但这仍然改变了 benchmark 发布的评分
实现。当前代码已经撤销这项修改，重新直接使用官方 `git status --short` 和官方
`BashEnv.parse_status`。因此下面 runner v6 产生的 200 题结果只保留为历史工程诊断，
不能作为采用官方评测逻辑的论文结果。

这次最小修复没有顺手改官方 p2 中未给路径加 shell 引号的 `md5sum`/`md5deep` 命令。
因此“agent 和 gold 恰好共同新增同一个带空格路径”仍可能触发官方既有的哈希误判；
它不影响上述 agent/gold 路径完全不同的崩溃复现，但正式结果需要把这一点列为
benchmark caveat。

## 修改 parser 后的历史 200 题结果（不作为正式结果）

当时改用自定义 parser 后从全新目录重跑。每种方法按文件系统内题号模 3 分成三个
shard，GPU 1、3、6
各同时放一个 TokMem 和一个 TapMem 进程，共六个 evaluator。每道题都在真实
InterCode 容器中执行模型命令，并由独立 evaluation 容器运行参考命令和计算 reward；
不是只比较模型输出文本。全部 shard 完成后，又用无 shard 的 `--resume` 逐题校验
checkpoint、runner、镜像、runtime 和 task identity，恰好读到 200 个 episode 后才
生成 `summary_10_turn.json`。

TokMem 的 `fs1:015` 再次确定性地产生了带空格路径：

```text
/testbed/dir3/subdir1/subsubdir1 not
```

第一轮 reward 为 0.34，之后继续执行到第 10 轮，没有再触发 parser crash。这道题最终
失败是因为模型没有删除错误目录，不是评测器中止。

六路并发期间，TapMem shard 1 在切换文件系统时遇到一次 Docker 竞态：官方
`get_container()` 先列出全部容器，另一进程恰好删除一个旧容器，随后 inspect 该旧 ID
得到 404。该 shard 使用完全相同的 checkpoint、runner identity 和 shard index 加
`--resume` 恢复；已完成 episode 通过 identity 校验后直接复用，只补跑缺题。其余
shard 未受影响。

历史非官方工程结果如下：

| 方法 | 成功题 | statusz_v6 下 reward==1 比例 | 平均最高 reward | 平均轮数 |
|---|---:|---:|---:|---:|
| TokMem | 85 / 200 | 42.50% | 0.82615 | 6.525 |
| TapMem | 78 / 200 | 39.00% | 0.81445 | 6.855 |

逐文件系统结果：

| 文件系统 | 题数 | TokMem 成功 | TapMem 成功 | TokMem 平均 reward | TapMem 平均 reward |
|---|---:|---:|---:|---:|---:|
| fs1 | 60 | 19（31.67%） | 20（33.33%） | 0.79550 | 0.80433 |
| fs2 | 53 | 25（47.17%） | 21（39.62%） | 0.83604 | 0.79151 |
| fs3 | 60 | 29（48.33%） | 27（45.00%） | 0.83267 | 0.84383 |
| fs4 | 27 | 12（44.44%） | 10（37.04%） | 0.86037 | 0.81667 |

同一道题配对比较后，65 题两者都成功，102 题两者都失败，20 题只有 TokMem 成功，
13 题只有 TapMem 成功。两侧精确 McNemar 检验的 p 值为 0.296；当前单个训练 seed
没有显示 TapMem 优于 TokMem，3.5 个百分点的差距也不能解释成稳定显著差异。这批结果
只证明 modified-parser 的工程链路跑通，不能用于支持官方 InterCode benchmark 上的
泛化或效果结论，也不能作为“TCRA 在这个设置上带来增益”的证据。

两种方法均没有 context overflow，也没有删除历史轮。TokMem 有 6 个 episode、15 个
turn 的 observation 被按统一字节规则裁剪；TapMem 对应为 8 个 episode、24 个 turn。

结果目录和总汇总哈希：

```text
compositional_intercode_bash/evaluations/
  tokmem_llama8b_sampled1729_seed42_full200_10turn_rootless_statusz_v6_parallel6/
    summary_10_turn.json
    SHA-256 2a8474c6a2817b36bcf44d83f7ccd22a8e6f75696736e92c48f40ea8cc6a7668

  tapmem_llama8b_sampled1729_seed42_full200_10turn_rootless_statusz_v6_parallel6/
    summary_10_turn.json
    SHA-256 0edc9b5e4fafbb6f8cf1b1af76a02cdd7cb04d999a2756b651dadacd742b31c7
```

### 不能忽略的官方评分限制

本次镜像忠实沿用官方 Dockerfile 的依赖列表。官方 `bash_env.py` 在比较新增目录内容时
调用 `md5deep -r`，但官方 Dockerfile 没有安装 `md5deep`/`hashdeep`。daemon 日志确认
该命令不存在；agent 和 gold 容器可能返回相同错误文字，官方代码又只比较两段输出是否
相同，因此目录内容项可能被错误记为一致。

按每轮 `diff_same` 反查，本次 TokMem 有 7 个 episode、19 个 turn 可能走到该分支，
TapMem 有 4 个 episode、16 个 turn 可能走到该分支；其中 reward==1 的 episode 分别为
5 和 2。两种方法使用同一套历史 statusz_v6 评分实现，但它与官方 parser 不同，绝对
分数还存在偏高风险。正式评测必须保留官方镜像和依赖并把这个缺陷列为 benchmark
caveat；若另做安装 `hashdeep` 的敏感性分析，必须明确标为非官方设置，并让所有比较
方法从零重跑，不能替代正式结果或在现有 episode 上静默改分。

此外，这次 Docker runtime 为 single-UID rootless、VFS、cgroup v1，所以两个总汇总都
明确记录 `paper_ready=false`。运行结束后 evaluator 数、残留容器和临时镜像标签均为
0，专用 rootless daemon 已正常停止。现有 400 个 episode 是完整真实执行的工程结果，
但在标准隔离 Docker 和评分依赖问题解决前，不应直接作为论文最终表格。

## 2026-07-27：官方 parser 无法解析时的样本级判错口径

151-epoch TapMem checkpoint 的新一轮评测在 `fs1:047` 确定性复现了另一类问题。
第一轮命令限定了目标目录，得到 0.67；第二轮模型生成：

```text
grep -c -h /testbed/dir3/subdir1/subsubdir1/tmp/*/*/* | find -type f -mtime +2 -delete
```

第二个 `find` 没有给起始目录，因此从当前根目录递归删除旧文件，并删除了
`/bin/bash`。下一次评分收到的不是正常 Git 状态输出，而是：

```text
OCI runtime exec failed: ... exec: "/bin/bash": stat /bin/bash: no such file or directory
```

当前 runner 不修改官方 `GIT_STATUS_SCRIPT`、`BashEnv.parse_status`、
`BashEnv.get_reward` 或成功判定。运行前固定核对 InterCode v1.0.1、提交
`7c311cce135bd306ff47549def69f3fb0944d35d` 的关键源码哈希，并确认 Python 实际导入
的模块就是这份源码。正常题仍完整执行官方 `action → submit → reward` 流程。

唯一例外是：runner 在模型 action 后主动调用 `submit`，固定版本的
`BashEnv.get_reward` 正在解析 agent 侧 `diff_agent`，且固定版本的
`BashEnv.parse_status` 抛出 `IndexError`。三个条件同时成立时，才将该题标为失败并
立即结束。parser 失败的当前轮记 0，但该题保留此前各轮由官方 scorer 已经给出的最高
reward；例如第一轮为 0.67、第二轮 parser 失败时，最终
`max_reward=0.67, success=false`。如果第一轮就失败，最高 reward 才是 0。episode
同时保存实际 parser 输入的字节数、SHA-256、有界预览、切分字段数和 traceback
函数链，能够区分 agent 侧与 evaluation/gold 侧错误。

该题结束后关闭这对容器并从原镜像重建，再执行下一题。这是为了避免 `/bin/bash` 等
Git 无法恢复的文件被删后污染后续题，不会改变正常题之间仍使用的官方 reset。
evaluation/gold 侧 parser 错误、parser 之外的 `IndexError`、其他异常和模型直接生成
字面量 `submit` 时触发的异常都不进入该例外，仍按官方自身行为或原始异常处理。普通
reward 小于 1 的失败题不会触发容器重建。

本次改动同步升级了 episode、runner、summary 和 shard schema，并在 run identity 中
写入 `status_parse_failure_policy`。旧目录中已完成的 175 个 episode 因 identity
不同不能复用；使用这一口径的完整 200 题必须在全新目录从头重跑，最终结果完成后再补
在本节。

### 最终口径的真实两题回归

使用 151-epoch TapMem 8B checkpoint 在真实 fs1 容器中连续执行 `fs1:047` 和
`fs1:048`，结果如下：

| 题目 | 轮数 | 各轮 reward | 最高 reward | 成功 | 结束原因 |
|---|---:|---|---:|---|---|
| `fs1:047` | 2 | `[0.67, 0.0]` | 0.67 | 否 | `official_status_parse_error` |
| `fs1:048` | 10 | 最高 0.78 | 0.78 | 否 | `max_turns` |

`fs1:047` 保存的实际 agent 状态为 144 UTF-8 bytes、21 个空格切分字段，SHA-256
为 `bf345324d787b9b559d991f37c88ea056a3267145cb658374c3eb6be033c73d7`，调用链末尾
为 `submit → step → get_reward → parse_status`。这证明触发的是 agent 侧官方 parser
异常；第一轮官方 0.67 被保留，失败轮为 0，题目仍判失败。

随后 `fs1:048` 在重建后的容器中正常完成 10 次官方评分，没有 parser 异常，最高
reward 为 0.78，说明前一题删除 `/bin/bash` 没有污染下一题。两题平均最高 reward 为
0.725；这是两题诊断结果，不是完整 200 题论文结果。产物目录：

```text
compositional_intercode_bash/evaluations/
  tapmem_llama8b_delayed_gate05_e151_seed42_v2_official_agent_parse_preserve_fs1_047_048_v1/
```

`summary_10_turn.json` 的 SHA-256 为
`9e948bb048f1990569ad89be5ca1735de3c9001ac52a4a6e7c79bdcd6dc9854d`。

最终代码的 106 项 CPU 测试全部通过。

### 最终口径的完整 200 题结果

2026-07-27 使用同一个 151-epoch TapMem 8B checkpoint，在全新目录中从头执行完整
200 题。评测按文件系统内题号模 4 分成四个 shard，GPU 1、3、5、6 各运行一个
进程；四个 shard 分别完成 51、50、50、49 题。每道题都实际在 InterCode 容器中执行
模型生成的 Bash 命令，并由独立 evaluation 容器运行参考命令和调用官方 scorer，
不是只比较文本答案。四个 shard 结束后，又用无 shard 的 `--resume` 入口逐题校验
200 个 episode，确认 task ID 没有缺失或重复后生成统一汇总。

总结果：

| 指标 | 结果 |
|---|---:|
| 题目数 | 200 |
| reward 等于 1 的题目 | 76 |
| 成功率 | 38.00% |
| 平均最高 reward | 0.80675 |
| 平均执行轮数 | 7.005 |
| 达到满分后结束 | 76 |
| 运行满 10 轮后结束 | 123 |
| 官方 parser 无法解析后结束 | 1 |
| 上下文溢出 | 0 |

逐文件系统结果：

| 文件系统 | 题数 | 成功题 | 成功率 | 平均最高 reward |
|---|---:|---:|---:|---:|
| fs1 | 60 | 19 | 31.67% | 0.79733 |
| fs2 | 53 | 24 | 45.28% | 0.81075 |
| fs3 | 60 | 24 | 40.00% | 0.81533 |
| fs4 | 27 | 9 | 33.33% | 0.80074 |

唯一一次 parser 失败仍是 `fs1:047`。该题第一轮官方 reward 为 0.67，第二轮 parser
失败记 0，最终保留 `max_reward=0.67`，但 `success=false`；失败后重建容器，后续题
继续正常执行。完整 200 题中没有第二个 parser 失败，也没有上下文溢出或删除历史轮。
共有 6 题、16 轮的原始 observation 超过统一的 2048-byte 输入上限，按既定的首尾
保留规则裁剪；最大一条原始 observation 为 22,360,086 bytes。TapMem 的 memory-bank
约束共触发 1,458 次，其中 610 个生成位置因约束而改变了最终 token。

产物目录：

```text
compositional_intercode_bash/evaluations/
  tapmem_llama8b_delayed_gate05_e151_seed42_v2_official_agent_parse_preserve_full200_v1/
    episodes_10_turn/                         # 200 个唯一 episode
    shard_summary_10_turn_000_of_004.json
    shard_summary_10_turn_001_of_004.json
    shard_summary_10_turn_002_of_004.json
    shard_summary_10_turn_003_of_004.json
    summary_10_turn.json
```

`summary_10_turn.json` 的 SHA-256 为
`ef57f182174eef2f9902da4792f8c27bfe478555b291080cb3e21706735f99e0`。
汇总记录的 checkpoint identity 为
`a9fd47471a53f51f80e93fa4e19b2041fe1053d6d842a8bab22431d9ac5af9bb`，
官方 InterCode 源码 identity 为
`7a9a107bb34b572626b85aa3169959253390d5ad2a7d8c9111ca8a1f9ec4c690`。
统一汇总通过了 200 题数量、`60/53/60/27` 文件系统分布、runner identity、
checkpoint 哈希、官方源码哈希和镜像 ID 检查，`evaluation_complete=true`。

这次仍使用 single-UID rootless Docker、VFS 和 cgroup v1。它符合“不申请管理员权限”
的运行要求，也完整跑通了真实容器评测，但没有标准容器用户身份和 cgroup v2 隔离，
所以汇总如实记录 `input_paper_ready=false`、`paper_ready=false` 和
`formal_ready=false`。因此这是一份采用最终评分口径的完整工程结果；若要把数字直接
放入论文最终表格，建议在具有标准隔离能力的 Docker 环境中用同一 checkpoint 和
runner 再复跑一次。

### 同口径 TokMem 200 题结果和配对比较

为了与上面的 TapMem 结果使用完全相同的 runner，2026-07-27 又从全新目录执行了
TokMem 8B 的完整 200 题。原 TokMem checkpoint 是
`intercode_bash_memory_checkpoint_v3`，而当前 evaluator 只接受 v4。v3 与 v4 对
TokMem 的实际差别只有 v4 多记录了 `memory_bank_probability_threshold`；TokMem
不会启用 memory-bank constraint，这个字段不参与训练或推理。因此没有修改 evaluator
代码，而是保留原 checkpoint 不动，创建一个 v4 元数据兼容副本：

```text
compositional_intercode_bash/runs/
  tokmem_llama8b_sampled1729_seed42_v4_compat/
```

兼容副本的 `trainable.safetensors` 与原 checkpoint 的 SHA-256 均为
`671b0ce37b5b03eceeb4161628d6a08ba956feadf9c7983efb8949e87ab93b66`，
tokenizer 目录也逐文件相同。副本没有保存冻结的 Llama backbone；只在
`checkpoint.json` 中把 schema 标为 v4、记录阈值 0.5，并明确注明该阈值对 TokMem
不生效。这样 TokMem 和 TapMem 的 runner identity 都是
`247d90b3f7bdd15149ea4555cfd8607820019dcef8242d4f2761143599df3456`，
官方源码 identity 和四个 Docker 镜像 ID 也完全相同。

TokMem 总结果：

| 指标 | 结果 |
|---|---:|
| 题目数 | 200 |
| reward 等于 1 的题目 | 86 |
| 成功率 | 43.00% |
| 平均最高 reward | 0.82725 |
| 平均执行轮数 | 6.575 |
| 达到满分后结束 | 86 |
| 运行满 10 轮后结束 | 114 |
| 官方 parser 无法解析后结束 | 0 |
| 上下文溢出 | 0 |

与最终口径 TapMem 的直接对比：

| 方法 | 成功题 | 成功率 | 平均最高 reward | 平均轮数 |
|---|---:|---:|---:|---:|
| TokMem | 86 / 200 | 43.00% | 0.82725 | 6.575 |
| TapMem | 76 / 200 | 38.00% | 0.80675 | 7.005 |
| TokMem - TapMem | +10 | +5.00 个百分点 | +0.02050 | -0.430 |

逐文件系统结果：

| 文件系统 | TokMem 成功 | TapMem 成功 | TokMem 平均 reward | TapMem 平均 reward |
|---|---:|---:|---:|---:|
| fs1 | 21 / 60 | 19 / 60 | 0.80700 | 0.79733 |
| fs2 | 25 / 53 | 24 / 53 | 0.83283 | 0.81075 |
| fs3 | 31 / 60 | 24 / 60 | 0.83983 | 0.81533 |
| fs4 | 9 / 27 | 9 / 27 | 0.83333 | 0.80074 |

逐题配对后，64 题两者都成功，102 题两者都失败，22 题只有 TokMem 成功，12 题
只有 TapMem 成功。两侧精确 McNemar 检验的 p 值为 0.12145：TokMem 的成功率数值
更高，但单个训练 seed 下还没有达到 0.05 显著性水平。连续 reward 方面，TokMem
更高的题有 55 道，TapMem 更高的题有 30 道，另外 115 道相同；只比较非平局题的
两侧符号检验 p 值为 0.00884。这个结果说明当前两个 checkpoint 上，TokMem 不仅成功
题更多，平均 reward 也更高；它不支持“TapMem 在 InterCode-Bash 上优于 TokMem”的
说法。

仍需注意训练预算没有完全对齐：TokMem checkpoint 训练 100 epochs，TapMem
checkpoint 训练 151 epochs。两者使用相同基础模型、data seed、model seed、procedure
inventory 和评测 runner，但不同训练轮数仍是方法比较中的混杂因素。若要把结果解释为
纯粹的 TokMem/TapMem 方法差异，应再训练一个 151-epoch TokMem，或把两种方法都按
相同更新次数重新训练多个 seed。当前结果可以可靠比较这两个具体 checkpoint，不能据此
断言 TokMem 方法在所有训练设置下都优于 TapMem。

TokMem 有 5 题、18 轮 observation 被统一规则裁剪，TapMem 对应为 6 题、16 轮；
两者都没有删除历史轮或 context overflow。TokMem 有 27 个生成轮没有正常结束标志，
TapMem 为 7 个，说明 EOC 确实改善了回答结束控制，但这一点没有转化为更高的任务成功
率。TapMem 的 memory-bank 约束触发 1,458 次，并改变 610 个最终 token；TokMem
按定义不使用该约束。

TokMem 结果目录：

```text
compositional_intercode_bash/evaluations/
  tokmem_llama8b_sampled1729_seed42_official_agent_parse_preserve_full200_v1/
    episodes_10_turn/                         # 200 个唯一 episode
    shard_summary_10_turn_000_of_004.json
    shard_summary_10_turn_001_of_004.json
    shard_summary_10_turn_002_of_004.json
    shard_summary_10_turn_003_of_004.json
    summary_10_turn.json
```

TokMem `summary_10_turn.json` 的 SHA-256 为
`e0cc2b865c39632f85139c2ec594755d556389bba1bde8111a95911211bf91d5`。
200 个 episode、200 个唯一 task ID 和四个 shard summary 均已核对，统一汇总记录
`evaluation_complete=true`。与 TapMem 一样，本次仍使用 single-UID rootless Docker
和 cgroup v1，所以 `paper_ready=false`；标准隔离环境要求仍未满足。
