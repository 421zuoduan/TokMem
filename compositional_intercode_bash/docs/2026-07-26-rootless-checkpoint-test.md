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

## TapMem 阈值 1.0、100 epochs 的完整结果

### 目的和设置

2026-07-27 又训练并测试了一版 TapMem，把
`memory_bank_probability_threshold` 从 0.5 改为 1.0。其他主要训练设置保持不变：

| 设置 | 数值 |
|---|---:|
| 基础模型 | Llama-3.1-8B-Instruct |
| 训练轮数 | 100 epochs |
| 样本呈现次数 | 5,200 |
| 参数更新次数 | 1,300 |
| batch size | 1 |
| gradient accumulation | 4 |
| memory learning rate | 0.005 |
| routing learning rate | 0.005 |
| model seed | 42 |
| data seed | 1729 |

训练数据、procedure inventory、基础模型和随机种子都与前面的实验相同。最终 loss
为 0.73842。checkpoint 位于：

```text
compositional_intercode_bash/runs/
  tapmem_llama8b_nogate_threshold1_e100_seed42_v1/
```

checkpoint 只保存 4 个训练张量：procedure embedding、EOC embedding、routing head
权重和偏置，共 2,396,452 个参数；没有保存冻结的 8B backbone。
`trainable.safetensors` 的 SHA-256 为
`8459afefc208f4e03b6c8ca45a62d31dcfb875bd3b71766737b86c2e1ea0aa9f`。

阈值 1.0 应理解为“尽量关闭基于概率总和的硬门控”，不能写成严格的“完全没有硬
门控”。生成时程序先用 float32 计算全部 memory token 的概率总和，只有计算结果
`>= 1.0` 才强制从 memory bank 选 token。通常这个条件不会满足；但是当普通 token
的概率相对极小时，float32 的减法和指数运算可能把总和舍入为恰好 1.0，此时仍会触发
硬门控。EOC 和 TCRA 的 logit bias 始终保留，这版仍然是 TapMem。

这个阈值只用于生成，不进入训练 forward 和 loss。因此“用阈值 1.0 训练”的实际含义
是：按同一 TapMem 训练方法得到 checkpoint，并在 checkpoint 中固定记录阈值 1.0，
评测加载后按这个阈值生成。仓库里原有的 100-epoch TapMem checkpoint 使用旧版 v3
训练语义，不能作为“同一权重、只改阈值”的严格对照。

### 完整 200 题结果

完整结果如下：

| 指标 | 结果 |
|---|---:|
| 题目数 | 200 |
| reward 等于 1 的题目 | 76 |
| 成功率 | 38.00% |
| 平均最高 reward | 0.80730 |
| 平均执行轮数 | 6.930 |
| 达到满分后结束 | 76 |
| 运行满 10 轮后结束 | 124 |
| 官方 parser 无法解析后结束 | 0 |
| 上下文溢出 | 0 |

逐文件系统结果：

| 文件系统 | 题数 | 成功题 | 成功率 | 平均最高 reward |
|---|---:|---:|---:|---:|
| fs1 | 60 | 19 | 31.67% | 0.79767 |
| fs2 | 53 | 25 | 47.17% | 0.81528 |
| fs3 | 60 | 24 | 40.00% | 0.81117 |
| fs4 | 27 | 8 | 29.63% | 0.80444 |

完整评测中，概率阈值导致的 memory-bank 硬门控触发 961 次，其中 350 次改变了原本
会生成的 token。这证实阈值 1.0 大幅减少了门控，但没有在 float32 数值意义上把它
彻底关掉。作为参考，前一版阈值 0.5、151-epoch TapMem 分别为 1,458 次和 610 次。
两版 checkpoint 的训练轮数和权重不同，所以触发次数的减少不能全部归因于阈值。

阈值 1.0 版本有 15 个生成轮没有正常结束标志，5 题、15 轮 observation 被统一规则
裁剪；没有删除历史轮、parser 失败或 context overflow。

### 与已有结果的比较

| 方法 | 训练轮数 | 阈值 | 成功题 | 成功率 | 平均 reward | 平均轮数 |
|---|---:|---:|---:|---:|---:|---:|
| TapMem，本次 | 100 | 1.0 | 76 / 200 | 38.00% | 0.80730 | 6.930 |
| TapMem，前一版 | 151 | 0.5 | 76 / 200 | 38.00% | 0.80675 | 7.005 |
| TokMem | 100 | 不使用 | 86 / 200 | 43.00% | 0.82725 | 6.575 |

本次 TapMem 与阈值 0.5 TapMem 逐题比较：55 题都成功，103 题都失败，21 题只有本次
成功，21 题只有阈值 0.5 版本成功。两侧精确 McNemar 检验 `p=1.0`。连续 reward
方面，本次更高 46 题，阈值 0.5 版本更高 42 题，112 题相同；两侧符号检验
`p=0.74933`。两者的总体结果基本持平，但由于训练轮数为 100 对 151，不能把这个
比较解释为纯粹的阈值消融。

本次 TapMem 与 100-epoch TokMem 逐题比较：62 题都成功，100 题都失败，14 题只有
TapMem 成功，24 题只有 TokMem 成功。两侧精确 McNemar 检验 `p=0.14331`。连续
reward 方面，TapMem 更高 31 题，TokMem 更高 47 题，122 题相同；两侧符号检验
`p=0.08878`。TokMem 数值上仍高 5 个百分点，但单个 seed 的成功率差异没有达到
0.05 显著性水平。

因此，这次实验没有显示把阈值提高到 1.0 能提升 TapMem 的最终成功率。它与旧
TapMem checkpoint 的成功率相同，并且仍低于当前 TokMem checkpoint。若要把差异
严格归因于阈值，下一步应固定本次同一组训练权重，只生成两个仅阈值元数据不同的
checkpoint，分别用 0.5 和 1.0 跑同一批 200 题；不需要重新训练两次。

### 并行执行和产物核验

评测先分成四个 shard：GPU 3 跑 shard 0、1，GPU 5 跑 shard 2、3，每张卡两个模型
进程。峰值显存低于 50 GB/卡。shard 3 在 fs3 切换到 fs4 时遇到一次 rootless Docker
并发竞态：一个进程刚列出容器，另一个进程随即清理了它，前者 inspect 时收到 404。
这个错误发生在新环境初始化阶段，没有写入半道 episode，也没有改变评分。其他三个
shard 完成后，用完全相同的命令和 `--resume` 串行补完 shard 3 的 6 道 fs4 题。

最后又运行一次无 shard 的 `--resume`。程序核对了 200 个 episode、200 个唯一 task
ID、`60/53/60/27` 的文件系统分布、四个完整 shard summary、checkpoint identity、
runner identity、官方源码 identity 和镜像 ID，统一汇总记录
`evaluation_complete=true`。

结果目录：

```text
compositional_intercode_bash/evaluations/
  tapmem_llama8b_nogate_threshold1_e100_seed42_v1_official_agent_parse_preserve_full200_v1/
    episodes_10_turn/
    shard_summary_10_turn_000_of_004.json
    shard_summary_10_turn_001_of_004.json
    shard_summary_10_turn_002_of_004.json
    shard_summary_10_turn_003_of_004.json
    summary_10_turn.json
```

`summary_10_turn.json` 的 SHA-256 为
`5a135695f892cb847daa05b8e13b6de14f9180b6cfad952cad361b794a08da04`；
checkpoint identity 为
`c243f45d582f9c843a12e56f44bf7636aa4bcf5d2cd8d881c1be1f863aec3173`；
runner identity 仍为
`247d90b3f7bdd15149ea4555cfd8607820019dcef8242d4f2761143599df3456`。
本次继续使用 single-UID rootless Docker、VFS 和 cgroup v1，因此
`paper_ready=false`；如需把数字放进论文正式表格，仍应在具有标准隔离能力的 Docker
环境中复跑。

## procedure 外屏蔽 EOC 后的 50-epoch 后台复训

2026-07-27 修正了 TapMem 的 EOC 状态约束：procedure 外先把 EOC logit 设为负无穷，
再用同一份已屏蔽 logits 计算原始 top-1、memory-bank 概率和 TCRA 后的最终选择；
进入 procedure 后仍允许 EOC 正常结束 procedure。这样孤立 EOC 不会再进入当前回答
的自回归上下文。评测 runner protocol 同步从 v12 更新为 v13。

针对修改后的生成逻辑，更新了原先允许 orphan EOC 的单测。定向 13 项测试和整个
`compositional_intercode_bash` 的 106 项 CPU 测试全部通过。

本轮继续使用 Llama-3.1-8B-Instruct、data seed 1729、model seed 42、batch size 1、
gradient accumulation 4、memory/routing learning rate 0.005 和概率阈值 1.0。固定
views 源文件包含 100 epochs，程序不允许把不完整源 schedule 声称为正式
`--training-epochs 50`，所以按已有入口使用 `--epoch-limit 50`，严格训练前 50 个
source epochs，预期执行 650 次参数更新。该 checkpoint 会标记为 exploratory，后续
评测必须显式允许 exploratory checkpoint。

固定 launcher：

```text
compositional_intercode_bash/train_tapmem_eocmask_t1_e50_seed42.sh
```

输出目录：

```text
compositional_intercode_bash/runs/
  tapmem_llama8b_eocmask_t1_e50_seed42_v1/
```

2026-07-27 10:11 CST 使用物理 GPU 3，通过 `nohup + setsid` 启动后台训练。启动 PID
为 `2278707`；核验时 `PPID=1`、session ID 与 PID 相同、无 TTY，说明进程已经脱离
发起终端，终端断开不会终止训练。10:12 CST 模型已进入 GPU 训练阶段，占用约
17.6 GB 显存。PID、launcher 路径和日志位置同时保存在输出目录的
`background.pid`、`background_run.md` 和 `background.log` 中。

训练于 2026-07-27 10:16 CST 完成。实际执行 50 epochs、2600 次样本呈现和 650 次
参数更新，最后一条训练 loss 为 `0.7631701596577962`。后台进程正常退出，日志末尾
包含完整训练汇总，未发现 traceback、运行时错误或显存溢出。生成的
`trainable.safetensors` 只含 `procedure_embeddings`、`eoc_embedding` 和 routing
head 的权重、偏置，共 2,396,452 个可训练参数，没有保存冻结的 Llama 主干；其
SHA-256 为
`6d6549ac02797141ba51ad0e3a60ca1196a1595c5a3f85123285ad49e61b7464`。

## EOC 屏蔽版 50-epoch checkpoint 的 200 题后台评测

2026-07-27 10:21 CST 启动四分片真实环境评测。固定入口为：

```text
compositional_intercode_bash/evaluate_tapmem_eocmask_t1_e50_seed42_sharded.sh
```

输出目录为：

```text
compositional_intercode_bash/evaluations/
  tapmem_llama8b_eocmask_t1_e50_seed42_v1_official_agent_parse_preserve_full200_v1/
```

shard 0、1 放在物理 GPU 3，shard 2、3 放在物理 GPU 5。总控进程 PID 为
`2291008`，启动后 `PPID=1`、session ID 与 PID 相同，已经脱离发起终端。启动核验
确认四个 evaluator 都在运行，GPU 3 和 GPU 5 各加载两个模型进程，每个模型进程约占
26 GB 显存。

总控会等待四个 shard。失败的 shard 会使用同一编号和 `--resume` 串行恢复，然后
自动运行无 shard 的 `--resume`：补齐仍缺少的 episode，核对完整 200 题，再写
`summary_10_turn.json`。这一流程不改变 benchmark 的逐轮执行和评分逻辑，也不需要
人工监控或手动合并。分片日志写在 `logs/`，整体进度写在 `coordinator.log`。

本轮显式允许加载 exploratory checkpoint。由于训练只取固定 100-epoch 日程的前
50 epochs，而且使用 single-UID rootless Docker，结果按预期不能标记为论文正式
隔离环境结果；这不影响完整评测流程以及与同环境结果进行诊断性比较。

四个 shard 于 2026-07-27 11:05 CST 全部完成，第一次无 shard 严格合并于 11:06
CST 完成。输出目录包含 200 个 episode、四个完整 shard summary 和最终
`summary_10_turn.json`；最终文件记录 `evaluation_complete=true`，文件系统题量为
`60/53/60/27`。

结果为 54 / 200 题成功，成功率 27.00%，平均最高 reward 为 0.77310，平均执行
7.985 轮。逐文件系统成功题数为 fs1 9 / 60、fs2 23 / 53、fs3 15 / 60、fs4
7 / 27。没有 context overflow，也没有官方 parser 失败；11 题、25 轮 observation
经过统一长度裁剪。共有 210 个生成轮没有正常结束标志。memory-bank 约束触发 1724
次，其中 1045 次改变了原本会生成的 token。

最终 `summary_10_turn.json` 的 SHA-256 为
`fad6567d6994e7e80e06727f0ea322cc10d028730b4471a8a95b44c15ff7cc90`。
`paper_ready=false` 是由 50-epoch exploratory checkpoint 和 single-UID rootless
Docker 共同决定，并非评测未完成。

与已有 100-epoch TokMem 的 200 题结果直接比较，本次 50-epoch TapMem 成功率为
27.00%，TokMem 为 43.00%；平均最高 reward 分别为 0.77310 和 0.82725，平均轮数
分别为 7.985 和 6.575。逐题配对后，47 题两者都成功，107 题两者都失败，39 题只有
TokMem 成功，7 题只有 TapMem 成功；两侧精确 McNemar 检验
`p=1.8315e-6`。连续 reward 上 TokMem 更高 71 题、TapMem 更高 30 题、99 题相同，
两侧符号检验 `p=5.5331e-5`。

当前结果只能说明“这一个 50-epoch TapMem checkpoint 明显不如已有 100-epoch
TokMem checkpoint”。两者训练量不同：TokMem 有 5200 次样本呈现和 1300 次更新，
TapMem 只有 2600 次样本呈现和 650 次更新；评测 runner 也分别为 v12 和 v13。
v13 的代码差异只在 TapMem 的 EOC 状态约束分支，但严格比较仍应把 TapMem 训练到
100 epochs，并用 v13 对 TokMem 重跑同一批 200 题。

对 210 个“回答未正常结束”轮次逐个检查后，可以排除“代码误把回答结束 token
屏蔽掉”这一解释。EOC 是 `<|reserved_special_token_247|>`，回答结束标记是独立的
`<|eot_id|>`；v13 只在 procedure 外屏蔽前者。210 个异常轮次全部生成到 512-token
上限，207 轮进入过 procedure，199 轮在达到上限时仍没有用 EOC 退出 procedure。

204 / 210 个异常轮次触发过 memory-bank 约束，150 轮至少有一次由约束改变了原本的
token。触发约束的轮次有 16.25% 缺少 `<|eot_id|>`，未触发的轮次为 1.75%。这说明
异常与 TCRA 把生成送入 procedure 强相关；进入以后，当前 checkpoint 又经常不能
及时生成 EOC，于是命令片段和 memory token 循环到长度上限。

训练不足也有直接证据。50-epoch 模型只得到 4846 次 EOC 监督，旧 100-epoch
TapMem 为 9708 次；两者总体自回归 loss 分别为 5.058 和 3.604，routing loss
分别为 33.382 和 22.592。旧 100-epoch 模型只有 15 / 1386 个轮次用满 512 token。
不过旧模型使用 v12，仍允许 procedure 外 EOC，因此要严格区分训练轮数和 EOC 屏蔽
的作用，需要用同一组权重分别在 v12/v13 解码，或把当前 v13 TapMem 训练到 100
epochs 后再比较。

## 条件概率 routing loss 的 100-epoch TapMem 重跑

本轮不使用 routing head 零初始化，不改 `logit-bias-scale=1.0`，TCRA 继续把
routing correction 加到原始 memory-token logits 上。取消 procedure 外 EOC
屏蔽，并把 memory-bank 概率阈值保持为 `1.0`，因此不会用 hard gate 强制进入
memory bank。

routing 辅助损失改为 `fixed_count_posterior`。每个真实 routing 位置分别根据当前
atom 起点和剩余 procedure 数，计算固定段数条件下下一段各 procedure 的概率；不按
`sample_id + atom_start` 合并不同前缀，也不把候选平均分配。现有 5200 条 views
共有 9708 个 routing 位置，其中 3296 个位置存在两个或更多合法候选；当前被抽中
procedure 的平均条件概率为 0.83695，说明大多数位置仍接近 one-hot，真正模糊的
位置才会得到软监督。

正式设置使用 Llama-3.1-8B-Instruct、100 epochs、memory learning rate 0.005、
routing learning rate 0.001、route loss weight 0.1、batch size 1 和 gradient
accumulation 4。固定的训练后自动评测入口为：

```text
compositional_intercode_bash/
  run_tapmem_posteriorroute_t1_e100_seed42_pipeline.sh
```

训练输出和 200 题真实环境评测输出分别为：

```text
compositional_intercode_bash/runs/
  tapmem_llama8b_posteriorroute_t1_e100_seed42_v1/
compositional_intercode_bash/evaluations/
  tapmem_llama8b_posteriorroute_t1_e100_seed42_v1_official_agent_parse_preserve_full200_v1/
```

2026-07-27 12:00 CST 已用 `nohup + setsid` 启动上述流水线，总控 PID 为
`3533768`，训练子进程 PID 为 `3533772`。核验时总控 `PPID=1`、没有 TTY，说明终端
断开不会终止任务；训练进程已在物理 GPU 1 占用约 17.9 GB 显存并进入计算。训练成功
后脚本会自动在物理 GPU 1、3、5、6 各启动一个评测分片，不需要人工继续操作或持续
监控。

该流水线于 2026-07-27 12:41 CST 完成全部 200 题。条件概率 routing loss 的
TapMem 成功 75 / 200 题，成功率 37.50%，平均最高 reward 为 0.81820，平均执行
6.93 轮；各文件系统为 fs1 19 / 60、fs2 20 / 53、fs3 28 / 60、fs4 8 / 27。
没有 parser 失败和 context overflow，共 12 个生成轮缺少正常结束标志。

与旧 100-epoch TapMem 相比，成功题数从 76 变为 75，但平均 reward 从 0.80730
提高到 0.81820；与 100-epoch TokMem 相比仍少 11 道成功题。阈值虽然设为 `1.0`，
hard memory-bank constraint 仍因浮点概率取整触发 362 次并改变 144 个 token，
因此该设置不能视为真正关闭 hard gate。

## 纠错型 routing loss 与纯加性 TCRA 重跑

下一轮继续复用相同 5200 条 views，不重新生成数据。只有 `boundary_sample` 按固定
段数条件概率累计到 90% 构造合理 procedure 集合；coverage/reference anchor 使用
当前 gold。9708 个 routing 位置中，2477 个位置包含多个合理 procedure：2474 个
集合大小为 2，3 个集合大小为 3。

routing loss 直接使用“原始 memory logits + TCRA correction”。原始 memory top-1
错误时使用 margin 0.5 的纠错损失，原始 top-1 已合理时使用权重 0.25 的排序保持
损失。AR 前向保留加性 correction，但隔离其对 routing head 的梯度。routing head
仍使用原有初始化，`logit-bias-scale=1.0`，routing LR 为 0.001，route loss
weight 为 0.1，共训练 100 epochs。

以上是当时实际运行的历史设置。当前代码已经删除排序保持 loss，也不再隔离 routing
head 的 AR 梯度；训练和推理统一改为在 procedure 外、原始全词表 top-1 为 memory
token 时才加入 TCRA bias。因此下面的旧结果仍不能直接代表当前实现。

本轮使用 `--disable-memory-bank-constraint` 显式关闭 hard gate，推理时只有原始
top-1 已是 memory token 才应用加性 TCRA。训练后自动评测入口为：

```text
compositional_intercode_bash/
  run_tapmem_residualroute_additive_e100_seed42_pipeline.sh
```

输出目录为：

```text
compositional_intercode_bash/runs/
  tapmem_llama8b_residualroute_additive_e100_seed42_v1/
compositional_intercode_bash/evaluations/
  tapmem_llama8b_residualroute_additive_e100_seed42_v1_official_agent_parse_preserve_full200_v1/
```

2026-07-27 13:16 CST 已用 `nohup + setsid` 启动流水线，总控 PID 为 `4014982`，
训练子进程 PID 为 `4014985`。启动核验时总控 `PPID=1`、没有 TTY；训练进程已在
物理 GPU 1 占用约 17.6 GB 显存并进入实际计算。训练完成后会自动转入 GPU
1、3、5、6 的四分片真实环境评测。
