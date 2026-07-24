# Rebuttal 实验记录

## Compositional Table 1 多随机种子实验

本实验用于补充 TapMem 论文 Table 1 的随机种子误差条。实验覆盖
Llama-3.2-1B、Llama-3.2-3B 和 Llama-3.1-8B，以及 ICL、RAG、TokMem、
TapMem、Fine-Tuning/LoRA、TokMem with adaptation 和 TapMem with
adaptation，共 21 组设置。

| 项目 | 设置 |
| --- | --- |
| 新增随机种子 | 40、41 |
| 主表参考随机种子 | 42 |
| 数据 | tools 51-100；5000 条训练样本；500 条测试样本；2-4 calls |
| 序列长度 | 所有可训练方法均为 512，包括 3B TokMem |
| TokMem/TapMem 学习率 | embedding/TCRA 为 `5e-3` |
| Fine-Tuning/LoRA 学习率 | 1B/3B 为 `5e-5`，8B 为 `8e-5` |
| TokMem adaptation LoRA 学习率 | 1B/3B 为 `5e-5`，8B 为 `8e-5` |
| TapMem adaptation LoRA 学习率 | 全部模型为 `8e-5` |
| 误差条 | 三个 seed 等权计算 `mean ± sample standard deviation` |

除随机种子和明确要求统一为 512 的 3B TokMem 序列长度外，主队列的模型、
数据、batch size、训练轮次、LoRA 参数和评测设置均与论文主表实验保持
一致。TapMem 使用
`--use_eoc --use_logit_bias --use_logit_train_add --detach`：TCRA 接收
AR loss 和 routing loss，但 TCRA 路径的梯度不回传 embedding；
embedding 仍通过普通 AR 主路径训练。

实验启动脚本为
`scripts/compositional/run_paper_table1_seeds40_41.sh`。它运行 42 个
seed40/41 主任务，并额外运行一个 3B TokMem seed42、`max_length=512`
的校准任务，以避免将主表中 1024 长度的结果混入该行的 seed error
bar。完成后由
`scripts/compositional/summarize_paper_table1_seeds.py` 生成
`table1_error_bars.json` 和 `table1_error_bars.md`。

启动脚本采用空闲 GPU 任务队列：仅调度 `GPU_IDS` 指定的显卡，默认每
30 秒检查一次；当 `nvidia-smi` 报告显存占用不超过 2048 MiB 时，该卡
从 FIFO 队列领取下一个任务。调度器每轮最多串行启动一个程序；已经启动
的程序可以在不同 GPU 上并行运行。不同 GPU 之间没有 wave 同步等待，
直至全部任务均被执行。任务运行期间还会持有
`/tmp/tokmem_gpu_locks/gpu_<id>.lock` 的每卡 `flock`，避免使用相同锁
约定的多个 launcher 同时抢占一张显卡。`GPU_IDS` 只接受 `0,2,5`
这类规范十进制 index。该调度器不管理复杂的 worker 进程组；正式运行时
应保持父 launcher 存活，直至整个实验队列结束。

为利用已占用 GPU 的剩余显存，1B/3B 的 ICL 和 RAG 共 8 个任务单独组成
inference 队列；其余 35 个任务组成主队列。Inference 队列要求至少
16000 MiB 空余显存，并将 1B ICL/RAG batch 调为 `8/32`、3B 调为
`4/16`；模型、数据、seed 和 greedy decoding 设置不变。两个队列共享
同一个 suite，全部 43 个任务完成后统一汇总。

正式 suite
`rebuttal_compositional_table1_seeds40_41_20260724_100205` 已完成
43/43 个任务，失败数为 0，并生成 `table1_error_bars.json` 和
`table1_error_bars.md`。其中主队列的 35 个任务以及 3B TokMem 的
seed42、512 长度校准任务均已完成。

需要注意，1B/3B ICL 和 RAG 的 8 个任务为了在已占用 GPU 上运行，使用
了低于主表的 evaluation batch。复核发现低 batch 与主表 full batch 的
原始预测并不完全一致，因此这 8 个结果仅作为资源共享运行的诊断结果，
不能与主表 seed42 直接组成严格的 seed-only error bar。论文最终误差条
需要用主表 batch（1B ICL/RAG 为 `64/256`，3B 为 `32/192`）重跑这
8 个任务后重新汇总。

此外，除 3B TokMem 外，seed42 使用论文主表中经过舍入、部分由同 seed
重复实验聚合的参考值，因此即使完成上述重跑，最终结果也应表述为
**published-reference error bar**，而不是完全均衡的三随机种子实验。
