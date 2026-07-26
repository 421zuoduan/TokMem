# Local runs

本目录保存未归档的 smoke、pilot、训练和 rollout 输出。成功正式运行按仓库规范复制
到 `results/<run_name>/`，并补充中文 `run_summary.md`。

`mixed_v1_llama31_8b_{tokmem,tapmem}_seed42_e10/` 是 2026-07-26 完成的
66-trajectory、460-step 训练。归档副本位于
`results/compositional/toolathlon_mixed_v1_llama8b_e10_20260726/`；两组
checkpoint 均只含 trainable memory/TCRA 增量，不含 frozen backbone。
