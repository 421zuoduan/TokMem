# mixed_v1 训练快照

该快照按 2026-07-26 的实验决定，将旧 `smoke_v5` 的 14 条 train trajectory
与 `formal_attempt_1` 的 52 条完整 trajectory 合并。这里不按 collector 的
`accepted` 或 deterministic evaluator 结果过滤；只要求轨迹具有完整的
call/observation 对、真实执行元数据并以 `claim_done` 结束。

## 规模

- trajectory：66（48 accepted，18 rejected）
- evaluator：53 pass，13 fail
- next-tool-call step：460
- 含失败 observation 的 trajectory：9
- 失败 observation：11
- target tools：38/38
- 每个 target tool 的成功 train trajectory 覆盖：6–25 条

失败 observation 对应的 target step 也按本次实验口径进入监督，但每条 step
显式保留：

- `episode_accepted`
- `episode_evaluator_passed`
- `episode_rejection_reasons`
- `target_call_success`

因此后续可以在不重新解析原始轨迹的情况下进行 clean-only 消融。

## 冻结文件

- `episodes/train_all.jsonl`
  - SHA-256: `776d9f59c70ecb9047b4e097888982d61484c09fb6978ef83d82364490aec0ee`
- `tasks/train_all.jsonl`
  - SHA-256: `0c4b66f8aa694733b3cdc3d14d65f4adebae5676b84a4a1222c1e6f4c22c14b4`
- `steps/train.jsonl`
  - SHA-256: `c3a3b4d34263c21991db36be61df82163963dd86aeac71df68afb9a99af2fcf7`
- `coverage/train_all.json`
  - SHA-256: `d521c9f1b55038a4a40717c5d01291322a507bf766bfad02c2139fe601b05bdb`
- `provenance/semantic_leakage.json`
  - SHA-256: `ff3f0a262590ef783610ce2bfebdc2102e77a4c8c3c122023b3ee397e80a1dad`

## 转换口径

```bash
python -m compositional_toolathlon.episode_to_steps \
  --episodes compositional_toolathlon/data/mixed_v1/episodes/train_all.jsonl \
  --output-dir compositional_toolathlon/data/mixed_v1/steps \
  --include-rejected \
  --include-failed-calls
```

TokMem 与 TapMem 必须读取同一个 `steps/train.jsonl` 和
`coverage/train_all.json`。训练 checkpoint 使用 `trainable_only` 格式，不保存
冻结的 Llama backbone、optimizer 或 scheduler。
