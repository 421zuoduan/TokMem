# Generated data

`generated/` 由工具发现、task 生成、teacher 执行和 episode-to-step 转换产生，不纳入
源码提交。建议布局：

```text
generated/
├── manifests/
├── tasks/
├── episodes/
├── steps/
├── coverage/
├── predictions/
├── provenance/
├── runtime/
└── leakage_audit/
```

只有真实执行、deterministic evaluator Pass、无受保护 benchmark 内容且 provenance
完整的 episode 才能进入 `episodes/accepted`。

`manifests/usable_tools.json` 记录至少一次真实 smoke 成功的工具；coverage gate 应
读取单独冻结的 `manifests/target_tools.json`，不能把环境辅助工具自动加入分母。
当前 `smoke_v5` 的 usable 集为完整 47 工具，38 个目标来自 accepted episode，其余
8 个非终止干扰工具来自独立 fresh-gateway probe，`claim_done` 来自每条 episode。
每个 task 的可用菜单同样保留全部 47 工具，但只对冻结的 38 个目标检查 train 中
至少 3 个不同成功 episode。
当前 `smoke_v5` 为 14 个 train、0 个 validation、2 个 synthetic-test episode，
对应 142/0/20 个 step；空的 validation 文件只作为格式兼容占位。
训练单位是 next-tool-call step：每个 epoch 对 142 条 train step 无放回打乱并各
训练一次，validation 占位文件不会传给训练入口。
`coverage/smoke.json` 还绑定真实执行要求、干扰项要求、各 split 的 episode ID、
step 数量和 canonical 内容哈希；`steps/metadata.json` 中的数量与哈希必须完全一致。
`runtime/<smoke_id>/` 保留每题的 fresh workspace、gateway 配置、日志和 checkpoint，
不作为训练输入。

`mixed_v1/` 是 2026-07-26 冻结的扩展训练条件：旧 14 条 train trajectory 加
52 条 observation-conditioned trajectory，共 66 条轨迹和 460 个 step。按本次
实验决定，它显式使用 `--include-rejected --include-failed-calls`，因此与只接收
clean accepted episode 的主协议不同；每个 step 保留 collector、evaluator 和
调用成功状态，详见 `mixed_v1/README.md`。
