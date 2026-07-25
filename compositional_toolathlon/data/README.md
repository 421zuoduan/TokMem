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
└── leakage_audit/
```

只有真实执行、deterministic evaluator Pass、无受保护 benchmark 内容且 provenance
完整的 episode 才能进入 `episodes/accepted`。
