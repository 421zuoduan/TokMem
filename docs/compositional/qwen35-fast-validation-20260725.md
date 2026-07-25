# Qwen3.5 fast-environment validation (2026-07-25)

The validation suite was run with `tokmem-qwen35` on four A100 GPUs:

```bash
bash scripts/compositional/qwen35/validation/run_validations.sh 4 5 6 7
```

The runner executes the causal-conv1d check and TapMem optimizer-step smoke
sequentially on GPU 4, while the 9B fast-path, 4B fast-path, and 9B
fast-versus-fallback checks run concurrently on GPUs 5, 6, and 7. Raw logs are
retained under
`results/compositional/qwen35_environment_artifacts/validation/20260725/`.

All four workers exited successfully:

- causal-conv1d forward and update maximum absolute error: `0.03125`; state
  error: `0`; backward gradients finite.
- Qwen3.5-9B: 24 linear-attention and 8 full-attention layers; prefill used
  causal-conv1d, chunk gated-delta, and fused norm 24 times each; cached decode
  used causal-conv1d update, recurrent gated-delta, and fused norm 24 times
  each; peak allocated memory `17,352.5 MiB`.
- Qwen3.5-4B: identical dispatch counts and layer split; peak allocated memory
  `8,392.0 MiB`.
- 9B fast versus fallback: finite logits, identical top-1 token (`198`),
  cosine `0.9999660254`, mean/max absolute difference `0.0181984/0.125`, and
  relative CE difference `0.00880621`.
- One real four-call 9B TapMem batch completed forward, backward, and AdamW
  update. Only 413,746 parameters (`0.0046%`) were trainable; the base LM had
  zero trainable parameters and zero gradients. Memory/head gradient norms and
  update deltas were finite, and peak allocated/reserved memory was
  `17.530/17.869 GiB`.

Relevant input hashes:

```text
d0883072e01861ed0b2d47be3c16c36a8e81c224c7ffaa310c6558fb3f932b05  Qwen3.5-9B/config.json
26d3539b516be613f39563617cb9d33b3f83d401298125be392c80cefb8f7fe5  Qwen3.5-9B/model.safetensors.index.json
316230d6a809701f4db5ea8f8fc862bc3a6f3229c937c174e674ff3ca0a64ac8  Qwen3.5-9B/tokenizer_config.json
ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670  Qwen3.5-4B/config.json
cf3f798ee02ba45f9622aa8892a47369ab667d0afbf154ee7c2212de42e6302d  Qwen3.5-4B/model.safetensors.index.json
316230d6a809701f4db5ea8f8fc862bc3a6f3229c937c174e674ff3ca0a64ac8  Qwen3.5-4B/tokenizer_config.json
37846645e82eb1f07afb2b302b58b4567eba72cb2234dd1f304d26dcfb71dbea  function_calling_train_tools51-100_4calls.json
3eac9a070384e5dda0afae6bccadec5dc9ce72d38ade2eb53ed28bbd0352f392  function_calling_test_tools51-100_4calls.json
```

The suite launcher independently records source/data/model-config checksums and
the complete runtime fingerprint in each formal suite directory.
