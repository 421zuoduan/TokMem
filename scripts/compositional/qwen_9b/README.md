# Qwen3.5-9B compositional launcher

`tokmem_eoc_logit_bias_qwen_9b.sh` runs the maintained `main_sequential.py`
TapMem path on `models/Qwen3.5-9B`.

The Qwen3.5 wrapper keeps the existing tool/EOC targets, logit-bias behavior,
training loop, checkpoint layout, and evaluation semantics. It maps the
existing conversation roles onto the native Qwen3.5 chat template with
thinking disabled, and uses `<|im_end|>` consistently as the supervised and
decoded response ending. Existing non-Qwen backbones retain their original
prompt and generation behavior.

Defaults:

- tools: `51-100`
- maximum calls: `4`
- epochs: `3`
- training batch size: `1`
- evaluation batch size: `4`
- maximum sequence length: `512`
- dtype: `bfloat16`

The local model path and capacity settings can be overridden without editing
the script:

```bash
TOKMEM_GPU=0 \
TOKMEM_MODEL_PATH=/path/to/Qwen3.5-9B \
TOKMEM_BATCH_SIZE=1 \
TOKMEM_EVAL_BATCH_SIZE=4 \
bash scripts/compositional/qwen_9b/tokmem_eoc_logit_bias_qwen_9b.sh
```

Qwen3.5 requires a Transformers release that recognizes `qwen3_5`; the
repository's `tokmem` environment is currently validated with Transformers
5.3.0.
