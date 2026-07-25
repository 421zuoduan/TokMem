# Qwen3.5 fast-kernel environment

`tokmem-qwen35` is an isolated clone of `tokmem` for Qwen3.5. The original
environment is unchanged. The earlier fallback suite
`qwen35_table1_rebuttal_20260725_v3` was stopped on 2026-07-25; its completed
sweep and partial artifacts remain archived under `results/compositional/`.

## Validated stack

| Component | Version |
| --- | --- |
| Python | 3.10.0 plus the tracked `inspect.BlockFinder` compatibility patch |
| PyTorch | 2.7.1+cu128 |
| torchvision | 0.22.1+cu128 |
| Triton | 3.3.1 |
| Transformers | 5.3.0 |
| fla-core | 0.5.1 |
| flash-linear-attention | 0.5.1 |
| causal-conv1d | 1.6.2.post1, local SM80 build |
| einops | 0.8.1 |

The stack targets the eight local A100 SM80 GPUs and driver 570.133.07. The
PyTorch wheel supplies the CUDA 12.8 runtime. The host system has glibc 2.31
and its default `nvcc` is CUDA 11.6.

The official causal-conv1d Torch-2.7 wheel cannot be used on this host: its
extension directly requires `GLIBC_2.32`. The installed wheel was therefore
built from the official `v1.6.2.post1` source with the cached CUDA 12.8
toolchain under `/home/shilong/anaconda3/envs/draftOPRD`, CXX11 ABI enabled,
and only `sm_80` generated. Its SHA-256 is
`8776fc06b41772b62e824bad15edd50d30ea8fdb19079b1f6c5626d7cd09abdb`;
the resulting extension requires at most GLIBC 2.14.

Python 3.10.0 has an `inspect.BlockFinder` bug that truncates Triton functions
when a decorator contains lambdas together with nested calls or parentheses.
The compatibility module at
`scripts/compositional/qwen35/python310_compat/tokmem_qwen35_inspect_patch.py`
backports only the corrected Python 3.11 token-eater method. It is loaded by a
dedicated `.pth` file and does not replace a general `sitecustomize.py`. It is
automatically a no-op on a Python 3.10 patch release where the buggy
`decoratorhasargs` implementation is absent. It changes source inspection
only; the Triton source passed to the JIT is byte-identical to that produced
by Python 3.10.20.

Install and verify the compatibility module with the explicit target-environment
installer:

```bash
bash scripts/compositional/qwen35/python310_compat/install.sh
```

When reconstructing the environment, remove the cloned `torch/`, `torchgen/`,
`functorch/`, and `triton/` namespaces after uninstalling their old packages,
then install the pinned wheels into clean directories. A normal force reinstall
does not remove files outside a wheel's `RECORD`; stale Torch and Triton files
were the cause of the initial mixed-version import failure. Install
`fla-core==0.5.1` and `flash-linear-attention==0.5.1` without allowing pip to
replace the pinned Torch/Triton stack.

The numerical-path package constraints are
`scripts/compositional/qwen35/environment/critical-package-constraints.txt`;
known source and wheel checksums are in the adjacent
`known-artifacts.sha256`. The constraints intentionally describe the critical
stack rather than claiming to be a complete transitive lock. The custom
causal-conv1d wheel and its source archive are persisted locally under
`results/compositional/qwen35_environment_artifacts/`. The build is reproducible
from the pinned source archive and SM80 patch with:

```bash
bash scripts/compositional/qwen35/environment/build_causal_conv1d_sm80.sh
```

The build script uses Python and Torch from `tokmem-qwen35` while pointing
`CUDA_HOME` at the CUDA 12.8 toolkit in `draftOPRD`. It validates the source
checksum, Python/Torch/CUDA/CXX11-ABI versions, wheel tag, three SM80 cubins,
and GLIBC compatibility before copying the rebuilt wheel to `rebuilt-wheels/`.
It never overwrites the validated golden wheel. Public Torch, Triton, cuDNN,
and NCCL wheels may be redownloaded and must match
`known-artifacts.sha256`; the locally compiled golden causal-conv1d wheel is
retained in the persistent artifact directory.

Compiler output embeds the temporary build path, so a rebuilt wheel is not
expected to have the golden wheel's byte-for-byte SHA-256. Treat a rebuild as
a recovery artifact: install it only after rerunning the binary and numerical
validations below, which will also produce a new suite environment fingerprint.

Install that wheel only into the explicit Qwen3.5 environment:

```bash
/home/shilong/anaconda3/envs/tokmem-qwen35/bin/python -m pip install \
    --no-deps \
    results/compositional/qwen35_environment_artifacts/wheels/causal_conv1d-1.6.2.post1-cp310-cp310-linux_x86_64.whl
```

## Completed validation

The reusable validators, exact command, input hashes, and 2026-07-25 output
summary are recorded in
`docs/compositional/qwen35-fast-validation-20260725.md`.

- `python -m pip check`: no broken requirements.
- CUDA: Torch reports 12.8, CXX11 ABI is enabled, and the A100 is available.
- Transformers reports `is_fast_path_available=True`.
- Direct causal-conv1d prefill/backward/update passed against the PyTorch
  reference; maximum absolute error was `0.03125`.
- Qwen3.5-9B and 4B each expose 24 linear-attention plus 8 full-attention
  layers. Prefill called `causal_conv1d_fn` and `chunk_gated_delta_rule` 24
  times; cached decode called `causal_conv1d_update` and
  `fused_recurrent_gated_delta_rule` 24 times. Fused gated RMSNorm ran in all
  24 linear layers and logits were finite.
- Peak allocated model-validation memory was 17,352.5 MiB for 9B and
  8,296.0 MiB for 4B.
- The 9B fast/fallback last-token logits had cosine `0.999966`, mean absolute
  difference `0.01820`, identical top-1, and relative CE difference `0.00881`.
- A real 9B TapMem four-call batch completed forward, backward, and AdamW
  update. The base LM had no gradients; memory-token and logit-bias gradients
  and parameter deltas were finite. Peak allocated memory was 17.53 GiB.
- Synthetic IDs remain tools `248077..248126`, EOC `248127`, native pad
  `248044`, and native EOS `248046`.

## Launcher

The Qwen3.5 suite accepts `--conda-env`. Its default remains `tokmem` so older
invocations retain their behavior; formal fast-path runs must pass
`--conda-env tokmem-qwen35`. The selected environment, package manifest, and
critical implementation hashes are saved in `suite_config.txt`. A suite cannot
be resumed under a different environment or after the selected environment is
modified.

```bash
bash scripts/compositional/qwen35/run_qwen35_table1_rebuttal.sh \
    --suite-name qwen35_table1_rebuttal_20260725_fast \
    --gpus 0,1,2,3,4,5,6,7 \
    --conda-env tokmem-qwen35
```
