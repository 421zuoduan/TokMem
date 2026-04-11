# Compositional EOC + Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a baseline-compatible `eoc + gate` training and decoding path to `compositional/`, keep original TokMem behavior behind default-off flags, and raise the default `max_length` to `1024`.

**Architecture:** Extend the existing native-tool-token pipeline rather than creating a second training stack. `dataset.py` remains responsible for sequence construction, `model.py` owns reserved-token parameterization and gated decoding, `training.py` owns loss composition and parsing-aware evaluation, and `main_sequential.py` wires the new configuration into the round-based training loop.

**Tech Stack:** Python, PyTorch, Hugging Face Transformers, existing `compositional/` training pipeline

---

## File Structure

- Modify: `compositional/main_sequential.py`
  Add CLI flags for `use_eoc`, `use_gate`, loss weights, gate threshold, and change default `max_length` to `1024`.

- Modify: `compositional/model.py`
  Reserve one extra special token for `eoc`, extend trainable reserved-token rows, add gate MLP, expose hidden states when needed, implement gated decoding, and update sequence parsing to understand `eoc`.

- Modify: `compositional/dataset.py`
  Build `eoc`-augmented targets when requested and keep truncation semantics stable.

- Modify: `compositional/training.py`
  Add auxiliary-mask construction on truncated sequences, compute `L_ar/L_eoc/L_tool/L_gate`, expose richer metrics, and route eval generation through the new gated decoding path.

- Modify: `compositional/README.md`
  Document the three experiment modes and the new CLI knobs.

- Test/validation entrypoints:
  - `python -m py_compile compositional/dataset.py compositional/model.py compositional/training.py compositional/main_sequential.py`
  - Targeted dataset/model sanity scripts run inline from the repo root with the `tokmem` environment

### Task 1: Add Configuration Surface

**Files:**
- Modify: `compositional/main_sequential.py`
- Test: inline CLI parse smoke check via `python compositional/main_sequential.py --help`

- [ ] **Step 1: Write the failing test expectation**

Expected new CLI behavior:

- `--use_eoc` defaults to `False`
- `--use_gate` defaults to `False`
- `--max_length` defaults to `1024`
- `--gate_threshold` defaults to `0.5`
- `--use_gate` without `--use_eoc` raises an argument/config error

- [ ] **Step 2: Run the current CLI help to verify the new flags are absent**

Run:

```bash
python compositional/main_sequential.py --help
```

Expected:

- No `use_eoc` / `use_gate` flags yet
- `max_length` help still shows default `512`

- [ ] **Step 3: Implement minimal CLI additions**

Add parser arguments and an early configuration validation block in `main_sequential.py`.

Code shape:

```python
parser.add_argument("--use_eoc", action="store_true", help="Insert and supervise an explicit end-of-control token")
parser.add_argument("--use_gate", action="store_true", help="Enable gate loss in training and gate-controlled decoding in generation")
parser.add_argument("--eoc_loss_weight", type=float, default=0.1, help="Weight for the eoc boundary loss")
parser.add_argument("--tool_loss_weight", type=float, default=0.1, help="Weight for the tool-only selection loss")
parser.add_argument("--gate_loss_weight", type=float, default=0.1, help="Weight for the gate BCE loss")
parser.add_argument("--gate_threshold", type=float, default=0.5, help="Sigmoid threshold for positive gate decisions during decoding")
parser.add_argument("--max_length", type=int, default=1024, help="Maximum sequence length")
```

Validation shape:

```python
if args.use_gate and not args.use_eoc:
    parser.error("--use_gate requires --use_eoc")
```

- [ ] **Step 4: Thread the new args into model / dataloader / training / eval calls**

Pass the new booleans and weights explicitly instead of reading globals.

- [ ] **Step 5: Run the CLI check again**

Run:

```bash
python compositional/main_sequential.py --help
```

Expected:

- New flags are present
- `max_length` default is `1024`

### Task 2: Add `eoc` Token Support to the Model

**Files:**
- Modify: `compositional/model.py`
- Test: inline model-construction sanity check

- [ ] **Step 1: Write the failing test expectation**

Expected new model behavior:

- Tool tokens still occupy the first `num_tools` reserved slots
- One additional reserved slot is assigned to `eoc` when `use_eoc=True`
- The model exposes `eoc_token_id`
- Trainable reserved-token rows include `eoc`

- [ ] **Step 2: Verify current model state is insufficient**

Run:

```bash
python - <<'PY'
from transformers import AutoTokenizer
from compositional.model import FunctionCallingModel
tok = AutoTokenizer.from_pretrained("models/Llama-3.2-3B-Instruct", local_files_only=True)
model = FunctionCallingModel(model_name="models/Llama-3.2-3B-Instruct", num_tools=3, tokenizer=tok, device="cpu", dtype=None)
print(hasattr(model, "eoc_token_id"))
PY
```

Expected:

- `False`, or construction fails because no `eoc` support exists

- [ ] **Step 3: Extend model initialization for `use_eoc`**

Add constructor parameters:

```python
def __init__(..., use_eoc=False, use_gate=False, gate_threshold=0.5):
```

Reserve `num_tools + 1` rows when `use_eoc=True` and keep the first `num_tools` rows mapped to tools.

Code shape:

```python
self.num_reserved_slots = self.num_tools + (1 if use_eoc else 0)
self.reserved_token_names = [token for token, _ in sorted_reserved[:self.num_reserved_slots]]
self.reserved_token_ids = [token_id for _, token_id in sorted_reserved[:self.num_reserved_slots]]
self.tool_reserved_token_ids = self.reserved_token_ids[:self.num_tools]
self.eoc_token_id = self.reserved_token_ids[self.num_tools] if use_eoc else None
```

- [ ] **Step 4: Keep tool mappings stable**

Ensure tool mappings are built from `self.tool_reserved_token_ids`, not the full reserved list.

Code shape:

```python
self.tool_id_to_token_id = {i: self.tool_reserved_token_ids[i] for i in range(self.num_tools)}
self.token_id_to_tool_id = {self.tool_reserved_token_ids[i]: i for i in range(self.num_tools)}
```

- [ ] **Step 5: Extend trainable input/output row replacement to include `eoc`**

The overridden embedding and lm_head replacement loops should iterate over all trainable reserved rows, not just tool rows.

- [ ] **Step 6: Add gate MLP and helper methods**

Add a two-layer MLP only when `use_gate=True`.

Code shape:

```python
self.gate_mlp = nn.Sequential(
    nn.Linear(self.config.hidden_size, self.config.hidden_size),
    nn.GELU(),
    nn.Linear(self.config.hidden_size, 1),
)
```

Also add helpers:

- `get_eoc_token_id()`
- `is_tool_token_id(token_id)`
- `mask_logits_to_tool_tokens(logits)`

- [ ] **Step 7: Run the model sanity check**

Run:

```bash
python - <<'PY'
from transformers import AutoTokenizer
import torch
from compositional.model import FunctionCallingModel
tok = AutoTokenizer.from_pretrained("models/Llama-3.2-3B-Instruct", local_files_only=True)
model = FunctionCallingModel(model_name="models/Llama-3.2-3B-Instruct", num_tools=3, tokenizer=tok, device="cpu", dtype=torch.float32, use_eoc=True, use_gate=True)
print(model.eoc_token_id is not None)
print(len(model.tool_id_to_token_id), model.num_tools)
print(model.gate_mlp is not None)
PY
```

Expected:

- `True`
- Matching tool mapping lengths
- `True`

### Task 3: Add `eoc`-Aware Dataset Construction

**Files:**
- Modify: `compositional/dataset.py`
- Test: inline sample rendering sanity check

- [ ] **Step 1: Write the failing test expectation**

Expected new dataset behavior:

- `use_eoc=False` keeps the old target format
- `use_eoc=True` inserts one `eoc` after every tool-controlled JSON span
- Truncation still happens after the final sequence is built

- [ ] **Step 2: Verify the current dataset emits no `eoc`**

Run:

```bash
python - <<'PY'
import json
from transformers import AutoTokenizer
from compositional.model import FunctionCallingModel
from compositional.dataset import NativeFunctionCallingDataset
tok = AutoTokenizer.from_pretrained("models/Llama-3.2-3B-Instruct", local_files_only=True)
model = FunctionCallingModel(model_name="models/Llama-3.2-3B-Instruct", num_tools=100, tokenizer=tok, device="cpu", dtype=None)
ds = NativeFunctionCallingDataset("compositional/data/training/function_calling_train_tools1-50_4calls.json", tok, 1024, model)
item = ds[0]
print(item["input_ids"][-10:].tolist())
PY
```

Expected:

- No dedicated `eoc` token can appear because the dataset does not know about it

- [ ] **Step 3: Add `use_eoc` support to the dataset**

Pass the flag and append `model.eoc_token_id` after each JSON span when enabled.

Code shape:

```python
if self.use_eoc:
    full_sequence.append(self.model.eoc_token_id)
    labels.append(self.model.eoc_token_id)
```

- [ ] **Step 4: Keep eval mode prompt-only behavior unchanged**

Do not inject `eoc` into eval-mode prompts.

- [ ] **Step 5: Run a dataset sanity check**

Run:

```bash
python - <<'PY'
from transformers import AutoTokenizer
import torch
from compositional.model import FunctionCallingModel
from compositional.dataset import NativeFunctionCallingDataset
tok = AutoTokenizer.from_pretrained("models/Llama-3.2-3B-Instruct", local_files_only=True)
model = FunctionCallingModel(model_name="models/Llama-3.2-3B-Instruct", num_tools=100, tokenizer=tok, device="cpu", dtype=torch.float32, use_eoc=True)
ds = NativeFunctionCallingDataset("compositional/data/training/function_calling_train_tools1-50_4calls.json", tok, 1024, model, mode="train", use_eoc=True)
item = ds[0]
print((item["labels"] == model.eoc_token_id).sum().item())
print(item["raw_data"]["tools"])
PY
```

Expected:

- The number of `eoc` labels equals the number of tools in the raw sample

### Task 4: Add Auxiliary Target Construction and Loss Computation

**Files:**
- Modify: `compositional/training.py`
- Modify: `compositional/model.py`
- Test: inline one-batch loss sanity check

- [ ] **Step 1: Write the failing test expectation**

Expected new training behavior:

- Baseline mode computes only `L_ar`
- `eoc` mode adds `L_eoc` and `L_tool`
- Full mode adds `L_gate`
- Auxiliary targets are derived after truncation from the actual batch tensors

- [ ] **Step 2: Verify the current trainer only exposes one optimization loss**

Inspect or run one batch and confirm only the main CE is used.

- [ ] **Step 3: Refactor model forward to expose hidden states on demand**

Change `FunctionCallingModel.forward(...)` to optionally return the full HF output object or `(logits, hidden_states)` when requested.

Code shape:

```python
def forward(self, input_ids, attention_mask, output_hidden_states=False, return_dict=False):
    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=output_hidden_states,
        return_dict=True,
    )
    if return_dict or output_hidden_states:
        return outputs
    return outputs.logits
```

- [ ] **Step 4: Add helper functions in `training.py`**

Implement focused helpers:

- `build_shift_supervision_masks(labels, model, use_eoc, use_gate)`
- `gather_gate_examples(hidden_states, labels, model)`
- `compute_tool_subset_targets(shift_labels, model)`

- [ ] **Step 5: Compute all four losses with mode-aware gating**

Core shape:

```python
loss = ar_loss
if use_eoc:
    loss = loss + eoc_loss_weight * eoc_loss
    loss = loss + tool_loss_weight * tool_loss
if use_gate:
    loss = loss + gate_loss_weight * gate_loss
```

Use:

- `F.cross_entropy` for `L_ar`
- `F.cross_entropy` for `L_eoc`
- `F.cross_entropy` on the tool subset for `L_tool`
- `F.binary_cross_entropy_with_logits` for `L_gate`

- [ ] **Step 6: Add batch logging for the auxiliary losses**

Track:

- `ar_loss`
- `eoc_loss`
- `tool_loss`
- `gate_loss`
- counts of `eoc` sites, tool sites, gate sites

- [ ] **Step 7: Run a one-batch sanity script**

Run:

```bash
python - <<'PY'
from transformers import AutoTokenizer
import torch
from compositional.model import FunctionCallingModel
from compositional.dataset import create_native_dataloader
from compositional.training import train_native_function_calling_model
tok = AutoTokenizer.from_pretrained("models/Llama-3.2-3B-Instruct", local_files_only=True)
model = FunctionCallingModel(model_name="models/Llama-3.2-3B-Instruct", num_tools=100, tokenizer=tok, device="cpu", dtype=torch.float32, use_eoc=True, use_gate=True)
train_dl, _, _, _, _ = create_native_dataloader(
    model=model,
    train_data_path="compositional/data/training/function_calling_train_tools1-50_4calls.json",
    test_data_path="compositional/data/test/function_calling_test_tools1-50_4calls.json",
    tokenizer=tok,
    batch_size=1,
    max_length=1024,
    eval_batch_size=1,
    validation_split=0,
)
batch = next(iter(train_dl))
print(batch["input_ids"].shape, batch["labels"].shape)
PY
```

Expected:

- Batch builds successfully with the new mode enabled

### Task 5: Add Gate-Controlled Decoding

**Files:**
- Modify: `compositional/model.py`
- Modify: `compositional/training.py`
- Test: inline generation smoke checks

- [ ] **Step 1: Write the failing test expectation**

Expected new generation behavior:

- `use_gate=False` keeps original full-vocab generation
- `use_gate=True` evaluates gate only at the assistant-start state and after generated `eoc`
- Positive gate decisions restrict the next-token choice to tool tokens
- Negative gate decisions leave the full vocabulary active

- [ ] **Step 2: Preserve the original generation path**

Keep `generate_with_tool_prediction(...)` as the baseline path for `use_gate=False`.

- [ ] **Step 3: Add a gated generation path**

Implement a step-wise decoder, for example:

- `generate_with_optional_gate(...)`

Core loop:

```python
need_gate = self.use_gate
for step in range(max_new_tokens):
    outputs = self.model(...)
    logits = outputs.logits[:, -1, :]
    if self.use_gate and need_gate:
        gate_logit = self.gate_mlp(outputs.hidden_states[-1][:, -1, :]).squeeze(-1)
        if torch.sigmoid(gate_logit) >= self.gate_threshold:
            logits = self.mask_logits_to_tool_tokens(logits)
        need_gate = False
    next_tokens = decode_from_logits(logits, do_sample, temperature, top_p)
    if self.use_gate and self.use_eoc and next_tokens.item() == self.eoc_token_id:
        need_gate = True
```

- [ ] **Step 4: Route eval/demo generation through the correct path**

In `training.py`, call the gated decoding path only when `model.use_gate` is true.

- [ ] **Step 5: Run generation smoke checks**

Run:

```bash
python - <<'PY'
from transformers import AutoTokenizer
import torch
from compositional.model import FunctionCallingModel
tok = AutoTokenizer.from_pretrained("models/Llama-3.2-3B-Instruct", local_files_only=True)
prompt = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\nhello<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
batch = tok(prompt, return_tensors="pt")
model = FunctionCallingModel(model_name="models/Llama-3.2-3B-Instruct", num_tools=3, tokenizer=tok, device="cpu", dtype=torch.float32, use_eoc=True, use_gate=True)
result = model.generate_with_tool_prediction(batch["input_ids"], batch["attention_mask"], tok, max_new_tokens=5, do_sample=False)
print(type(result), len(result))
PY
```

Expected:

- Generation returns parsed outputs without crashing

### Task 6: Make Parsing and Evaluation `eoc`-Aware

**Files:**
- Modify: `compositional/model.py`
- Modify: `compositional/training.py`

- [ ] **Step 1: Write the failing test expectation**

Expected parsing behavior:

- `eoc` is not included in decoded JSON strings
- In `use_eoc=True` mode, function-call spans end at `eoc` before falling back to legacy slicing

- [ ] **Step 2: Update `_parse_generated_sequences(...)`**

When `use_eoc=True`, split each tool span at the first following `eoc` or, if missing, fall back to the next tool token / `eot`.

- [ ] **Step 3: Keep backward compatibility fields stable**

Do not remove:

- `predicted_tools`
- `function_calls`
- `predicted_tool_name`
- `function_call`

- [ ] **Step 4: Run a parser sanity check**

Use a small handcrafted token sequence or decoded sample to confirm `eoc` is excluded from `function_calls`.

### Task 7: Document the New Modes

**Files:**
- Modify: `compositional/README.md`

- [ ] **Step 1: Add a short mode table**

Document:

- baseline
- `eoc` only
- full `eoc + gate`

- [ ] **Step 2: Add the new CLI flags and the new default `max_length=1024`**

- [ ] **Step 3: Keep the README aligned with the design doc**

### Task 8: Verification

**Files:**
- No new files

- [ ] **Step 1: Run Python syntax verification**

Run:

```bash
python -m py_compile compositional/dataset.py compositional/model.py compositional/training.py compositional/main_sequential.py
```

Expected:

- No syntax errors

- [ ] **Step 2: Run dataset/model sanity checks in all three modes**

Run small inline checks for:

- baseline
- `eoc` only
- full `eoc + gate`

Expected:

- All three modes construct batches successfully
- No missing-token or shape mismatch errors

- [ ] **Step 3: Run a reduced eval-mode generation smoke check**

Use one or two examples from `compositional/data/test/` and confirm:

- baseline uses full-vocab generation
- full mode uses the gated decoding path without crashing

- [ ] **Step 4: Record what was and was not verified**

In the final summary, explicitly state:

- syntax verification result
- smoke checks run
- whether a full training run was performed
- whether GPU-backed end-to-end training remains unverified
