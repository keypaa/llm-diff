# LLM Activation Analyzer — Build Plan

> **Companion reference:** See [`ai-studio-export-2026-05-25.md`](./ai-studio-export-2026-05-25.md) for the full raw conversation — detailed code blocks, edge case discussions, and rationale that this plan is distilled from. Refer back to it whenever you need more context on a specific decision.

## Philosophy

> "Lazy, efficient, and smart like a mathematician."  
> Build tools that ease everyday life. Avoid complex solutions when simple ones work.

**Hard constraint:** Must run on a 16GB RAM laptop with small models (~7B).

---

## Architecture Overview

```
app.py  (Gradio UI — concurrency_limit=1)
  |
  v
models/loader.py  (Singleton model manager + loading with quantization)
  |
  v
analysis/metrics.py  (Ephemeral math engine — computes & discards on the fly)
  |
  v
app.py  (Plotly charts returned to Gradio)
```

**Key principle:** Never store raw tensors. Compute metrics immediately, discard, keep only ~5KB of floats.

---

## File-by-File Specification

### 1. `models/loader.py` — Model Loading & Execution Plan

**Responsibility:** Load models with optional 4-bit/8-bit quantization, handle Base vs. LoRA adapter mode, compatibility verification.

**Classes:**
- `ModelExecutionPlan` — dataclass holding:
  - `mode: str` — `"single_model_adapter_swap"` or `"dual_model_comparison"`
  - `model_a`, `model_b` — model instances
  - `tokenizer_a`, `tokenizer_b` — tokenizers
  - `warnings: list[str]` — compatibility issues

**Functions:**
- `get_quantization_config(quant_mode: str) -> Optional[BitsAndBytesConfig]` — returns config for `"4-bit"`, `"8-bit"`, or `None`
- `load_model_pair(model_a_path, model_b_path=None, adapter_path=None, quantization="None", allow_mismatch=False) -> ModelExecutionPlan`
  - **CASE 1 (LoRA):** Single model + adapter. Use `PeftModel.from_pretrained`. Runtime uses `model.disable_adapter()` to get base vs adapter states in one model — zero extra VRAM.
  - **CASE 2 (Dual):** Two independent models. Run compatibility checks:
    1. Architecture mismatch
    2. Hidden dimension mismatch (disables direct vector math)
    3. Tokenizer/vocabulary mismatch
    - If warnings exist and `not allow_mismatch`, raise `ValueError`.
  - Always print loading progress.

### 2. `models/singleton.py` — Global Model Manager (Memory Protector)

**Responsibility:** Prevent Gradio from duplicating models via `gr.State()`. Hold models in a strict Python singleton.

**Behavior on new load request:**
- If requested model paths match currently loaded → skip.
- If mismatch → `del model`, `gc.collect()`, `torch.cuda.empty_cache()`, then load fresh.
- Never hold more than one `ModelExecutionPlan` at a time.

### 3. `analysis/metrics.py` — Ephemeral Math Engine

**Responsibility:** Compute metrics from `outputs.hidden_states`, return only lightweight float dicts.

**Matched mode** (same hidden dim + same token count):
- Cosine Similarity per layer (`F.cosine_similarity`)
- MSE per layer (`F.mse_loss`)

**Mismatched mode** (different dims or tokenizers — "apples to oranges"):
- Mean L2 Magnitude per layer per model (independent scalars)
- Activation Sparsity (Hoyer's measure: `1 - L1/(L2*sqrt(dim))`)
- Layer Velocity (MSE between consecutive layers within same model)

**Payload structure (<5KB):**
```python
{
  "layer_0": {
    "cosine_sim": 0.998, "mse": 0.001,
    "model_a_l2": 14.5,   "model_b_l2": 14.6,
    "model_a_velocity": 0.0, "model_b_velocity": 0.0,
    "model_a_sparsity": 0.45, "model_b_sparsity": 0.42
  },
  ...
}
```

**Token alignment for mismatched models:** Use **last-token** (`hidden_state[:, -1, :]`) for per-layer comparison. This avoids complex sequence alignment.

**No hooks.** Use `output_hidden_states=True` on the forward pass — works across all architectures, no memory leaks.

### 4. `app.py` — Gradio UI & Pipeline

**Layout:**
- **Left column (Control Center):**
  - Textbox: Model A path
  - Textbox: Model B path OR LoRA adapter path
  - Dropdown: Quantization (`4-bit` default, `8-bit`, `None`)
  - Checkbox: `Authorize Mismatched Models`
  - Textarea: Prompt
  - Button: **Run Analysis**
- **Right column (Dashboard):**
  - Dynamic status banner (hidden by default, shows warnings on mismatch)
  - **Tab 1: Matched Vectors** — Cosine Similarity line chart + MSE bar chart (hidden when mismatch)
  - **Tab 2: Structural Integrity** — Layer Velocity + L2 Magnitudes overlay + Sparsity (always visible)

**Pipeline (`run_pipeline`):**
1. Receive strings from UI
2. Check singleton: load required models or swap
3. Tokenize prompt
4. Forward pass with `output_hidden_states=True`
5. Compute metrics → immediately discard hidden states
6. Generate Plotly charts from 5KB payload
7. Return charts + UI visibility updates

**Safety:**
- `concurrency_limit=1` — sequential queue, no VRAM corruption
- `gr.State` for warnings only (never store models in Gradio state)
- Dynamic UI toggle on mismatch via `update_ui_for_mismatch()`

---

## Data Flow (End-to-End)

```
User clicks "Run"
  → Gradio fires run_prompt(...) with form strings
  → Singleton.load_if_needed(model_a, model_b/adapter, quant)
      → calls loader.load_model_pair() if cache miss
      → returns ModelExecutionPlan
  → Tokenize prompt
  → model(**inputs, output_hidden_states=True)
  → metrics.extract(payload) → 5KB dict
  → charting functions → Plotly figures
  → Return [plots..., warning_banner_update, tab_visibility_updates]
  → Gradio renders
```

---

## Traps Avoided (Design Wins)

| Trap | Solution |
|------|----------|
| Tokenizer misalignment | Last-token pooling for mismatched models |
| Memory leak from hooks | `output_hidden_states=True` — no hooks needed |
| CPU RAM explosion | Compute metrics inline, discard tensors immediately |
| Gradio concurrency | `concurrency_limit=1` + singleton model manager |
| Layer naming variance | HF `output_hidden_states=True` is architecture-agnostic |
| Gradio duplicating models | Singleton bypasses `gr.State()` for model objects |

---

## Build Order

1. `models/loader.py` — loading, quantization, execution plan
2. `models/singleton.py` — global state manager
3. `analysis/metrics.py` — matched + mismatched math
4. `app.py` — UI layout, pipeline, charting, dynamic toggles
