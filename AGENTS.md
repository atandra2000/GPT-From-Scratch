# AGENTS.md — GPT-2

> Read root `AGENTS.md` and `self.md` first. Workspace rules are
> authoritative; this file adds project-specific rules only.

> **Project:** `LLM/GPT2/` · **Type:** educational decoder-only LM
> **Architecture:** GPT-2 from Radford et al. 2019 — 12 layers, 768 dim,
> 12 heads, 50,257 vocab · **Scale:** 124M params, 200 iters
> **Hardware:** MPS / CUDA / CPU · **Status:** educational / demonstration.
> **Detail:** see `README.md §1`.

## 1. Subagent: `gpt2-educational`

**Triggers:** "Show me a minimal GPT-2", "How does CausalSelfAttention
work?", "Why does the c_proj init use std=0.02 / sqrt(2 * n_layer)?",
"Train GPT-2 from scratch on Shakespeare."

**Knows cold:**
- The portfolio's **educational** project — minimal from-scratch GPT-2 in
  ~204 lines (`train_gpt2.py`). Each component built layer-by-layer:
  fused QKV, GELU-tanh MLP, pre-norm LayerNorm, tied I/O embeddings.
- Loads HuggingFace pretrained weights (`gpt2` → `gpt2-xl`) for inference /
  fine-tuning.
- Tokenizer: `tiktoken` (GPT-2 BPE).
- Training: AdamW (lr 3e-4), 200 iters, batch 4 × seq 128, auto-selects
  `mps`/`cuda`/`cpu`. Dataset: Shakespeare *Coriolanus* (`input.txt`).

## 2. Hard rules

1. **Always** preserve `c_proj` weight init `std = 0.02 / sqrt(2 * n_layer)`.
   Naive init over-shoots activations at depth.
2. **Always** use AdamW (lr 3e-4) for the canonical 200-iter run. Different
   recipes (Lion, SOAP) lose the educational signal.
3. **Never** swap to GPT-3 / GPT-4 architecture — this project must remain
   GPT-2 (124M), the educational reference.

## 3. Files

- `train_gpt2.py` — full model + loop in ~204 lines.
- `play.ipynb` — interactive generation.
- `input.txt` — Shakespeare *Coriolanus* corpus.

## 4. Known caveats

- Educational only; not benchmarked against other portfolio projects.
- `gpt2-xl` loading requires ~6 GB RAM.
