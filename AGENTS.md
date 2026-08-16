# AGENTS.md — GPT-2

> Read the workspace `LLM/AGENTS.md` and the parent `CoreProjects/AGENTS.md`
> (+ `self.md`) first. Higher-level rules are authoritative; this file adds
> project-specific rules only.
>
> **This project is frozen at educational scale.** Default to reading and
> explaining its code; only modify it for bug fixes or doc corrections.

> **Project:** `LLM/GPT2/` · **Type:** foundational educational decoder-only LM
> **Architecture:** GPT-style from scratch — 4 layers, 256 dim, 8 heads,
> char-level vocab (5,000) · **Scale:** ~6M params, 5 epochs
> **Hardware:** P100 (Kaggle) · **Status:** complete, frozen at educational scale.
> **Detail:** see `README.md`.

## 1. Subagent: `gpt2-educational`

**Triggers:** "Show me a from-scratch GPT-style model", "How does the causal
attention mask work?", "Why pre-norm blocks?", "Train GPT from scratch on
Tiny Shakespeare."

**Knows cold:**
- The portfolio's **foundational educational** project — a from-scratch
  GPT-style decoder built entirely without abstraction layers
  (`src/model.py`). Each component implemented by hand: `GPTEmbeddings`
  (learned token + positional), `MultiHeadAttention` (8 heads, upper-triangular
  causal mask), pre-norm `TransformerBlock` (ReLU FFN, residual), linear head.
- No HuggingFace, no tiktoken — character-level tokenizer (`src/dataset.py`).
- Training: Adam (lr 3e-4), 5 epochs, batch 32 × seq 128. Tiny Shakespeare,
  auto-downloaded (Karpathy `char-rnn`). Loss 8.69 → 0.83 over ~94 min on P100.
- Sampling: temperature + top-k (`src/generate.py`).

## 2. Hard rules

1. **Keep it from-scratch and dependency-light.** No HF Trainer, no
   `transformers` model classes, no tiktoken — the educational value is in
   hand-built components.
2. **Preserve the canonical 5-epoch recipe.** Adam (lr 3e-4) on Tiny
   Shakespeare. Different recipes lose the clean loss-curve signal.
3. **Never scale it up.** Do not swap to GPT-3/4-style architecture, add MoE,
   MLA, or any 2024-era technique — this project stays the ~6M char-level
   educational reference.
4. **Don't backport the old nanoGPT docs.** The repo's self-description
   (char-level, ~6M) is authoritative — earlier "124M / tiktoken / HF
   loading" notes describe a superseded version and must not resurface.

## 3. Files

- `src/model.py` — `GPTEmbeddings`, `MultiHeadAttention`, `TransformerBlock`, `GPT`.
- `src/dataset.py` — `CharTokenizer` (UNK handling), `TextDataset`, auto-download of Tiny Shakespeare.
- `src/train.py` — full loop: Adam, CrossEntropyLoss, per-batch logging, checkpointing.
- `src/generate.py` — autoregressive generation (temperature + top-k).
- `configs/config.py` — `GPTConfig` dataclass (single source of truth).

## 4. Known caveats

- Educational only; not benchmarked against the other portfolio projects.
- Character-level, so results are per-char (perp 2.29, 1.20 bits/char) — not
  comparable to BPE-tokenized models.
- Frozen at ~6M params by design.
