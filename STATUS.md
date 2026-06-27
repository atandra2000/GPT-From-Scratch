# Repository Status

**Last reviewed:** 2026-06-27
**Last code push:** 2026-03-28 (educational project — content is stable)

## Summary

`GPT-From-Scratch` is a **foundational educational project** in the portfolio. It predates and motivates the four serious LLM reproductions that followed:

- `DeepSeek-v3-Lite` — 422M params, MLA + AuxLossFreeGate MoE + MTP
- `LLaMA-3-Lite` — 515M params, GQA + SwiGLU + chunked cross-entropy
- `FusionLLM` — 415.6M active / 868.6M stored, MLA + GDN + MoE + MTP hybrid
- `TranslationLM` — encoder–decoder Transformer for EN→IT

The repo is intentionally **frozen at educational scale** (~6M params, character-level tokenizer, Tiny Shakespeare). Every later LLM project in this portfolio inherits the same component-by-component from-scratch discipline this repo established.

## Stability

The training run, architecture, and loss curves are reproducible from `src/train.py`. Loss decreased from **8.69 → 0.83** over 5 epochs on P100 in ~94 minutes; final perplexity **2.29**.

Because the educational deliverable is the *complete and correct* decoder-only stack, the codebase is intentionally not actively iterated. Maintenance activity would be:

- Updating dependency versions in `requirements.txt` (a one-off PR, not ongoing)
- Adding tests if requested
- Responding to issues

## Architecture (canonical reference)

For the *reference* decoder-only Transformer block implemented in this repo, see `src/model.py`. The same pre-norm + causal-mask + residual pattern is reused (with improvements) in:

- `LLM/LLaMA-3-Lite/model.py` — adds GQA, RoPE, SwiGLU, RMSNorm, gradient checkpointing
- `LLM/DeepSeek-v3-Lite/models/transformer.py` — adds MLA absorption trick
- `LLM/FusionLLM/` — interleaves MLA + MoE blocks with Gated Delta Net linear attention

## License

Apache 2.0 — see `LICENSE`.
