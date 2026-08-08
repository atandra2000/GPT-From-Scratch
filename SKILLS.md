# SKILLS.md — GPT-2

> Read root `AGENTS.md` and `self.md` first. Workspace rules are
> authoritative; this file adds project-specific workflows.

## Skill 1: Run the canonical 5-epoch training

```bash
cd LLM/GPT2
python src/train.py
```

`src/dataset.py` auto-downloads Tiny Shakespeare (Karpathy `char-rnn`).
Prints per-batch and per-epoch loss; saves `gpt_model.pth`. Expected: loss
8.69 → 0.83 over ~94 min on P100 (perplexity 2.29, 1.20 bits/char).

## Skill 2: Generate interactively

```bash
python src/generate.py --prompt "To be or not to be" --max-len 200 --temperature 0.8 --top-k 40
```

| Flag | Default | Description |
|------|---------|-------------|
| `--prompt` | `"To be or not to be"` | Seed text |
| `--max-len` | `200` | Tokens to generate |
| `--temperature` | `1.0` | Creativity scale |
| `--top-k` | `0` (disabled) | Restrict to top-k tokens |
| `--checkpoint` | `gpt_model.pth` | Saved weights |

## Skill 3: Ablate a component

To demonstrate e.g. "what happens without pre-norm":

1. Edit `TransformerBlock.forward` in `src/model.py`.
2. Re-run `python src/train.py`.
3. Compare loss curves.

## Pitfalls

- **Educational scope:** intentionally tiny. Don't add MoE, MLA, or any
  2024-era technique — it would defeat the purpose.
- **Character-level:** the tokenizer is char-based (vocab 5,000, UNK
  handling) — do not assume BPE / tiktoken semantics.
- **No HF weights:** this repo never loads HuggingFace pretrained checkpoints;
  it trains from scratch on Tiny Shakespeare.
