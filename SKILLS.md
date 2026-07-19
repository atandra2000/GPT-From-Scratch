# SKILLS.md — GPT-2

> Read root `AGENTS.md` and `self.md` first. Workspace rules are
> authoritative; this file adds project-specific workflows.

## Skill 1: Run the canonical 200-iter training

```bash
cd LLM/GPT2
python train_gpt2.py
```

Expects `input.txt` (Shakespeare) in the repo root. Outputs loss curve in
terminal + `log.txt`.

## Skill 2: Load a HuggingFace pretrained checkpoint

```python
import tiktoken
from train_gpt2 import GPT, ModelConfig

enc = tiktoken.get_encoding("gpt2")
m = GPT(ModelConfig())                                  # 124M scratch
# Or load any HF size:
from transformers import GPT2LMHeadModel
m.load_state_dict(GPT2LMHeadModel.from_pretrained("gpt2").state_dict())
```

Supported HF sizes: `gpt2`, `gpt2-medium`, `gpt2-large`, `gpt2-xl`.

## Skill 3: Generate interactively

Open `play.ipynb`. Prompts with `The capital of France is` should yield
plausible completions on the pretrained `gpt2`. The 200-iter scratch model
will produce Shakespearean-ish gibberish.

## Skill 4: Modify the block to ablate a component

If you want to demonstrate e.g. "what happens without pre-norm":

1. Edit `Block` in `train_gpt2.py` (the `forward` method, ~10 lines).
2. Re-run `python train_gpt2.py`.
3. Compare loss curves.

**Pitfall:** always re-init `c_proj` after changing the residual add path.

## Pitfalls

- **Educational scope:** this project is intentionally tiny. Don't add MoE,
  MLA, or any LLM/2024-era technique — it would defeat the purpose.
- **`c_proj` init matters:** don't skip
  `nn.init.normal_(self.c_proj.weight, std=0.02 / math.sqrt(2 * config.n_layer))`.
- **Pretrained weight loading:** `gpt2-xl` (~1.5B params) won't fit on a
  P100 / T4; use A100 80GB or larger.
