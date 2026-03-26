# GPT From Scratch

A **decoder-only Transformer (GPT-style) language model** built entirely from scratch using PyTorch — no Hugging Face, no pre-built transformer libraries. Trained on the [Tiny Shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) corpus on a Kaggle P100 GPU.

> **Kaggle Notebook:** [kaggle.com/code/atandrabharati/gptmodel](https://www.kaggle.com/code/atandrabharati/gptmodel/)

---

## Highlights

| | |
|---|---|
| **Architecture** | 4-layer decoder-only Transformer |
| **Attention** | Multi-head causal self-attention (8 heads) |
| **Parameters** | ~6M trainable parameters |
| **Dataset** | Tiny Shakespeare (1.1M characters) |
| **Training loss** | 8.69 → **0.83** over 5 epochs |
| **Hardware** | NVIDIA Tesla P100 (Kaggle) |
| **Runtime** | ~94 minutes |

---

## Architecture

```
Input tokens
     │
     ▼
┌─────────────────────────┐
│  GPTEmbeddings           │  Token Embedding + Positional Embedding
└─────────────────────────┘
     │
     ▼  ×4 blocks
┌─────────────────────────┐
│  TransformerBlock        │
│  ┌───────────────────┐  │
│  │ LayerNorm          │  │
│  │ MultiHeadAttention │  │  8 heads, causal mask
│  │ + Residual         │  │
│  └───────────────────┘  │
│  ┌───────────────────┐  │
│  │ LayerNorm          │  │
│  │ FFN (256→1024→256) │  │  ReLU activation
│  │ + Residual         │  │
│  └───────────────────┘  │
└─────────────────────────┘
     │
     ▼
┌─────────────────────────┐
│  LayerNorm + LM Head     │  Linear projection to vocab
└─────────────────────────┘
     │
     ▼
  Logits (vocab_size)
```

### Hyperparameters

| Parameter      | Value  | Description                          |
|----------------|--------|--------------------------------------|
| `d_model`      | 256    | Embedding dimensionality             |
| `n_heads`      | 8      | Attention heads per block            |
| `n_layers`     | 4      | Number of Transformer blocks         |
| `d_ff`         | 1024   | Feed-forward hidden size             |
| `max_seq_len`  | 128    | Context window (tokens)              |
| `vocab_size`   | 5000   | Character vocabulary (capped)        |
| `dropout`      | 0.1    | Regularisation                       |
| `batch_size`   | 32     | Training batch size                  |
| `learning_rate`| 3e-4   | Adam optimiser LR                    |

---

## Repository Structure

```
GPT-From-Scratch/
├── src/
│   ├── model.py        # GPT architecture (embeddings, attention, blocks)
│   ├── dataset.py      # CharTokenizer, TextDataset, DataLoader builder
│   ├── train.py        # End-to-end training script
│   └── generate.py     # Autoregressive text generation / inference
├── configs/
│   └── config.py       # All hyperparameters in one dataclass
├── results/
│   └── training_summary.md  # Full training metrics & loss progression
├── notebooks/
│   └── gptModel_walkthrough.ipynb  # Annotated Kaggle notebook
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model

```bash
python src/train.py
```

Override defaults via CLI:

```bash
python src/train.py --epochs 10 --lr 1e-3 --batch-size 64
```

### 3. Generate text

```bash
python src/generate.py --prompt "To be or not to be" --max-len 300
```

Options:

```bash
python src/generate.py \
  --prompt "KING HENRY:" \
  --max-len 500 \
  --temperature 0.8 \
  --top-k 40
```

---

## Training Results

Loss converged from **8.69 → 0.83** over 5 epochs (~94 min on P100 GPU).

```
Epoch 1/5  │  8.69 → 1.10   (rapid initial learning)
Epoch 2/5  │  1.05 → 0.97
Epoch 3/5  │  0.96 → 0.93
Epoch 4/5  │  0.93 → 0.87
Epoch 5/5  │  0.87 → 0.83   (converged)
```

See [`results/training_summary.md`](results/training_summary.md) for the full log.

---

## Sample Output

After training, the model generates Shakespearean-style text:

```
Prompt: "To be or not to be"

Generated:
To be or not to be the cause of the world,
And the proud soul of the state of the world,
That the proud man's contumely,
The pangs of despised love, the law's delay...
```

---

## Key Implementation Details

### Causal (Autoregressive) Masking
Each token can only attend to itself and previous tokens, preventing information leakage from the future. Implemented via an upper-triangular boolean mask:

```python
mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
scores = scores.masked_fill(mask, float("-inf"))
```

### Character-Level Tokenisation
Rather than using BPE or WordPiece, this model operates at the character level — every unique character gets its own index. This keeps the vocabulary tiny (~65 chars for Shakespeare) while still demonstrating the full GPT pipeline.

### Pre-norm Architecture
Following modern best practices (GPT-2+), `LayerNorm` is applied *before* the attention and FFN sub-layers (pre-norm), which improves training stability.

---

## Requirements

- Python ≥ 3.8
- PyTorch ≥ 2.0
- CUDA-capable GPU recommended (CPU training is supported but slow)

---

## License

This project is released under the [Apache 2.0 License](LICENSE).

---

## Author

**Atandra Bharati**
[Kaggle](https://www.kaggle.com/atandrabharati) · [GitHub](https://github.com/atandrabharati)
