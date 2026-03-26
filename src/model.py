"""
GPT Model Architecture
======================
A decoder-only Transformer (GPT-style) built from scratch with PyTorch.

Components:
  - GPTEmbeddings   : Token + positional embeddings
  - MultiHeadAttention : Scaled dot-product attention with causal mask
  - TransformerBlock   : Attention + FFN with residual connections & LayerNorm
  - GPT                : Full model stacking all blocks
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GPTEmbeddings(nn.Module):
    """Combines token embeddings with learnable positional embeddings."""

    def __init__(self, vocab_size: int, d_model: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        embeddings = self.token_emb(x) + self.pos_emb(positions)
        return self.dropout(embeddings)


class MultiHeadAttention(nn.Module):
    """
    Multi-head causal self-attention.

    Uses an upper-triangular causal mask so each token can only
    attend to itself and previous tokens (autoregressive property).
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_k = d_model // n_heads
        self.n_heads = n_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape

        # Project and split into heads: (B, n_heads, T, d_k)
        q = self.q_linear(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_linear(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Causal (autoregressive) mask — prevent attending to future tokens
        causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(1), float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Weighted sum of values
        context = torch.matmul(attn_weights, v)  # (B, n_heads, T, d_k)
        context = context.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.d_k)
        return self.out_linear(context)


class TransformerBlock(nn.Module):
    """
    Single GPT Transformer block:
      x -> LayerNorm -> MultiHeadAttention -> residual
        -> LayerNorm -> FeedForward          -> residual
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm: normalise → sublayer → residual (GPT-2 style)
        x = x + self.dropout(self.attn(self.norm1(x)))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class GPT(nn.Module):
    """
    GPT Language Model.

    Stacks multiple TransformerBlocks on top of embeddings, then
    projects to vocabulary logits for next-token prediction.

    Args:
        vocab_size  : Size of the character vocabulary.
        d_model     : Dimensionality of token/position embeddings.
        n_heads     : Number of attention heads per block.
        n_layers    : Number of stacked TransformerBlocks.
        d_ff        : Hidden size of the feed-forward network.
        max_seq_len : Maximum supported context length.
        dropout     : Dropout probability (applied in embeddings, attention, FFN).
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        max_seq_len: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embeddings = GPTEmbeddings(vocab_size, d_model, max_seq_len, dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embeddings(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.head(x)

    def num_parameters(self) -> int:
        """Returns the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
