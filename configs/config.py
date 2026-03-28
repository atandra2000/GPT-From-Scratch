"""
Model & Training Configuration
================================
All hyper-parameters in one place. Modify this file to experiment with
different model sizes, learning rates, or training durations.
"""

from dataclasses import dataclass


@dataclass
class GPTConfig:
    # ── Model architecture ──────────────────────────────────────────────────
    vocab_size:  int   = 5000   # Character vocabulary size (capped)
    d_model:     int   = 256    # Token/positional embedding dimensionality
    n_heads:     int   = 8      # Number of attention heads per block
    n_layers:    int   = 4      # Number of stacked Transformer blocks
    d_ff:        int   = 1024   # Hidden size of the FFN within each block
    max_seq_len: int   = 128    # Maximum context length (tokens)
    dropout:     float = 0.1    # Dropout probability

    # ── Training ────────────────────────────────────────────────────────────
    batch_size:    int   = 32      # Samples per gradient update
    num_epochs:    int   = 5       # Full passes over the dataset
    learning_rate: float = 3e-4   # Adam initial learning rate
    log_interval:  int   = 100    # Print loss every N batches

    def __str__(self) -> str:
        lines = [
            "GPTConfig",
            "─" * 40,
            f"  Architecture : {self.n_layers} layers × {self.n_heads} heads × d={self.d_model}",
            f"  FFN size     : {self.d_ff}",
            f"  Vocab size   : {self.vocab_size}",
            f"  Seq length   : {self.max_seq_len}",
            f"  Dropout      : {self.dropout}",
            "─" * 40,
            f"  Batch size   : {self.batch_size}",
            f"  Epochs       : {self.num_epochs}",
            f"  Learning rate: {self.learning_rate}",
        ]
        return "\n".join(lines)
