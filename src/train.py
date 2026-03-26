"""
Training Script
===============
End-to-end training pipeline for the GPT character-level language model.

Usage
-----
    python src/train.py                         # uses default config
    python src/train.py --epochs 10 --lr 1e-3  # override specific args

The script:
  1. Downloads the Tiny Shakespeare corpus.
  2. Builds the character tokenizer and DataLoader.
  3. Instantiates the GPT model.
  4. Runs the training loop with per-batch loss logging.
  5. Saves the trained model weights to disk.
"""

import argparse
import time

import torch
import torch.nn as nn

# Local imports
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.dataset import download_shakespeare, build_dataloader
from src.model import GPT
from configs.config import GPTConfig


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(model: GPT, dataloader, config: GPTConfig, device: str) -> list:
    """
    Trains the GPT model and returns a list of (epoch, avg_loss) tuples.

    Args:
        model      : Instantiated GPT model (moved to `device`).
        dataloader : PyTorch DataLoader yielding (x, y) batches.
        config     : GPTConfig with training hyper-parameters.
        device     : 'cuda' or 'cpu'.

    Returns:
        history : List of dicts with 'epoch' and 'avg_loss' keys.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    history = []
    model.train()

    for epoch in range(1, config.num_epochs + 1):
        epoch_start = time.time()
        total_loss = 0.0

        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x)                                      # (B, T, vocab_size)
            loss = criterion(logits.view(-1, config.vocab_size), y.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % config.log_interval == 0:
                print(
                    f"Epoch {epoch}/{config.num_epochs}  "
                    f"Batch {batch_idx:>6,}  "
                    f"Loss: {loss.item():.4f}"
                )

        avg_loss = total_loss / len(dataloader)
        elapsed = time.time() - epoch_start
        print(
            f"\n{'='*60}\n"
            f"Epoch {epoch}/{config.num_epochs} complete  "
            f"Avg Loss: {avg_loss:.4f}  "
            f"Time: {elapsed:.1f}s\n"
            f"{'='*60}\n"
        )
        history.append({"epoch": epoch, "avg_loss": avg_loss})

    return history


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train a GPT language model from scratch.")
    parser.add_argument("--epochs", type=int, default=None, help="Override num_epochs in config")
    parser.add_argument("--lr", type=float, default=None, help="Override learning_rate in config")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch_size in config")
    parser.add_argument("--save-path", type=str, default="gpt_model.pth", help="Path to save model weights")
    return parser.parse_args()


def main():
    args = parse_args()
    config = GPTConfig()

    # Allow CLI overrides
    if args.epochs is not None:
        config.num_epochs = args.epochs
    if args.lr is not None:
        config.learning_rate = args.lr
    if args.batch_size is not None:
        config.batch_size = args.batch_size

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"\nConfig:\n{config}\n")

    # Data
    text = download_shakespeare()
    tokenizer, dataloader = build_dataloader(
        text,
        vocab_size=config.vocab_size,
        seq_len=config.max_seq_len,
        batch_size=config.batch_size,
    )
    # Update actual vocab size (may differ from requested)
    config.vocab_size = tokenizer.vocab_size

    # Model
    model = GPT(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        d_ff=config.d_ff,
        max_seq_len=config.max_seq_len,
        dropout=config.dropout,
    ).to(device)

    print(f"Model parameters: {model.num_parameters():,}")

    # Train
    history = train(model, dataloader, config, device)

    # Save weights
    torch.save(model.state_dict(), args.save_path)
    print(f"\nModel saved to: {args.save_path}")

    # Print loss summary
    print("\nTraining Summary:")
    print(f"{'Epoch':<8} {'Avg Loss':<12}")
    print("-" * 20)
    for entry in history:
        print(f"{entry['epoch']:<8} {entry['avg_loss']:<12.4f}")


if __name__ == "__main__":
    main()
