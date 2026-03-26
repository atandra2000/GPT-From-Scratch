"""
Dataset & Tokenizer
===================
Character-level tokenizer and a sliding-window PyTorch Dataset for
next-token prediction (language modelling objective).
"""

import requests
from typing import List

import torch
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# Data download
# ---------------------------------------------------------------------------

SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)


def download_shakespeare(url: str = SHAKESPEARE_URL) -> str:
    """Downloads the Tiny Shakespeare corpus and returns it as a string."""
    print(f"Downloading dataset from {url} ...")
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    text = response.text
    print(f"Dataset downloaded: {len(text):,} characters")
    return text


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

class CharTokenizer:
    """
    Simple character-level tokenizer.

    Builds a vocabulary from all unique characters in the training text,
    capped at `vocab_size`. Unknown characters are mapped to <UNK>.

    Attributes:
        vocab_size  : Actual vocabulary size (may be smaller than requested).
        char_to_idx : Mapping from character to integer index.
        idx_to_char : Reverse mapping from index to character.
    """

    UNK = "<UNK>"

    def __init__(self, text: str, vocab_size: int):
        chars = sorted(set(text))
        # Reserve last slot for <UNK>
        self.vocab_size = min(len(chars), vocab_size)
        vocab_chars = chars[: self.vocab_size - 1]

        self.char_to_idx = {ch: i for i, ch in enumerate(vocab_chars)}
        self.char_to_idx[self.UNK] = self.vocab_size - 1

        self.idx_to_char = {i: ch for ch, i in self.char_to_idx.items()}

    def encode(self, text: str) -> List[int]:
        unk_idx = self.char_to_idx[self.UNK]
        return [self.char_to_idx.get(ch, unk_idx) for ch in text]

    def decode(self, indices: List[int]) -> str:
        return "".join(self.idx_to_char.get(idx, self.UNK) for idx in indices)

    def __len__(self) -> int:
        return self.vocab_size


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TextDataset(Dataset):
    """
    Sliding-window character-level language modelling dataset.

    Each sample is a (context, target) pair where `target` is `context`
    shifted one position to the right — the standard next-token prediction
    objective used during GPT pre-training.

    Args:
        data    : List of encoded token indices.
        seq_len : Context window length (tokens).
    """

    def __init__(self, data: List[int], seq_len: int):
        self.data = data
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.data) - self.seq_len

    def __getitem__(self, idx: int):
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + 1 : idx + self.seq_len + 1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


# ---------------------------------------------------------------------------
# Helper: build dataloader in one call
# ---------------------------------------------------------------------------

def build_dataloader(
    text: str,
    vocab_size: int,
    seq_len: int,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
):
    """
    Convenience wrapper: tokenize text and return (tokenizer, DataLoader).

    Example
    -------
    >>> text = download_shakespeare()
    >>> tokenizer, loader = build_dataloader(text, vocab_size=5000, seq_len=128, batch_size=32)
    """
    tokenizer = CharTokenizer(text, vocab_size)
    encoded = tokenizer.encode(text)
    dataset = TextDataset(encoded, seq_len)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
    print(
        f"Vocabulary size : {tokenizer.vocab_size}\n"
        f"Dataset samples : {len(dataset):,}\n"
        f"Batches/epoch   : {len(loader):,}"
    )
    return tokenizer, loader
