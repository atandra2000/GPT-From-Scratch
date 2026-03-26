"""
Text Generation (Inference)
===========================
Loads a trained GPT checkpoint and autoregressively generates text.

Usage
-----
    python src/generate.py --prompt "To be or not to be" --max-len 200
    python src/generate.py --prompt "KING:" --temperature 0.8 --max-len 300
"""

import argparse

import torch
import torch.nn.functional as F

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.dataset import download_shakespeare, CharTokenizer
from src.model import GPT
from configs.config import GPTConfig


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate(
    model: GPT,
    tokenizer: CharTokenizer,
    prompt: str,
    max_len: int = 200,
    temperature: float = 1.0,
    top_k: int = 0,
    device: str = "cpu",
) -> str:
    """
    Autoregressively generates text from a prompt.

    Args:
        model       : Trained GPT model in eval mode.
        tokenizer   : CharTokenizer used during training.
        prompt      : Seed text to start generation from.
        max_len     : Number of new tokens to generate.
        temperature : Softmax temperature. Lower = more deterministic.
                      Recommended range: 0.5 – 1.2.
        top_k       : If > 0, restrict sampling to top-k most likely tokens.
        device      : Torch device string.

    Returns:
        Generated text string (includes the original prompt).
    """
    model.eval()
    tokens = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)

    for _ in range(max_len):
        # Truncate context to max_seq_len if needed
        context = tokens[:, -model.embeddings.pos_emb.num_embeddings :]

        logits = model(context)[:, -1, :] / temperature  # (1, vocab_size)

        if top_k > 0:
            # Zero out all logits except top-k
            values, _ = torch.topk(logits, top_k)
            logits[logits < values[:, -1].unsqueeze(-1)] = float("-inf")

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)  # (1, 1)
        tokens = torch.cat([tokens, next_token], dim=1)

    return tokenizer.decode(tokens[0].cpu().tolist())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Generate text with a trained GPT model.")
    parser.add_argument("--checkpoint", type=str, default="gpt_model.pth",
                        help="Path to saved model weights (.pth file)")
    parser.add_argument("--prompt", type=str, default="To be or not to be",
                        help="Seed text for generation")
    parser.add_argument("--max-len", type=int, default=200,
                        help="Number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature (default: 1.0)")
    parser.add_argument("--top-k", type=int, default=0,
                        help="Top-k sampling; 0 = disabled (default: 0)")
    return parser.parse_args()


def main():
    args = parse_args()
    config = GPTConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Rebuild tokenizer from corpus (same vocab as training)
    print("Building tokenizer from Shakespeare corpus ...")
    text = download_shakespeare()
    tokenizer = CharTokenizer(text, config.vocab_size)
    config.vocab_size = tokenizer.vocab_size

    # Load model
    model = GPT(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        d_ff=config.d_ff,
        max_seq_len=config.max_seq_len,
        dropout=config.dropout,
    ).to(device)

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(
            f"Checkpoint not found: '{args.checkpoint}'. "
            "Run `python src/train.py` first to train the model."
        )

    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    print(f"Loaded checkpoint: {args.checkpoint}")

    # Generate
    print(f"\nPrompt : {args.prompt!r}")
    print(f"Temperature: {args.temperature}  |  Top-k: {args.top_k}  |  Max tokens: {args.max_len}")
    print("\n" + "=" * 60)

    output = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_len=args.max_len,
        temperature=args.temperature,
        top_k=args.top_k,
        device=device,
    )
    print(output)
    print("=" * 60)


if __name__ == "__main__":
    main()
