"""Train TKC-AI from random initialization on local UTF-8 .txt files.

Put legally usable training text in data/training_data/. This script never
downloads data or pretrained weights. Start with a small corpus only to test
the pipeline; meaningful language ability requires a large, high-quality corpus.
"""
import argparse
import math
import random
from pathlib import Path

import torch

from brain.tokenizer import ByteTokenizer, VOCAB_SIZE
from brain.transformer import AIConfig, TKCAI


def load_corpus(folder):
    paths = sorted(Path(folder).rglob("*.txt"))
    if not paths:
        raise FileNotFoundError(
            f"No .txt files found in {folder}. Add UTF-8 training text first."
        )
    chunks = []
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        if text:
            chunks.append(text)
    if not chunks:
        raise ValueError("Training text files are empty.")
    return "\n\n".join(chunks)


def make_batch(tokens, batch_size, seq_len, device):
    max_start = len(tokens) - seq_len - 1
    if max_start < 1:
        raise ValueError(
            f"Corpus too small: need more than {seq_len + 1} tokens, "
            f"found {len(tokens)}."
        )
    starts = [random.randint(0, max_start) for _ in range(batch_size)]
    x = torch.stack([tokens[s:s + seq_len] for s in starts])
    y = torch.stack([tokens[s + 1:s + seq_len + 1] for s in starts])
    return x.to(device), y.to(device)


@torch.no_grad()
def evaluate(model, tokens, config, device, batches=10, batch_size=4):
    model.eval()
    losses = []
    for _ in range(batches):
        x, y = make_batch(tokens, batch_size, config.max_seq_len, device)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/training_data")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = ByteTokenizer()
    corpus = load_corpus(args.data)
    ids = tokenizer.encode(corpus)
    split = max(1, int(len(ids) * 0.95))
    train_tokens = torch.tensor(ids[:split], dtype=torch.long)
    val_tokens = torch.tensor(ids[split:], dtype=torch.long)
    if len(val_tokens) <= args.seq_len + 1:
        val_tokens = train_tokens

    config = AIConfig()
    config.vocab_size = VOCAB_SIZE
    config.max_seq_len = args.seq_len
    model = TKCAI(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=0.1
    )

    print(f"Device: {device}")
    print(f"Training text files: {len(list(Path(args.data).rglob('*.txt')))}")
    print(f"Tokens: {len(ids):,}; parameters: "
          f"{sum(p.numel() for p in model.parameters()):,}")
    print("Training from random initialization; no pretrained weights.")

    model.train()
    for step in range(1, args.steps + 1):
        x, y = make_batch(
            train_tokens, args.batch_size, config.max_seq_len, device
        )
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step == 1 or step % args.eval_every == 0:
            val_loss = evaluate(
                model, val_tokens, config, device,
                batch_size=args.batch_size
            )
            perplexity = math.exp(min(val_loss, 20))
            print(
                f"step {step}/{args.steps} "
                f"train_loss={loss.item():.4f} "
                f"val_loss={val_loss:.4f} "
                f"val_perplexity={perplexity:.2f}"
            )
            checkpoint_dir = Path(args.checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": vars(config) if hasattr(config, "__dict__")
                              else {
                                  "vocab_size": config.vocab_size,
                                  "max_seq_len": config.max_seq_len,
                                  "d_model": config.d_model,
                                  "n_heads": config.n_heads,
                                  "n_layers": config.n_layers,
                                  "dropout": config.dropout,
                              },
                    "step": step,
                    "validation_loss": val_loss,
                },
                checkpoint_dir / "tkc_ai_latest.pt",
            )


if __name__ == "__main__":
    main()
