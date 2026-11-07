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


def split_tokens(token_ids, seq_len, val_fraction=0.05):
    """Split tokens chronologically and require independent train/val windows."""
    if seq_len < 1:
        raise ValueError("seq_len must be positive.")
    if not 0.0 < val_fraction < 0.5:
        raise ValueError("val_fraction must be greater than 0 and less than 0.5.")

    split = int(len(token_ids) * (1.0 - val_fraction))
    train_ids = token_ids[:split]
    val_ids = token_ids[split:]
    minimum = seq_len + 2  # make_batch needs at least one valid next-token window.
    if len(train_ids) < minimum or len(val_ids) < minimum:
        raise ValueError(
            "Corpus is too small for a separate validation split: "
            f"need at least {minimum} tokens in both partitions, got "
            f"train={len(train_ids)} and validation={len(val_ids)}. "
            "Add more licensed training text, reduce --seq-len, or choose "
            "a larger --val-fraction. Validation will not reuse training data."
        )
    return (
        torch.tensor(train_ids, dtype=torch.long),
        torch.tensor(val_ids, dtype=torch.long),
    )


def make_batch(tokens, batch_size, seq_len, device, rng=None):
    max_start = len(tokens) - seq_len - 1
    if max_start < 1:
        raise ValueError(
            f"Corpus too small: need more than {seq_len + 1} tokens, "
            f"found {len(tokens)}."
        )
    sampler = rng if rng is not None else random
    starts = [sampler.randint(0, max_start) for _ in range(batch_size)]
    x = torch.stack([tokens[s:s + seq_len] for s in starts])
    y = torch.stack([tokens[s + 1:s + seq_len + 1] for s in starts])
    return x.to(device), y.to(device)


@torch.no_grad()
def evaluate(model, tokens, config, device, batches=10, batch_size=4):
    """Use a repeatable validation sample without consuming training RNG state."""
    was_training = model.training
    model.eval()
    losses = []
    validation_rng = random.Random(0)
    try:
        for _ in range(batches):
            x, y = make_batch(
                tokens, batch_size, config.max_seq_len, device, rng=validation_rng
            )
            _, loss = model(x, y)
            losses.append(loss.item())
    finally:
        model.train(was_training)
    return sum(losses) / len(losses)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/training_data")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 1 or args.eval_every < 1:
        parser.error("steps, batch-size and eval-every must be positive")
    if args.seq_len < 1 or args.lr <= 0:
        parser.error("seq-len and learning rate must be positive")
    if not 0.0 < args.val_fraction < 0.5:
        parser.error("val-fraction must be greater than 0 and less than 0.5")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = ByteTokenizer()
    corpus = load_corpus(args.data)
    ids = tokenizer.encode(corpus)
    train_tokens, val_tokens = split_tokens(
        ids, args.seq_len, val_fraction=args.val_fraction
    )

    config = AIConfig()
    config.vocab_size = VOCAB_SIZE
    config.max_seq_len = args.seq_len
    model = TKCAI(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=0.1
    )

    print(f"Device: {device}")
    print(f"Training text files: {len(list(Path(args.data).rglob('*.txt')))}")
    print(
        f"Tokens: {len(ids):,}; train: {len(train_tokens):,}; "
        f"validation: {len(val_tokens):,}; "
        f"parameters: {sum(p.numel() for p in model.parameters()):,}"
    )
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

        if step == 1 or step % args.eval_every == 0 or step == args.steps:
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
                    "config": vars(config),
                    "step": step,
                    "validation_loss": val_loss,
                    "val_fraction": args.val_fraction,
                },
                checkpoint_dir / "tkc_ai_latest.pt",
            )


if __name__ == "__main__":
    main()
