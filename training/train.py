"""Train TKC-AI from random initialization on local UTF-8 .txt files.

Put legally usable training text in data/training_data/. This script never
downloads data or pretrained weights. Start with a small corpus only to test
the pipeline; meaningful language ability requires a large, high-quality corpus.
"""
import argparse
import hashlib
import math
import os
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
    minimum = seq_len + 2
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


def _capture_rng_state():
    return {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng_state(state):
    if not isinstance(state, dict):
        return False
    try:
        if "python" in state:
            random.setstate(state["python"])
        if "torch" in state:
            torch.set_rng_state(state["torch"].cpu())
        cuda_restored = True
        if torch.cuda.is_available():
            cuda_states = state.get("cuda")
            if not cuda_states or len(cuda_states) != torch.cuda.device_count():
                cuda_restored = False
            else:
                torch.cuda.set_rng_state_all([item.cpu() for item in cuda_states])
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError(f"Checkpoint RNG state is invalid: {error}") from error
    else:
        if not torch.cuda.is_available():
            cuda_restored = True
    return "python" in state and "torch" in state and cuda_restored


def save_checkpoint(
    path, model, optimizer, config, step, val_loss,
    val_fraction, data_sha256, seed, batch_size,
):
    """Atomically save model, optimizer and RNG state for reproducible resume."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": vars(config).copy(),
        "step": int(step),
        "validation_loss": float(val_loss),
        "val_fraction": float(val_fraction),
        "data_sha256": str(data_sha256),
        "seed": int(seed),
        "batch_size": int(batch_size),
        "rng_state": _capture_rng_state(),
    }
    try:
        torch.save(checkpoint, temporary_path)
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def restore_checkpoint_state(
    checkpoint, model, optimizer, config, data_sha256,
    val_fraction, batch_size, allow_data_change=False,
):
    """Restore compatible state; return step and restoration/verification flags."""
    if "model" not in checkpoint:
        raise ValueError("Checkpoint is missing model weights.")
    saved_config = checkpoint.get("config")
    if not isinstance(saved_config, dict):
        raise ValueError("Checkpoint is missing a valid model config.")
    for key, expected in vars(config).items():
        if saved_config.get(key) != expected:
            raise ValueError(
                f"Checkpoint config mismatch for {key}: "
                f"saved={saved_config.get(key)!r}, requested={expected!r}."
            )

    saved_fraction = checkpoint.get("val_fraction")
    if saved_fraction is not None and not math.isclose(
        float(saved_fraction), float(val_fraction), rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError(
            "Validation split differs from the checkpoint. Use the same "
            "--val-fraction to resume comparable training."
        )

    saved_batch_size = checkpoint.get("batch_size")
    if saved_batch_size is not None and int(saved_batch_size) != int(batch_size):
        raise ValueError(
            "Batch size differs from the checkpoint. Use the same --batch-size "
            "to preserve the training schedule."
        )

    saved_fingerprint = checkpoint.get("data_sha256")
    data_matches = saved_fingerprint is not None and saved_fingerprint == data_sha256
    if saved_fingerprint is not None and not data_matches and not allow_data_change:
        raise ValueError(
            "Training corpus differs from the checkpoint. Resume with the same "
            "corpus, or pass --allow-data-change to accept a non-identical run."
        )

    model.load_state_dict(checkpoint["model"])
    optimizer_restored = "optimizer" in checkpoint
    if optimizer_restored:
        optimizer.load_state_dict(checkpoint["optimizer"])
    rng_restored = _restore_rng_state(checkpoint.get("rng_state"))
    return (
        int(checkpoint.get("step", 0)),
        optimizer_restored,
        rng_restored,
        data_matches,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/training_data")
    parser.add_argument(
        "--steps", type=int, default=1000,
        help="Total target step, not additional steps when resuming.",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--val-fraction", type=float, default=None)
    parser.add_argument(
        "--lr", type=float, default=None,
        help="Leave unset when resuming to keep the saved optimizer LR.",
    )
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--resume", help="Resume from a TKC-AI training checkpoint.")
    parser.add_argument(
        "--allow-data-change", action="store_true",
        help="Allow a different corpus; the resumed run will use changed data.",
    )
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    if args.steps < 1 or args.eval_every < 1:
        parser.error("steps and eval-every must be positive")
    if args.lr is not None and args.lr <= 0:
        parser.error("learning rate must be positive")

    resume_state = None
    if args.resume:
        try:
            resume_state = torch.load(
                args.resume,
                map_location="cuda" if torch.cuda.is_available() else "cpu",
                weights_only=True,
            )
        except FileNotFoundError:
            parser.error(f"Resume checkpoint not found: {args.resume}")
        if not isinstance(resume_state, dict) or not isinstance(
            resume_state.get("config"), dict
        ):
            parser.error("Resume checkpoint is missing its model config.")

    config = AIConfig()
    if resume_state is not None:
        for key, value in resume_state["config"].items():
            if hasattr(config, key):
                setattr(config, key, value)
        if args.seq_len is not None and args.seq_len != config.max_seq_len:
            parser.error("--seq-len must match the resume checkpoint context length")
        val_fraction = (
            args.val_fraction
            if args.val_fraction is not None
            else resume_state.get("val_fraction", 0.05)
        )
        seed = args.seed if args.seed is not None else resume_state.get("seed", 42)
        batch_size = (
            args.batch_size if args.batch_size is not None
            else resume_state.get("batch_size", 4)
        )
    else:
        config.vocab_size = VOCAB_SIZE
        config.max_seq_len = args.seq_len if args.seq_len is not None else 256
        val_fraction = args.val_fraction if args.val_fraction is not None else 0.05
        seed = args.seed if args.seed is not None else 42
        batch_size = args.batch_size if args.batch_size is not None else 4

    if config.vocab_size != VOCAB_SIZE:
        parser.error("Checkpoint vocabulary does not match the current ByteTokenizer")
    if batch_size < 1:
        parser.error("batch-size must be positive")
    if config.max_seq_len < 1:
        parser.error("seq-len must be positive")
    if not 0.0 < val_fraction < 0.5:
        parser.error("val-fraction must be greater than 0 and less than 0.5")

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = ByteTokenizer()
    corpus = load_corpus(args.data)
    data_sha256 = hashlib.sha256(corpus.encode("utf-8")).hexdigest()
    ids = tokenizer.encode(corpus)
    train_tokens, val_tokens = split_tokens(
        ids, config.max_seq_len, val_fraction=val_fraction
    )

    model = TKCAI(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr if args.lr is not None else 3e-4,
        weight_decay=0.1,
    )
    start_step = 0
    if resume_state is not None:
        (
            start_step,
            optimizer_restored,
            rng_restored,
            data_verified,
        ) = restore_checkpoint_state(
            resume_state, model, optimizer, config, data_sha256,
            val_fraction, batch_size, allow_data_change=args.allow_data_change,
        )
        if not optimizer_restored:
            print("Warning: legacy checkpoint has no optimizer state; optimizer starts fresh.")
        if not rng_restored:
            print("Warning: checkpoint has no complete RNG state; exact reproducibility is unavailable.")
        if resume_state.get("batch_size") is None:
            print("Warning: legacy checkpoint does not record batch size; using the selected value.")
        if resume_state.get("data_sha256") is None:
            print("Warning: legacy checkpoint has no corpus fingerprint; data identity cannot be verified.")
        elif not data_verified:
            print("Warning: corpus fingerprint differs; proceeding because --allow-data-change was set.")
        if args.lr is not None:
            for group in optimizer.param_groups:
                group["lr"] = args.lr
        if start_step >= args.steps:
            parser.error(
                f"Checkpoint is already at step {start_step}; set --steps above that "
                "because --steps is the total target step."
            )

    print(f"Device: {device}")
    print(f"Training text files: {len(list(Path(args.data).rglob('*.txt')))}")
    print(
        f"Tokens: {len(ids):,}; train: {len(train_tokens):,}; "
        f"validation: {len(val_tokens):,}; "
        f"parameters: {sum(p.numel() for p in model.parameters()):,}"
    )
    if resume_state is None:
        print("Training from random initialization; no pretrained weights.")
    else:
        print(f"Resuming from step {start_step}; target total steps: {args.steps}.")

    model.train()
    for step in range(start_step + 1, args.steps + 1):
        x, y = make_batch(
            train_tokens, batch_size, config.max_seq_len, device
        )
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step == start_step + 1 or step % args.eval_every == 0 or step == args.steps:
            val_loss = evaluate(
                model, val_tokens, config, device,
                batch_size=batch_size
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
            save_checkpoint(
                checkpoint_dir / "tkc_ai_latest.pt", model, optimizer, config,
                step, val_loss, val_fraction, data_sha256, seed, batch_size,
            )


if __name__ == "__main__":
    main()
