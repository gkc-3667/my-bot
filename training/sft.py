"""Supervised instruction tuning for a TKC-AI checkpoint trained from scratch.

JSONL rows must contain string fields: prompt and answer. Prompt tokens are
masked from the loss; the model is trained to predict only the answer tokens.
No external model or pretrained weights are downloaded.
"""
import argparse
import json
import random
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence

from brain.tokenizer import ByteTokenizer, EOS_ID, PAD_ID
from brain.transformer import AIConfig, TKCAI


def load_examples(path, tokenizer, max_seq_len):
    examples = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row.get("prompt"), str) or not isinstance(row.get("answer"), str):
            raise ValueError(f"Line {line_number}: expected string prompt and answer")
        prompt = "Người dùng: " + row["prompt"].strip() + "\nTrợ lý:"
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        answer_ids = tokenizer.encode(" " + row["answer"].strip(), add_special_tokens=False) + [EOS_ID]
        if not answer_ids or len(answer_ids) >= max_seq_len:
            raise ValueError(f"Line {line_number}: answer is too long for max_seq_len")
        # Keep the complete answer and the most recent part of a long prompt.
        prompt_ids = prompt_ids[-(max_seq_len - len(answer_ids)):]
        input_ids = prompt_ids + answer_ids
        labels = [-100] * len(prompt_ids) + answer_ids
        examples.append((torch.tensor(input_ids, dtype=torch.long),
                         torch.tensor(labels, dtype=torch.long)))
    if len(examples) < 2:
        raise ValueError("Need at least two valid JSONL examples for train/validation.")
    return examples


def make_batch(examples, indices, device):
    xs = [examples[i][0] for i in indices]
    ys = [examples[i][1] for i in indices]
    x = pad_sequence(xs, batch_first=True, padding_value=PAD_ID)
    y = pad_sequence(ys, batch_first=True, padding_value=-100)
    return x.to(device), y.to(device)


@torch.no_grad()
def evaluate(model, examples, indices, batch_size, device):
    model.eval()
    losses = []
    for start in range(0, len(indices), batch_size):
        x, y = make_batch(examples, indices[start:start + batch_size], device)
        _, loss = model(x, y)
        losses.append(float(loss.item()))
    model.train()
    return sum(losses) / max(1, len(losses))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/tkc_ai_latest.pt")
    parser.add_argument("--data", default="data/reasoning_examples.jsonl")
    parser.add_argument("--output", default="checkpoints/tkc_ai_instruct.pt")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 1 or args.eval_every < 1:
        raise ValueError("steps, batch-size and eval-every must be positive")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = AIConfig()
    for key, value in state["config"].items():
        setattr(config, key, value)
    tokenizer = ByteTokenizer()
    examples = load_examples(args.data, tokenizer, config.max_seq_len)
    order = list(range(len(examples)))
    random.shuffle(order)
    split = max(1, min(len(order) - 1, int(len(order) * 0.9)))
    train_ids, val_ids = order[:split], order[split:]

    model = TKCAI(config).to(device)
    model.load_state_dict(state["model"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    print(f"Device: {device}; examples: {len(examples)}; train: {len(train_ids)}; validation: {len(val_ids)}")
    print("Instruction tuning an existing TKC-AI checkpoint; prompt tokens are excluded from loss.")

    model.train()
    for step in range(1, args.steps + 1):
        chosen = random.choices(train_ids, k=min(args.batch_size, len(train_ids)))
        x, y = make_batch(examples, chosen, device)
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            val_loss = evaluate(model, examples, val_ids, args.batch_size, device)
            print(f"step {step}/{args.steps} train_loss={loss.item():.4f} val_loss={val_loss:.4f}")
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model": model.state_dict(),
                "config": vars(config),
                "step": step,
                "validation_loss": val_loss,
                "training_stage": "instruction_tuning",
                "base_checkpoint": str(args.checkpoint),
            }, output)


if __name__ == "__main__":
    main()
