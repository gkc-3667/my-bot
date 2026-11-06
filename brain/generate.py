"""Generate text from a trained TKC-AI checkpoint."""
import argparse

import torch

from brain.tokenizer import BOS_ID, ByteTokenizer
from brain.transformer import AIConfig, TKCAI


@torch.no_grad()
def generate(model, prompt_ids, max_new_tokens, temperature, top_k, device):
    model.eval()
    tokens = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        context = tokens[:, -model.position_embedding.num_embeddings:]
        logits, _ = model(context)
        next_logits = logits[:, -1, :] / max(temperature, 1e-5)

        if top_k > 0:
            values, _ = torch.topk(
                next_logits, min(top_k, next_logits.size(-1))
            )
            cutoff = values[:, -1].unsqueeze(-1)
            next_logits = next_logits.masked_fill(
                next_logits < cutoff, float("-inf")
            )

        probabilities = torch.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probabilities, num_samples=1)
        tokens = torch.cat([tokens, next_token], dim=1)

    return tokens[0].tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/tkc_ai_latest.pt")
    parser.add_argument("--prompt", default="Xin chào")
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(
        args.checkpoint, map_location=device, weights_only=False
    )
    config = AIConfig()
    for key, value in checkpoint["config"].items():
        setattr(config, key, value)

    model = TKCAI(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    tokenizer = ByteTokenizer()
    prompt_ids = tokenizer.encode(args.prompt, add_special_tokens=False)
    if not prompt_ids:
        prompt_ids = [BOS_ID]
    output_ids = generate(
        model, prompt_ids, args.max_new_tokens,
        args.temperature, args.top_k, device
    )
    print(tokenizer.decode(output_ids))


if __name__ == "__main__":
    main()
