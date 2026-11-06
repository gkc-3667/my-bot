import torch
import torch.nn as nn
import torch.nn.functional as F


class AIConfig:
    """Configuration for the first TKC-AI prototype."""

    def __init__(self):
        self.vocab_size = 259
        self.max_seq_len = 512
        self.d_model = 256
        self.n_heads = 8
        self.n_layers = 6
        self.dropout = 0.1


class CausalAttention(nn.Module):
    """Multi-head self-attention that cannot see future tokens."""

    def __init__(self, config):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            config.d_model,
            config.n_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        seq_len = x.size(1)
        causal_mask = torch.triu(
            torch.ones(
                seq_len,
                seq_len,
                device=x.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )

        output, _ = self.attention(
            x, x, x,
            attn_mask=causal_mask,
            need_weights=False,
        )
        return self.dropout(output)


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.attention = CausalAttention(config)
        self.feed_forward = nn.Sequential(
            nn.Linear(config.d_model, config.d_model * 4),
            nn.GELU(),
            nn.Linear(config.d_model * 4, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.feed_forward(self.norm2(x))
        return x


class TKCAI(nn.Module):
    """Decoder-only language-model skeleton with randomly initialized weights."""

    def __init__(self, config):
        super().__init__()
        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.d_model,
        )
        self.position_embedding = nn.Embedding(
            config.max_seq_len,
            config.d_model,
        )
        self.blocks = nn.ModuleList([
            TransformerBlock(config)
            for _ in range(config.n_layers)
        ])
        self.final_norm = nn.LayerNorm(config.d_model)
        self.output = nn.Linear(
            config.d_model,
            config.vocab_size,
            bias=False,
        )

        # Weight tying: share token embedding and output projection weights.
        self.output.weight = self.token_embedding.weight

    def forward(self, tokens, targets=None):
        _, seq_len = tokens.shape
        if seq_len > self.position_embedding.num_embeddings:
            raise ValueError(
                f"Sequence length {seq_len} exceeds "
                f"maximum {self.position_embedding.num_embeddings}"
            )

        positions = torch.arange(seq_len, device=tokens.device)
        x = (
            self.token_embedding(tokens)
            + self.position_embedding(positions)
        )

        for block in self.blocks:
            x = block(x)

        logits = self.output(self.final_norm(x))
        loss = None

        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )

        return logits, loss


if __name__ == "__main__":
    config = AIConfig()
    model = TKCAI(config)

    sample = torch.randint(
        0,
        config.vocab_size,
        (2, 32),
    )
    logits, _ = model(sample)

    print("TKC-AI Transformer initialized.")
    print("Output shape:", tuple(logits.shape))
    print(
        "Parameters:",
        sum(parameter.numel() for parameter in model.parameters()),
    )
    print("Note: this model is randomly initialized and not trained yet.")
