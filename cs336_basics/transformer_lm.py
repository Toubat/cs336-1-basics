import torch
from jaxtyping import Float, Int
from torch import Tensor, nn

from cs336_basics.nn import Embedding, Linear, RMSNorm, RoPEConfig, TransformerBlock


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        theta: float,
        device: torch.device | None = None,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.theta = theta

        self.token_embeddings = Embedding(vocab_size, d_model, device=device)
        self.layers = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, d_ff, device=device) for _ in range(num_layers)]
        )
        self.ln_final = RMSNorm(d_model, device=device)
        self.lm_head = Linear(d_model, vocab_size, device=device)

    def forward(
        self,
        token_ids: Int[Tensor, "bs seq_len"],
    ) -> Float[Tensor, "bs seq_len d_model"]:
        x = self.token_embeddings(token_ids)  # (bs, seq_len, d_model)

        rope_config = RoPEConfig(theta=self.theta, d_k=self.d_model // self.num_heads, max_seq_len=self.context_length)
        for layer in self.layers:
            x = layer(x, rope_config=rope_config)

        x = self.lm_head(self.ln_final(x))  # (bs, seq_len, vocab_size)

        return x
