import torch
from einops import rearrange
from jaxtyping import Float, Int
from torch import Tensor, nn

from cs336_basics.nn import Linear
from cs336_basics.nn.modules.rope import apply_rope
from cs336_basics.nn.modules.utils import scaled_dot_product_attention


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        device: torch.device | None = None,
    ):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.qkv_proj = Linear(d_model, d_model * 3, device=device)
        self.o_proj = Linear(d_model, d_model, device=device)

    def forward(
        self,
        x: Float[Tensor, "bs seq_len d_model"],
        theta: float | None = None,
        max_seq_len: int | None = None,
        token_positions: Int[Tensor, " ... seq_len"] | None = None,
    ) -> Float[Tensor, "bs seq_len d_model"]:
        qkv = self.qkv_proj(x)  # (bs seq_len d_model * 3)
        q, k, v = rearrange(
            qkv,
            "bs seq_len (three num_heads d_k) -> three bs num_heads seq_len d_k",
            num_heads=self.num_heads,
            d_k=self.d_k,
        )  # (bs, num_heads, seq_len, d_k)

        if theta is not None and max_seq_len is not None and token_positions is not None:
            q = apply_rope(self.d_k, theta, max_seq_len, q, token_positions)
            k = apply_rope(self.d_k, theta, max_seq_len, k, token_positions)

        seq_len = q.shape[-2]
        mask = torch.tril(torch.ones((seq_len, seq_len))).bool()  # (seq_len, seq_len)
        o = scaled_dot_product_attention(q, k, v, mask)  # (bs, num_heads, seq_len, d_k)
        o = rearrange(o, "bs num_heads seq_len d_k -> bs seq_len (num_heads d_k)")

        return self.o_proj(o)
