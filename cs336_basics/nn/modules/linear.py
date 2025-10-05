import math

import torch
from einops import einsum
from jaxtyping import Float
from torch import Tensor, nn


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        std = math.sqrt(2 / (in_features + out_features))
        weights = torch.empty((out_features, in_features), device=device, dtype=dtype)
        nn.init.trunc_normal_(weights, mean=0.0, std=std, a=-3 * std, b=3 * std)

        self.weights = nn.Parameter(weights)

    def forward(self, x: Float[Tensor, " ... in_features"]) -> Float[Tensor, " ... out_features"]:
        return einsum(x, self.weights, " ... in_features, out_features in_features -> ... out_features")
