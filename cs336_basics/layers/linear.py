import numpy as np
import torch
import torch.nn as nn
from einops import einsum


class Linear(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()

        self.weights = nn.Parameter(torch.empty(size=(out_features, in_features), device=device, dtype=dtype))
        std = np.sqrt(2 / (in_features + out_features))
        nn.init.trunc_normal_(tensor=self.weights, mean=0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weights, "... d_in, d_out d_in -> ... d_out")
