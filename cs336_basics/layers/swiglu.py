import torch
import torch.nn as nn

from cs336_basics.layers.linear import Linear


def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    def __init__(
        self, d_model: int, d_ff: int, device: torch.device | None = None, dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()
        self.w1 = Linear(in_features=d_model, out_features=d_ff, device=device, dtype=dtype)
        self.w2 = Linear(in_features=d_ff, out_features=d_model, device=device, dtype=dtype)
        self.w3 = Linear(in_features=d_model, out_features=d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2.forward(silu(self.w1.forward(x)) * self.w3.forward(x))
