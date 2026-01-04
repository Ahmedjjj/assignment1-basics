import torch
import torch.nn as nn
from einops import reduce


class RmsNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.weights = nn.Parameter(torch.empty(size=(d_model,), device=device, dtype=dtype))
        nn.init.uniform_(tensor=self.weights, a=0, b=1)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype

        x = x.to(torch.float32)
        x_squared = torch.square(x)
        norms = torch.sqrt(
            reduce(x_squared, "batch_size sequence_length d_model -> batch_size sequence_length 1", "mean") + self.eps
        )
        result = (x / norms) * self.weights

        return result.to(in_dtype)
