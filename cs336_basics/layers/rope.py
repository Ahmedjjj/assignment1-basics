import torch
import torch.nn as nn
from einops import rearrange
from jaxtyping import Float, Int


class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None) -> None:
        super().__init__()
        assert d_k % 2 == 0

        k = torch.arange(1, d_k // 2 + 1, device=device)
        thetas = torch.arange(max_seq_len, device=device).unsqueeze(-1) / torch.pow(theta, (2 * k - 2) / d_k)
        self.sin = torch.repeat_interleave(torch.sin(thetas), 2, dim=-1)
        self.cos = torch.repeat_interleave(torch.cos(thetas), 2, dim=-1)

    def forward(
        self,
        x: Float[torch.Tensor, " ... sequence_length d_k"],
        token_positions: Int[torch.Tensor, " ... sequence_length"],
    ) -> torch.Tensor:
        sin_mult = self.sin[token_positions]
        cos_mult = self.cos[token_positions]

        return cos_mult * x + sin_mult * _prepare_sin_coord(x)


def _prepare_sin_coord(
    x: Float[torch.Tensor, " ... sequence_length d_k"],
) -> Float[torch.Tensor, " ... sequence_length d_k"]:
    x1, x2 = x[..., ::2], x[..., 1::2]
    stacked = torch.stack((-x2, x1), dim=-1)
    return rearrange(stacked, "... half_sequence_length n -> ... (half_sequence_length n)")
