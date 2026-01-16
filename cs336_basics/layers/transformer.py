import torch
import torch.nn as nn
from jaxtyping import Float

from cs336_basics.layers.attention import MultiHeadSelfAttention
from cs336_basics.layers.rmsnorm import RmsNorm
from cs336_basics.layers.rope import RoPE
from cs336_basics.layers.swiglu import SwiGLU


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        rope: RoPE | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.input_rms_norm = RmsNorm(d_model=d_model, device=device, dtype=dtype)
        self.attention = MultiHeadSelfAttention(
            d_model=d_model, num_heads=num_heads, rope=rope, device=device, dtype=dtype
        )
        self.ffn_rms_norm = RmsNorm(d_model=d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)

    def forward(
        self,
        in_features: Float[torch.Tensor, " batch sequence_length d_model"],
    ) -> Float[torch.Tensor, " batch sequence_length d_model"]:
        ffn_input = self.attention.forward(self.input_rms_norm.forward(in_features)) + in_features
        return self.ffn.forward(self.ffn_rms_norm.forward(ffn_input)) + ffn_input
