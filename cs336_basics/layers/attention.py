import numpy as np
import torch
import torch.nn as nn
from einops import einsum, rearrange
from jaxtyping import Bool, Float, Int
from torch import Tensor

from cs336_basics.layers.linear import Linear
from cs336_basics.layers.rope import RoPE


def softmax(input: Tensor, dim: int) -> torch.Tensor:
    input -= torch.max(input=input, dim=dim, keepdim=True)[0]
    input = torch.exp(input=input)
    return input / torch.sum(input, dim=dim, keepdim=True)


def attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, "... d_v"]:
    K = rearrange(K, "... keys d_k -> ... d_k keys")
    d_k = Q.size(-1)
    Q_T_K = einsum(Q, K, "... queries d_k, ... d_k keys -> ... queries keys") / np.sqrt(d_k)

    if mask is not None:
        Q_T_K[~mask] = float("-inf")

    return softmax(Q_T_K, dim=-1) @ V


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        rope: RoPE | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0

        self.linear = Linear(
            in_features=d_model, out_features=3 * d_model, device=device, dtype=dtype
        )  # 3 is for Q, K, V
        self.linear_o = Linear(in_features=d_model, out_features=d_model, device=device, dtype=dtype)

        self.num_heads = num_heads
        self.d_model = d_model
        self.rope = rope

    def forward(
        self,
        in_features: Float[Tensor, " ... sequence_length d_in"],
        token_positions: Int[Tensor, " ... sequence_length"] | None = None,
    ) -> Float[Tensor, " ... sequence_length d_in"]:
        Q, K, V = rearrange(
            self.linear.forward(in_features),
            " ... sequence_length (d1 h d_v) -> d1 ... h sequence_length d_v",
            d1=3,
            h=self.num_heads,
        )
        sequence_length = Q.size(-2)
        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(0, sequence_length, dtype=torch.long)
            Q = self.rope.forward(Q, token_positions=token_positions)
            K = self.rope.forward(K, token_positions=token_positions)

        mask = torch.tril(torch.ones((*Q.shape[:-1], sequence_length), dtype=torch.bool))
        new_features = rearrange(
            attention(Q=Q, K=K, V=V, mask=mask), "... h sequence_length d_v -> ... sequence_length (h d_v)"
        )
        return self.linear_o.forward(new_features)
