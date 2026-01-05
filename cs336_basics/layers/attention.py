import numpy as np
import torch
from einops import einsum, rearrange
from jaxtyping import Bool, Float
from torch import Tensor


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
