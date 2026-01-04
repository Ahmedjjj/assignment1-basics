import torch
import torch.nn as nn


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        self.weights = nn.Parameter(torch.empty(size=(num_embeddings, embedding_dim), device=device, dtype=dtype))
        nn.init.trunc_normal_(tensor=self.weights, mean=0, std=1, a=-3, b=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights[x]
