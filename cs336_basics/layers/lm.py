import torch
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor

from cs336_basics.layers.embedding import Embedding
from cs336_basics.layers.linear import Linear
from cs336_basics.layers.rmsnorm import RmsNorm
from cs336_basics.layers.rope import RoPE
from cs336_basics.layers.transformer import TransformerBlock


class TransformerLM(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.embedding = Embedding(num_embeddings=vocab_size, embedding_dim=d_model, device=device, dtype=dtype)
        self.blocks = nn.Sequential()

        rope = RoPE(theta=theta, d_k=d_model // num_heads, max_seq_len=context_length, device=device)
        for _ in range(num_layers):
            self.blocks.append(
                TransformerBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff, rope=rope, device=device, dtype=dtype)
            )

        self.final_norm = RmsNorm(d_model=d_model, device=device, dtype=dtype)
        self.final_linear = Linear(in_features=d_model, out_features=vocab_size, device=device, dtype=dtype)

    def forward(
        self,
        in_indices: Int[Tensor, " batch_size sequence_length"],
    ) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
        in_features = self.blocks.forward(self.embedding.forward(in_indices))
        return self.final_linear.forward(self.final_norm.forward(in_features))
