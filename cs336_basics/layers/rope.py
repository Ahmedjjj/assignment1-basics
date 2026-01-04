import numpy as np
import torch
import torch.nn as nn
from einops import rearrange


def _get_rotation_tensor(theta: float, device: torch.device | None = None) -> torch.Tensor:
    cos = np.cos(theta)
    sin = np.sin(theta)
    return torch.Tensor([[cos, -sin], [sin, cos]], device=device)


class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None) -> None:
        super().__init__()
        assert d_k % 2 == 0

        rot_tensors = []

        for i in range(max_seq_len):
            tensor = torch.block_diag(
                *(
                    _get_rotation_tensor(i / np.pow(theta, (2 * k - 2) / d_k), device=device)
                    for k in range(1, d_k // 2 + 1)
                )
            )
            rot_tensors.append(tensor)

        self.rot_tensor = rearrange(rot_tensors, "b ... -> b ...")
        self.register_buffer(name="rotation_tensors", tensor=self.rot_tensor, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        rotations = self.rot_tensor[token_positions]
        x = rearrange(x, "... seq_len d_k -> ... seq_len d_k 1")
        return rearrange(rotations @ x, "... seq_len d_k 1 -> ... seq_len d_k")
