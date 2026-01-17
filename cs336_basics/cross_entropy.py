import torch
from einops import reduce
from jaxtyping import Float, Int
from torch import Tensor


def cross_entropy(
    inputs: Float[Tensor, " ... batch_size vocab_size"], targets: Int[Tensor, " ... batch_size"]
) -> Float[Tensor, ""]:
    max_logits = reduce(inputs, "... batch_size vocab_size -> ... batch_size 1", reduction="max")
    inputs -= max_logits
    sum_logits = reduce(torch.exp(inputs), "... batch_size vocab_size -> ... batch_size 1", reduction="sum")
    loss = torch.log(sum_logits) - torch.gather(inputs, dim=-1, index=targets.unsqueeze(-1))
    return loss.mean()
