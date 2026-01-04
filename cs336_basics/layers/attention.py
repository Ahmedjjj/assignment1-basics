import torch


def softmax(input: torch.Tensor, dim:int) -> torch.Tensor:
    input -= torch.max(input=input, dim=dim, keepdim=True)[0]
    input = torch.exp(input=input)
    return input / torch.sum(input, dim=dim, keepdim=True)
