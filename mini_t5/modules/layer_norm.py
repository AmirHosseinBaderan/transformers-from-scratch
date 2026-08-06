import torch
from torch import nn


class LayerNorm(nn.Module):

    def __init__(
        self,
        embedding_dim,
        eps=1e-6,
    ):
        super().__init__()

        self.gamma = nn.Parameter(
            torch.ones(embedding_dim)
        )

        self.beta = nn.Parameter(
            torch.zeros(embedding_dim)
        )

        self.eps = eps


    def forward(
        self,
        x,
    ):

        mean = x.mean(
            dim=-1,
            keepdim=True
        )

        variance = x.var(
            dim=-1,
            keepdim=True,
            unbiased=False
        )


        x = (
            x - mean
        ) / torch.sqrt(
            variance + self.eps
        )


        return (
            self.gamma * x
            +
            self.beta
        )