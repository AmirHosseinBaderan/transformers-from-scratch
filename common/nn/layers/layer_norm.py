from __future__ import annotations

import torch.nn as nn


class LayerNorm(nn.Module):

    def __init__(
            self,
            embedding_dim: int,
            eps: float = 1e-5,
    ):
        super().__init__()


        self.norm = nn.LayerNorm(
            embedding_dim,
            eps=eps,
        )


    def forward(self, x):

        return self.norm(x)