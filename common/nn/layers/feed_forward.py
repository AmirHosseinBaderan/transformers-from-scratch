from __future__ import annotations

import torch.nn as nn


class FeedForward(nn.Module):

    def __init__(
            self,
            embedding_dim: int,
            dropout: float = 0.1,
    ):
        super().__init__()


        self.network = nn.Sequential(

            nn.Linear(
                embedding_dim,
                embedding_dim * 4,
            ),

            nn.GELU(),

            nn.Linear(
                embedding_dim * 4,
                embedding_dim,
            ),

            nn.Dropout(
                dropout
            )
        )


    def forward(self, x):

        return self.network(x)