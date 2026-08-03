from __future__ import annotations

import math

import torch
import torch.nn as nn


class SelfAttention(nn.Module):

    def __init__(
        self,
        head_dim: int,
        dropout: float = 0.1,
        causal: bool = False,
    ):
        super().__init__()

        self.head_dim = head_dim
        self.causal = causal

        self.query = nn.Linear(
            head_dim,
            head_dim,
        )

        self.key = nn.Linear(
            head_dim,
            head_dim,
        )

        self.value = nn.Linear(
            head_dim,
            head_dim,
        )

        self.dropout = nn.Dropout(
            dropout
        )


    def forward(
        self,
        x,
        causal: bool | None = None,
    ):

        batch, seq, dim = x.shape

        q = self.query(x)

        k = self.key(x)

        v = self.value(x)

        scores = torch.matmul(
            q,
            k.transpose(-2, -1)
        )

        scores = scores / math.sqrt(dim)


        use_causal = (
            self.causal
            if causal is None
            else causal
        )


        if use_causal:

            mask = torch.triu(
                torch.ones(
                    seq,
                    seq,
                    device=x.device,
                ),
                diagonal=1,
            )

            scores = scores.masked_fill(
                mask == 1,
                float("-inf")
            )


        weights = torch.softmax(
            scores,
            dim=-1
        )

        weights = self.dropout(
            weights
        )


        return torch.matmul(
            weights,
            v
        )