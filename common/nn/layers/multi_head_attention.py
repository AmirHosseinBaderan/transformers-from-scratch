from __future__ import annotations

import torch
import torch.nn as nn

from common.nn.layers.attention import SelfAttention



class MultiHeadAttention(nn.Module):

    def __init__(
            self,
            embedding_dim: int,
            num_heads: int,
            dropout: float = 0.1,
    ):
        super().__init__()


        if embedding_dim % num_heads != 0:
            raise ValueError(
                "embedding_dim must be divisible by num_heads"
            )


        self.embedding_dim = embedding_dim

        self.num_heads = num_heads

        self.head_dim = (
            embedding_dim // num_heads
        )


        self.qkv_projection = nn.Linear(
            embedding_dim,
            embedding_dim * 3,
        )


        self.output_projection = nn.Linear(
            embedding_dim,
            embedding_dim,
        )


        self.dropout = nn.Dropout(
            dropout
        )


    def forward(self, x):

        batch, seq, dim = x.shape


        qkv = self.qkv_projection(x)


        q, k, v = torch.chunk(
            qkv,
            3,
            dim=-1,
        )


        q = q.view(
            batch,
            seq,
            self.num_heads,
            self.head_dim,
        )


        k = k.view(
            batch,
            seq,
            self.num_heads,
            self.head_dim,
        )


        v = v.view(
            batch,
            seq,
            self.num_heads,
            self.head_dim,
        )

        q = q.transpose(1,2)

        k = k.transpose(1,2)

        v = v.transpose(1,2)

        scores = torch.matmul(
            q,
            k.transpose(-2,-1)
        )

        scores /= self.head_dim ** 0.5
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

        attention = torch.softmax(
            scores,
            dim=-1,
        )

        attention = self.dropout(
            attention
        )

        out = torch.matmul(
            attention,
            v,
        )

        out = out.transpose(
            1,
            2,
        )

        out = out.contiguous().view(
            batch,
            seq,
            dim,
        )

        return self.output_projection(out)