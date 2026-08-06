import torch
import math
from torch import nn


class PositionalEncoding(nn.Module):

    def __init__(
        self,
        embedding_dim,
        max_length=512,
        dropout=0.1,
    ):
        super().__init__()

        self.dropout = nn.Dropout(dropout)


        position = torch.arange(
            max_length
        ).unsqueeze(1)


        div_term = torch.exp(
            torch.arange(
                0,
                embedding_dim,
                2
            )
            *
            (-math.log(10000.0)
             /
             embedding_dim)
        )


        pe = torch.zeros(
            max_length,
            embedding_dim
        )


        pe[:,0::2] = torch.sin(
            position * div_term
        )


        pe[:,1::2] = torch.cos(
            position * div_term
        )


        pe = pe.unsqueeze(0)


        self.register_buffer(
            "pe",
            pe
        )


    def forward(
        self,
        x,
    ):

        x = x + self.pe[:,:x.size(1)]

        return self.dropout(x)