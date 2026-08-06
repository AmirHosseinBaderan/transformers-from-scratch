import torch
from torch import nn
import math



class MultiHeadAttention(nn.Module):

    def __init__(
        self,
        embedding_dim,
        num_heads,
        dropout=0.1,
    ):
        super().__init__()


        assert embedding_dim % num_heads == 0


        self.embedding_dim = embedding_dim

        self.num_heads = num_heads


        self.head_dim = (
            embedding_dim // num_heads
        )


        self.query = nn.Linear(
            embedding_dim,
            embedding_dim
        )

        self.key = nn.Linear(
            embedding_dim,
            embedding_dim
        )

        self.value = nn.Linear(
            embedding_dim,
            embedding_dim
        )


        self.output = nn.Linear(
            embedding_dim,
            embedding_dim
        )


        self.dropout = nn.Dropout(
            dropout
        )



    def split_heads(
        self,
        x,
    ):

        batch_size = x.size(0)

        seq_len = x.size(1)


        x = x.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim,
        )


        return x.transpose(
            1,
            2
        )



    def combine_heads(
        self,
        x,
    ):

        batch_size = x.size(0)


        x = x.transpose(
            1,
            2
        )


        return x.contiguous().view(
            batch_size,
            -1,
            self.embedding_dim
        )



    def forward(
        self,
        query,
        key,
        value,
        mask=None,
    ):


        Q = self.query(query)

        K = self.key(key)

        V = self.value(value)



        Q = self.split_heads(Q)

        K = self.split_heads(K)

        V = self.split_heads(V)



        scores = torch.matmul(
            Q,
            K.transpose(-2,-1)
        )


        scores = scores / math.sqrt(
            self.head_dim
        )



        if mask is not None:

            scores = scores.masked_fill(
                mask == 0,
                -1e9
            )



        attention = torch.softmax(
            scores,
            dim=-1
        )


        attention = self.dropout(
            attention
        )



        output = torch.matmul(
            attention,
            V
        )



        output = self.combine_heads(
            output
        )


        return self.output(output)