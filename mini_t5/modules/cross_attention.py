import torch
from torch import nn

from mini_t5.modules.multi_head_attention import MultiHeadAttention



class CrossAttention(nn.Module):

    def __init__(
        self,
        embedding_dim,
        num_heads,
        dropout=0.1,
    ):
        super().__init__()


        self.attention = MultiHeadAttention(
            embedding_dim,
            num_heads,
            dropout
        )


    def forward(
        self,
        decoder_states,
        encoder_states,
        mask=None,
    ):

        return self.attention(
            decoder_states,
            encoder_states,
            encoder_states,
            mask
        )