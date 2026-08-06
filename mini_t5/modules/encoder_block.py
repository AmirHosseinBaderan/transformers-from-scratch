import torch
from torch import nn


from mini_t5.modules.multi_head_attention import MultiHeadAttention
from mini_t5.modules.feed_forward import FeedForward
from mini_t5.modules.layer_norm import LayerNorm



class EncoderBlock(nn.Module):

    def __init__(
        self,
        embedding_dim,
        num_heads,
        ff_hidden_dim,
        dropout=0.1,
    ):
        super().__init__()


        self.attention = MultiHeadAttention(
            embedding_dim,
            num_heads,
            dropout
        )


        self.feed_forward = FeedForward(
            embedding_dim,
            ff_hidden_dim,
            dropout
        )


        self.norm1 = LayerNorm(
            embedding_dim
        )


        self.norm2 = LayerNorm(
            embedding_dim
        )


        self.dropout = nn.Dropout(
            dropout
        )



    def forward(
        self,
        x,
        mask=None,
    ):


        # Self Attention

        attention_output = self.attention(
            x,
            x,
            x,
            mask
        )


        x = self.norm1(
            x +
            self.dropout(
                attention_output
            )
        )



        # Feed Forward

        ff_output = self.feed_forward(
            x
        )


        x = self.norm2(
            x +
            self.dropout(
                ff_output
            )
        )


        return x