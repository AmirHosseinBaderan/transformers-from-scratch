import torch
from torch import nn


from mini_t5.modules.multi_head_attention import MultiHeadAttention
from mini_t5.modules.cross_attention import CrossAttention
from mini_t5.modules.feed_forward import FeedForward
from mini_t5.modules.layer_norm import LayerNorm



class DecoderBlock(nn.Module):

    def __init__(
        self,
        embedding_dim,
        num_heads,
        ff_hidden_dim,
        dropout=0.1,
    ):
        super().__init__()


        self.self_attention = MultiHeadAttention(
            embedding_dim,
            num_heads,
            dropout
        )


        self.cross_attention = CrossAttention(
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


        self.norm3 = LayerNorm(
            embedding_dim
        )


        self.dropout = nn.Dropout(
            dropout
        )



    def forward(
        self,
        x,
        encoder_output,
        self_mask=None,
        cross_mask=None,
    ):


        # 1) Masked Self Attention

        attention_output = self.self_attention(
            x,
            x,
            x,
            self_mask
        )


        x = self.norm1(
            x +
            self.dropout(
                attention_output
            )
        )



        # 2) Cross Attention

        cross_output = self.cross_attention(
            x,
            encoder_output,
            cross_mask
        )


        x = self.norm2(
            x +
            self.dropout(
                cross_output
            )
        )



        # 3) Feed Forward

        ff_output = self.feed_forward(
            x
        )


        x = self.norm3(
            x +
            self.dropout(
                ff_output
            )
        )


        return x