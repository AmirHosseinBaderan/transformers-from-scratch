import torch
from torch import nn


from mini_t5.modules.embedding import TokenEmbedding
from mini_t5.modules.positional_encoding import PositionalEncoding
from mini_t5.modules.encoder_block import EncoderBlock
from mini_t5.modules.layer_norm import LayerNorm



class T5Encoder(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        num_layers,
        num_heads,
        ff_hidden_dim,
        max_length,
        dropout=0.1,
    ):
        super().__init__()


        self.embedding = TokenEmbedding(
            vocab_size,
            embedding_dim
        )


        self.position = PositionalEncoding(
            embedding_dim,
            max_length,
            dropout
        )


        self.layers = nn.ModuleList(
            [
                EncoderBlock(
                    embedding_dim,
                    num_heads,
                    ff_hidden_dim,
                    dropout
                )

                for _ in range(num_layers)
            ]
        )


        self.norm = LayerNorm(
            embedding_dim
        )



    def forward(
        self,
        input_ids,
        mask=None,
    ):


        x = self.embedding(
            input_ids
        )


        x = self.position(
            x
        )


        for layer in self.layers:

            x = layer(
                x,
                mask
            )


        x = self.norm(
            x
        )


        return x