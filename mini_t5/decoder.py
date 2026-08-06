import torch
from torch import nn


from modules.embedding import TokenEmbedding
from modules.positional_encoding import PositionalEncoding
from modules.decoder_block import DecoderBlock
from modules.layer_norm import LayerNorm



class T5Decoder(nn.Module):

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
                DecoderBlock(
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
        decoder_input_ids,
        encoder_output,
        self_mask=None,
        cross_mask=None,
    ):


        x = self.embedding(
            decoder_input_ids
        )


        x = self.position(
            x
        )


        for layer in self.layers:

            x = layer(
                x,
                encoder_output,
                self_mask,
                cross_mask
            )


        x = self.norm(
            x
        )


        return x