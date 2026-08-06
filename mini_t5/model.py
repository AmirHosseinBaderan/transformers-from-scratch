import torch
from torch import nn


from encoder import T5Encoder
from decoder import T5Decoder



class MiniT5(nn.Module):

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


        self.encoder = T5Encoder(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ff_hidden_dim=ff_hidden_dim,
            max_length=max_length,
            dropout=dropout,
        )


        self.decoder = T5Decoder(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ff_hidden_dim=ff_hidden_dim,
            max_length=max_length,
            dropout=dropout,
        )


        self.output_projection = nn.Linear(
            embedding_dim,
            vocab_size
        )



    def forward(
        self,
        encoder_input_ids,
        decoder_input_ids,
        encoder_mask=None,
        decoder_mask=None,
        cross_mask=None,
    ):


        encoder_output = self.encoder(
            encoder_input_ids,
            encoder_mask
        )


        decoder_output = self.decoder(
            decoder_input_ids,
            encoder_output,
            decoder_mask,
            cross_mask
        )


        logits = self.output_projection(
            decoder_output
        )


        return logits