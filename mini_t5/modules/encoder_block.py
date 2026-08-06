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
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()

        self.use_gradient_checkpointing = use_gradient_checkpointing

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

    def _attention_forward(self, x, mask):
        """Forward pass for attention block (used for gradient checkpointing)."""
        return self.norm1(x + self.dropout(self.attention(x, x, x, mask)))

    def _feed_forward_forward(self, x):
        """Forward pass for feed forward block (used for gradient checkpointing)."""
        return self.norm2(x + self.dropout(self.feed_forward(x)))

    def forward(
        self,
        x,
        mask=None,
    ):

        # Self Attention
        if self.use_gradient_checkpointing and self.training:
            x = torch.utils.checkpoint.checkpoint(
                self._attention_forward,
                x,
                mask,
                use_reentrant=False,
            )
        else:
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
        if self.use_gradient_checkpointing and self.training:
            x = torch.utils.checkpoint.checkpoint(
                self._feed_forward_forward,
                x,
                use_reentrant=False,
            )
        else:
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