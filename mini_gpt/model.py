from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.nn.blocks.decoder_block import DecoderBlock
from common.nn.layers.input_embedding import InputEmbedding
from common.nn.layers.layer_norm import LayerNorm


class MiniGPT(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        embedding_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float = 0.1,
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.block_size = block_size
        self.embedding_dim = embedding_dim
        self._use_gradient_checkpointing = use_gradient_checkpointing

        self.embedding = InputEmbedding(
            vocab_size=vocab_size,
            block_size=block_size,
            embedding_dim=embedding_dim,
        )

        self.blocks = nn.ModuleList(
            [
                DecoderBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    use_gradient_checkpointing=use_gradient_checkpointing,
                )
                for _ in range(num_layers)
            ]
        )

        self.norm = LayerNorm(
            embedding_dim
        )

        self.lm_head = nn.Linear(
            embedding_dim,
            vocab_size,
            bias=False,
        )

        self.apply(self._init_weights)

    def _init_weights(
        self,
        module: nn.Module,
    ) -> None:

        if isinstance(module, nn.Linear):

            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02,
            )

            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):

            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02,
            )

    def gradient_checkpointing_enable(self) -> None:
        """
        Enable gradient checkpointing for all decoder blocks.
        This trades compute for memory by recomputing activations during backward pass.
        """
        self._use_gradient_checkpointing = True
        for block in self.blocks:
            block.use_gradient_checkpointing = True

    def gradient_checkpointing_disable(self) -> None:
        """
        Disable gradient checkpointing for all decoder blocks.
        """
        self._use_gradient_checkpointing = False
        for block in self.blocks:
            block.use_gradient_checkpointing = False

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:

        x = self.embedding(input_ids)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        logits = self.lm_head(x)

        loss = None

        if targets is not None:

            batch_size, sequence_length, vocab_size = logits.shape

            loss = F.cross_entropy(
                logits.reshape(
                    batch_size * sequence_length,
                    vocab_size,
                ),
                targets.reshape(
                    batch_size * sequence_length,
                ),
            )

        return logits, loss

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        do_sample: bool = False,
    ) -> torch.Tensor:

        if temperature <= 0:
            raise ValueError(
                "temperature must be greater than zero."
            )

        was_training = self.training
        self.eval()

        for _ in range(max_new_tokens):

            context = input_ids[:, -self.block_size:]

            logits, _ = self(context)

            logits = logits[:, -1, :]

            logits = logits / temperature

            probabilities = torch.softmax(
                logits,
                dim=-1,
            )

            if do_sample:

                next_token = torch.multinomial(
                    probabilities,
                    num_samples=1,
                )

            else:

                next_token = torch.argmax(
                    probabilities,
                    dim=-1,
                    keepdim=True,
                )

            input_ids = torch.cat(
                (
                    input_ids,
                    next_token,
                ),
                dim=1,
            )

        self.train(was_training)

        return input_ids
