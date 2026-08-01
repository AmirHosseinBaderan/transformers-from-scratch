from __future__ import annotations

import torch.nn as nn


class LanguageModelLoss(nn.Module):
    """
    Cross entropy loss for autoregressive language modeling.
    """

    def __init__(self):
        super().__init__()

        self.loss = nn.CrossEntropyLoss()


    def forward(
            self,
            logits,
            targets,
    ):

        """
        logits:
            (batch, sequence, vocab_size)

        targets:
            (batch, sequence)
        """

        batch_size, seq_len, vocab_size = logits.shape


        logits = logits.view(
            batch_size * seq_len,
            vocab_size,
        )


        targets = targets.view(
            batch_size * seq_len,
        )


        return self.loss(
            logits,
            targets,
        )