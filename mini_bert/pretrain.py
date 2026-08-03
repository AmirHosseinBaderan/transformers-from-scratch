from __future__ import annotations

import torch
import torch.nn as nn

from torch.utils.data import DataLoader


class BERTPreTrainer:

    def __init__(
        self,
        model,
        mlm_head,
        optimizer,
        device,
    ):

        self.model = model
        self.mlm_head = mlm_head
        self.optimizer = optimizer
        self.device = device

        self.loss_fn = nn.CrossEntropyLoss(
            ignore_index=-100
        )


    def train_step(
        self,
        batch,
    ):

        self.model.train()
        self.mlm_head.train()


        input_ids = batch["input_ids"].to(
            self.device
        )

        labels = batch["labels"].to(
            self.device
        )


        self.optimizer.zero_grad()


        hidden_states = self.model(
            input_ids
        )


        logits = self.mlm_head(
            hidden_states
        )


        loss = self.loss_fn(
            logits.view(
                -1,
                logits.size(-1)
            ),
            labels.view(
                -1
            ),
        )


        loss.backward()


        self.optimizer.step()


        return loss.item()



    def train_epoch(
        self,
        loader: DataLoader,
    ):

        total_loss = 0


        for batch in loader:

            loss = self.train_step(
                batch
            )

            total_loss += loss


        return total_loss / len(loader)