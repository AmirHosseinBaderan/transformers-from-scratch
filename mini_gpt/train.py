from pathlib import Path

import torch
from torch.utils.data import DataLoader

from mini_gpt.model import MiniGPT

from common.data.dataset import TextDataset

from common.training.losses import (
    LanguageModelLoss,
)

from mini_gpt.config import GPTConfig
from common.data.vocabulary import Vocabulary
from common.configs.model_config import ModelConfig

from mini_gpt.train_one_epoch import train_one_epoch

def main():


    dataset = TextDataset(
        Path(
            "common/data/processed/train.bin"
        ),
        GPTConfig.BLOCK_SIZE,
    )


    loader = DataLoader(
        dataset,
        batch_size=GPTConfig.BATCH_SIZE,
        shuffle=True,
    )

    vocabulary = Vocabulary.load(
        ModelConfig.VOCAB_PATH
    )

    model = MiniGPT(
        vocab_size=vocabulary.size,
        block_size=GPTConfig.BLOCK_SIZE,
        embedding_dim=GPTConfig.EMBEDDING_DIM,
        num_heads=GPTConfig.NUM_HEADS,
        num_layers=GPTConfig.NUM_LAYERS,
        dropout=GPTConfig.DROPOUT,
    )


    model.to(GPTConfig.DEVICE)


    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=GPTConfig.LEARNING_RATE,
    )


    criterion = LanguageModelLoss()


    for epoch in range(
        GPTConfig.EPOCHS
    ):
        avg_loss = train_one_epoch(
            model,
            loader,
            criterion,
            optimizer,
            GPTConfig.DEVICE,
        )

        print(
            f"Epoch {epoch+1} | Loss {avg_loss:.4f}"
        )



if __name__ == "__main__":
    main()
