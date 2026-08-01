from pathlib import Path

import torch
from torch.utils.data import DataLoader

from model import MiniGPT

from common.data.dataset import TextDataset

from common.training.losses import (
    LanguageModelLoss,
)

from config import GPTConfig
from common.data.vocabulary import Vocabulary
from common.configs.model_config import ModelConfig



device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

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


    model.to(device)


    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=GPTConfig.LEARNING_RATE,
    )


    criterion = LanguageModelLoss()



    model.train()


    for epoch in range(
        GPTConfig.EPOCHS
    ):

        total_loss = 0


        for x,y in loader:


            x = x.to(device)

            y = y.to(device)


            logits = model(x)


            loss = criterion(
                logits,
                y,
            )


            optimizer.zero_grad()


            loss.backward()


            optimizer.step()


            total_loss += loss.item()


        print(
            f"Epoch {epoch+1} | Loss {total_loss / len(loader):.4f}"
        )



if __name__ == "__main__":
    main()