import torch

from torch.utils.data import DataLoader

from torch import nn

from torch.optim import AdamW


from modules.dataset import TranslationDataset
from modules.tokenizer import CharacterTokenizer

from model import MiniT5



DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)



CSV_PATH = "data/raw/en_fa_translation_dataset.csv"


EPOCHS = 50

BATCH_SIZE = 16

LR = 1e-4


MAX_LENGTH = 64



# ---------------------
# Tokenizer
# ---------------------


tokenizer = CharacterTokenizer()


# فعلاً از CSV می‌خوانیم

import pandas as pd


df = pd.read_csv(
    CSV_PATH
)


texts = []

texts.extend(
    df["source"].astype(str).tolist()
)

texts.extend(
    df["target"].astype(str).tolist()
)


tokenizer.fit(
    texts
)



# ---------------------
# Dataset
# ---------------------


dataset = TranslationDataset(
    CSV_PATH,
    tokenizer,
    MAX_LENGTH
)



loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)



# ---------------------
# Model
# ---------------------


model = MiniT5(
    vocab_size=tokenizer.vocab_size,
    embedding_dim=256,
    num_layers=4,
    num_heads=8,
    ff_hidden_dim=1024,
    max_length=MAX_LENGTH,
).to(DEVICE)



# ---------------------
# Loss
# ---------------------


criterion = nn.CrossEntropyLoss(
    ignore_index=tokenizer.pad_id
)



optimizer = AdamW(
    model.parameters(),
    lr=LR
)



# ---------------------
# Training Loop
# ---------------------


for epoch in range(EPOCHS):

    model.train()


    total_loss = 0


    for batch in loader:


        encoder_input = batch[
            "encoder_input_ids"
        ].to(DEVICE)


        decoder_input = batch[
            "decoder_input_ids"
        ].to(DEVICE)


        labels = batch[
            "labels"
        ].to(DEVICE)



        optimizer.zero_grad()



        logits = model(
            encoder_input,
            decoder_input
        )



        loss = criterion(
            logits.reshape(
                -1,
                tokenizer.vocab_size
            ),

            labels.reshape(
                -1
            )
        )



        loss.backward()


        optimizer.step()



        total_loss += loss.item()



    avg_loss = (
        total_loss /
        len(loader)
    )


    print(
        f"Epoch {epoch+1}/{EPOCHS} "
        f"Loss: {avg_loss:.4f}"
    )