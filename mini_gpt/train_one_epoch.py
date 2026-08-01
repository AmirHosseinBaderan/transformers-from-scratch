from tqdm import tqdm

import torch


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
):
    model.train()

    total_loss = 0

    progress_bar = tqdm(
        loader,
        desc="Training",
        leave=True,
    )

    for x, y in progress_bar:
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

        progress_bar.set_postfix(
            loss=f"{total_loss / len(loader):.4f}"
        )

    return total_loss / len(loader)
