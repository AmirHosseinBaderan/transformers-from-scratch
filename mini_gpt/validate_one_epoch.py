from tqdm import tqdm

import torch


def validate_one_epoch(
    model,
    loader,
    criterion,
    device,
):
    """
    Run one epoch of validation.

    Args:
        model: The model to validate.
        loader: DataLoader for validation data.
        criterion: Loss function.
        device: Device to run on.

    Returns:
        Average validation loss for the epoch.
    """
    model.eval()

    total_loss = 0
    num_batches = 0

    progress_bar = tqdm(
        loader,
        desc="Validation",
        leave=True,
    )

    with torch.no_grad():
        for x, y in progress_bar:
            x = x.to(device)
            y = y.to(device)

            logits, loss = model(x, y)

            total_loss += loss.item()
            num_batches += 1

            progress_bar.set_postfix(
                loss=f"{total_loss / num_batches:.4f}"
            )

    return total_loss / num_batches
