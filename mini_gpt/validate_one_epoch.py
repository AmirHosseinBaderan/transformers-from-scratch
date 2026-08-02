from tqdm import tqdm

import gc
import torch


def validate_one_epoch(
    model,
    loader,
    criterion,
    device,
):
    """
    Run one epoch of validation with ultra low-memory optimizations.

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
        for batch_idx, (x, y) in enumerate(progress_bar):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            logits, loss = model(x, y)

            total_loss += loss.item()
            num_batches += 1

            progress_bar.set_postfix(
                loss=f"{total_loss / num_batches:.4f}"
            )

            # AGGRESSIVE MEMORY CLEANUP after each batch
            del x, y, logits, loss

            # Periodic CUDA cache cleanup every 50 batches
            if torch.cuda.is_available() and (batch_idx + 1) % 50 == 0:
                torch.cuda.empty_cache()

    # Final cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return total_loss / num_batches
