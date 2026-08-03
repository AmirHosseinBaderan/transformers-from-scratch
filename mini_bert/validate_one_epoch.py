from tqdm import tqdm

import gc
import torch


def validate_one_epoch(
    model,
    mlm_head,
    loader,
    criterion,
    device,
):
    """
    Run one epoch of BERT MLM validation with ultra low-memory optimizations.

    Args:
        model: The BERT model.
        mlm_head: The MaskedLMHead for producing logits from hidden states.
        loader: DataLoader for validation data.
        criterion: Loss function.
        device: Device to run on.

    Returns:
        Average validation loss for the epoch.
    """
    model.eval()
    mlm_head.eval()

    total_loss = 0
    num_batches = 0

    progress_bar = tqdm(
        loader,
        desc="Validation",
        leave=True,
    )

    with torch.no_grad():
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(
                device,
                non_blocking=True,
            )
            labels = batch["labels"].to(
                device,
                non_blocking=True,
            )

            hidden_states = model(input_ids)
            logits = mlm_head(hidden_states)
            loss = criterion(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
            )

            total_loss += loss.item()
            num_batches += 1

            progress_bar.set_postfix(
                loss=f"{total_loss / num_batches:.4f}"
            )

            # AGGRESSIVE MEMORY CLEANUP after each batch
            del input_ids, labels, hidden_states, logits, loss

            # Periodic CUDA cache cleanup every 50 batches
            if torch.cuda.is_available() and (num_batches + 1) % 50 == 0:
                torch.cuda.empty_cache()

    # Final cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return total_loss / num_batches
