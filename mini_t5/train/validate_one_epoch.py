from tqdm import tqdm

import gc
import torch


def validate_one_epoch(
    model,
    loader,
    criterion,
    device,
    tokenizer,
):
    """
    Run one epoch of T5 validation with ultra low-memory optimizations.

    Args:
        model: The T5 model.
        loader: DataLoader for validation data.
        criterion: Loss function.
        device: Device to run on.
        tokenizer: Tokenizer for vocab_size.

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
        for batch in progress_bar:
            encoder_input_ids = batch["encoder_input_ids"].to(
                device,
                non_blocking=True,
            )
            decoder_input_ids = batch["decoder_input_ids"].to(
                device,
                non_blocking=True,
            )
            labels = batch["labels"].to(
                device,
                non_blocking=True,
            )

            # Generate padding masks
            encoder_mask = (
                encoder_input_ids != tokenizer.pad_id
            ).unsqueeze(1).unsqueeze(2)
            cross_mask = encoder_mask

            logits = model(
                encoder_input_ids,
                decoder_input_ids,
                encoder_mask=encoder_mask,
                cross_mask=cross_mask,
            )
            loss = criterion(
                logits.view(-1, tokenizer.vocab_size),
                labels.view(-1),
            )

            total_loss += loss.item()
            num_batches += 1

            progress_bar.set_postfix(
                loss=f"{total_loss / num_batches:.4f}"
            )

            # AGGRESSIVE MEMORY CLEANUP after each batch
            del encoder_input_ids, decoder_input_ids, labels, logits, loss

            # Periodic CUDA cache cleanup every 50 batches
            if torch.cuda.is_available() and (num_batches + 1) % 50 == 0:
                torch.cuda.empty_cache()

    # Final cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return total_loss / num_batches
