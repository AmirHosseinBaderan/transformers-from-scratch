from tqdm import tqdm

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    gradient_accumulation_steps: int = 1,
    use_mixed_precision: bool = False,
    gradient_clip_norm: float | None = None,
):
    """
    Run one epoch of training with memory optimizations.

    Args:
        model: The model to train.
        loader: DataLoader for training data.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Device to run on.
        gradient_accumulation_steps: Number of steps to accumulate gradients before stepping.
        use_mixed_precision: Whether to use mixed precision training.
        gradient_clip_norm: Maximum norm for gradient clipping.

    Returns:
        Average training loss for the epoch.
    """
    model.train()

    total_loss = 0
    num_batches = 0

    # Initialize gradient scaler for mixed precision training
    scaler = GradScaler() if use_mixed_precision else None

    progress_bar = tqdm(
        loader,
        desc="Training",
        leave=True,
    )

    optimizer.zero_grad()

    for batch_idx, (x, y) in enumerate(progress_bar):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        # Mixed precision forward pass
        if use_mixed_precision:
            with autocast():
                logits = model(x)
                loss = criterion(logits, y)
                # Scale loss for gradient accumulation
                loss = loss / gradient_accumulation_steps
        else:
            logits = model(x)
            loss = criterion(logits, y)
            # Scale loss for gradient accumulation
            loss = loss / gradient_accumulation_steps

        # Backward pass
        if use_mixed_precision:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Step optimizer only after accumulating gradients
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            # Unscale gradients before clipping (required for mixed precision)
            if use_mixed_precision:
                scaler.unscale_(optimizer)

            # Gradient clipping
            if gradient_clip_norm is not None:
                nn.utils.clip_grad_norm_(
                    model.parameters(),
                    gradient_clip_norm,
                )

            # Optimizer step
            if use_mixed_precision:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad()

        # Track loss (use the unscaled loss for logging)
        total_loss += loss.item() * gradient_accumulation_steps
        num_batches += 1

        progress_bar.set_postfix(
            loss=f"{total_loss / num_batches:.4f}"
        )

    # Handle remaining gradients if batch count is not divisible by accumulation steps
    if num_batches % gradient_accumulation_steps != 0:
        if use_mixed_precision:
            scaler.unscale_(optimizer)

        if gradient_clip_norm is not None:
            nn.utils.clip_grad_norm_(
                model.parameters(),
                gradient_clip_norm,
            )

        if use_mixed_precision:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        optimizer.zero_grad()

    return total_loss / num_batches
