from pathlib import Path

import gc
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler

from common.data.masked_dataset import MaskedDataset
from common.data.preprocessing.masker import Masker
from common.training.checkpoint import CheckpointManager
from common.training.early_stopping import EarlyStopping
from common.training.tensorboard_logger import TensorBoardLogger

from common.utils.logger import logger
from common.configs.model_config import ModelConfig
from common.data.vocabulary import Vocabulary

from mini_bert.model import MiniBERT
from mini_bert.tasks import MaskedLMHead
from mini_bert.trainer import BERTTrainer


def build_trainer() -> BERTTrainer:

    vocabulary = Vocabulary.load(
        ModelConfig.VOCAB_PATH
    )

    masker = Masker(
        vocabulary=vocabulary,
        mask_probability=0.15,
    )

    train_dataset = MaskedDataset(
        path=Path("common/data/processed/train.bin"),
        block_size=ModelConfig.BLOCK_SIZE,
        masker=masker,
    )

    val_dataset = MaskedDataset(
        path=Path("common/data/processed/validation.bin"),
        block_size=ModelConfig.BLOCK_SIZE,
        masker=masker,
    )

    train_sampler = RandomSampler(
        train_dataset,
        replacement=True,
        num_samples=ModelConfig.STEPS_PER_EPOCH * ModelConfig.BATCH_SIZE,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=ModelConfig.BATCH_SIZE,
        sampler=train_sampler,
        pin_memory=ModelConfig.PIN_MEMORY and torch.cuda.is_available(),
        num_workers=0,
        prefetch_factor=None,
        persistent_workers=False,
        drop_last=True,
        multiprocessing_context=None,
    )

    val_sampler = RandomSampler(
        val_dataset,
        replacement=True,
        num_samples=ModelConfig.VAL_STEPS * ModelConfig.BATCH_SIZE,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=ModelConfig.BATCH_SIZE,
        sampler=val_sampler,
        pin_memory=ModelConfig.PIN_MEMORY and torch.cuda.is_available(),
        num_workers=0,
        prefetch_factor=None,
        persistent_workers=False,
        drop_last=True,
        multiprocessing_context=None,
    )

    model = MiniBERT(
        vocab_size=len(vocabulary),
        block_size=ModelConfig.BLOCK_SIZE,
        embedding_dim=ModelConfig.EMBEDDING_DIM,
        num_layers=ModelConfig.NUM_LAYERS,
        dropout=ModelConfig.DROPOUT,
    )

    model.to(ModelConfig.DEVICE)

    mlm_head = MaskedLMHead(
        embedding_dim=ModelConfig.EMBEDDING_DIM,
        vocab_size=len(vocabulary),
    )

    mlm_head.to(ModelConfig.DEVICE)

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(mlm_head.parameters()),
        lr=ModelConfig.LEARNING_RATE,
    )

    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    checkpoint_manager = CheckpointManager(
        checkpoint_dir="mini_bert/checkpoints",
        model=model,
        optimizer=optimizer,
        keep_last_n=ModelConfig.KEEP_LAST_N_CHECKPOINTS,
    )

    early_stopping = EarlyStopping(
        patience=ModelConfig.EARLY_STOPPING_PATIENCE,
        min_delta=ModelConfig.EARLY_STOPPING_MIN_DELTA,
        mode="min",
    )

    tb_logger = TensorBoardLogger("mini_bert/runs")

    return BERTTrainer(
        model=model,
        mlm_head=mlm_head,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        device=ModelConfig.DEVICE,
        checkpoint_manager=checkpoint_manager,
        early_stopping=early_stopping,
        tb_logger=tb_logger,
        epochs=ModelConfig.EPOCHS,
    )


def main():
    logger.info("=" * 60)
    logger.info("MiniBERT Training Configuration")
    logger.info("=" * 60)
    logger.info(f"Device: {ModelConfig.DEVICE}")
    logger.info(f"Batch Size: {ModelConfig.BATCH_SIZE}")
    logger.info(f"Steps Per Epoch: {ModelConfig.STEPS_PER_EPOCH}")
    logger.info(f"Validation Steps: {ModelConfig.VAL_STEPS}")
    logger.info(f"Gradient Accumulation Steps: {ModelConfig.GRADIENT_ACCUMULATION_STEPS}")
    logger.info(f"Effective Batch Size: {ModelConfig.BATCH_SIZE * ModelConfig.GRADIENT_ACCUMULATION_STEPS}")
    logger.info(f"Mixed Precision: {ModelConfig.USE_MIXED_PRECISION}")
    logger.info(f"Gradient Checkpointing: {ModelConfig.USE_GRADIENT_CHECKPOINTING}")
    logger.info(f"Gradient Clip Norm: {ModelConfig.GRADIENT_CLIP_NORM}")
    logger.info(f"DataLoader Workers: 0 (single process)")
    logger.info(f"Pin Memory: {ModelConfig.PIN_MEMORY}")
    logger.info(f"Persistent Workers: False")
    logger.info("=" * 60)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        logger.info(f"Initial GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")

    trainer = build_trainer()

    start_epoch = 0
    latest_checkpoint = trainer.checkpoint_manager.load_latest()
    if latest_checkpoint is not None:
        start_epoch = latest_checkpoint["epoch"]
        trainer.early_stopping.load_state_dict(
            latest_checkpoint.get(
                "early_stopping_state",
                trainer.early_stopping.state_dict(),
            )
        )
        logger.info(f"Resumed from epoch {start_epoch}")

    try:
        trainer.train(start_epoch=start_epoch)
        logger.info("Training complete!")
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logger.error("Out of memory error! Try:")
            logger.error("  1. Reduce BATCH_SIZE in config.py")
            logger.error("  2. Increase GRADIENT_ACCUMULATION_STEPS")
            logger.error("  3. Enable USE_GRADIENT_CHECKPOINTING")
            logger.error("  4. Reduce NUM_LAYERS or EMBEDDING_DIM")
            logger.error("  5. Reduce BLOCK_SIZE")
            logger.error("  6. Close other applications")
        raise
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    main()
