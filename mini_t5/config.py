"""
MiniT5 configuration - imports from the shared ModelConfig and overrides
T5-specific settings such as checkpoint directory, TensorBoard log directory,
and dataset paths.
"""

from pathlib import Path

from common.configs.model_config import ModelConfig


class T5Config:
    """
    MiniT5 configuration with T5-specific overrides.
    Inherits hardware-auto-detected settings from ModelConfig.
    """

    # Dataset paths
    TRAIN_CSV_PATH = Path("common/data/raw/train.csv")
    VAL_CSV_PATH = Path("common/data/raw/validation.csv")

    # Model architecture (override if needed)
    EMBEDDING_DIM = ModelConfig.EMBEDDING_DIM
    NUM_HEADS = ModelConfig.NUM_HEADS
    NUM_LAYERS = ModelConfig.NUM_LAYERS
    DROPOUT = ModelConfig.DROPOUT
    MAX_LENGTH = 64

    # Training settings (from ModelConfig)
    DEVICE = ModelConfig.DEVICE
    BATCH_SIZE = ModelConfig.BATCH_SIZE
    GRADIENT_ACCUMULATION_STEPS = ModelConfig.GRADIENT_ACCUMULATION_STEPS
    LEARNING_RATE = ModelConfig.LEARNING_RATE
    EPOCHS = ModelConfig.EPOCHS

    # Memory optimization (from ModelConfig)
    USE_MIXED_PRECISION = ModelConfig.USE_MIXED_PRECISION
    USE_GRADIENT_CHECKPOINTING = ModelConfig.USE_GRADIENT_CHECKPOINTING
    GRADIENT_CLIP_NORM = ModelConfig.GRADIENT_CLIP_NORM

    # DataLoader settings (from ModelConfig)
    PIN_MEMORY = ModelConfig.PIN_MEMORY
    NUM_WORKERS = ModelConfig.NUM_WORKERS
    PREFETCH_FACTOR = ModelConfig.PREFETCH_FACTOR

    # Checkpoint settings (T5-specific)
    CHECKPOINT_DIR = "mini_t5/checkpoints"
    KEEP_LAST_N_CHECKPOINTS = ModelConfig.KEEP_LAST_N_CHECKPOINTS

    # Early stopping settings (from ModelConfig)
    EARLY_STOPPING_PATIENCE = ModelConfig.EARLY_STOPPING_PATIENCE
    EARLY_STOPPING_MIN_DELTA = ModelConfig.EARLY_STOPPING_MIN_DELTA

    # TensorBoard settings (T5-specific)
    TENSORBOARD_LOG_DIR = "mini_t5/runs"
