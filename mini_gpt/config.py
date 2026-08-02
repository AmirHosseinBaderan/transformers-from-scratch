import torch


class GPTConfig:
    DEVICE = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    BLOCK_SIZE = 128

    EMBEDDING_DIM = 128

    NUM_HEADS = 4

    NUM_LAYERS = 4

    DROPOUT = 0.1

    BATCH_SIZE = 32

    LEARNING_RATE = 3e-4

    EPOCHS = 10

    # Checkpoint settings
    CHECKPOINT_DIR = "mini_gpt/checkpoints"
    KEEP_LAST_N_CHECKPOINTS = 3

    # Early stopping settings
    EARLY_STOPPING_PATIENCE = 5
    EARLY_STOPPING_MIN_DELTA = 0.0

    # TensorBoard settings
    TENSORBOARD_LOG_DIR = "mini_gpt/runs"
