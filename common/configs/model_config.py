from pathlib import Path

import torch


def _get_gpu_memory_gb() -> float | None:
    """
    Get available GPU memory in GB.
    Returns None if CUDA is not available.
    """
    if not torch.cuda.is_available():
        return None

    try:
        total_memory = torch.cuda.get_device_properties(0).total_memory
        reserved_memory = torch.cuda.memory_reserved(0)
        available_memory = total_memory - reserved_memory
        return available_memory / (1024 ** 3)
    except Exception:
        return None


def _get_cpu_memory_gb() -> float:
    """
    Get available system RAM in GB.
    """
    try:
        import psutil
        return psutil.virtual_memory().available / (1024 ** 3)
    except ImportError:
        return 8.0


def _get_optimal_config() -> dict:
    """
    Determine optimal configuration based on available hardware.
    ULTRA LOW MEMORY defaults for small memory systems.
    """
    config = {}

    is_cuda = torch.cuda.is_available()

    if is_cuda:
        gpu_memory_gb = _get_gpu_memory_gb()
        config["DEVICE"] = "cuda"

        if gpu_memory_gb is not None and gpu_memory_gb < 2.0:
            config.update({
                "BATCH_SIZE": 2,
                "GRADIENT_ACCUMULATION_STEPS": 16,
                "USE_MIXED_PRECISION": True,
                "USE_GRADIENT_CHECKPOINTING": True,
                "GRADIENT_CLIP_NORM": 1.0,
                "NUM_WORKERS": 0,
                "PREFETCH_FACTOR": None,
                "PIN_MEMORY": False,
            })
        elif gpu_memory_gb is not None and gpu_memory_gb < 4.0:
            config.update({
                "BATCH_SIZE": 4,
                "GRADIENT_ACCUMULATION_STEPS": 8,
                "USE_MIXED_PRECISION": True,
                "USE_GRADIENT_CHECKPOINTING": True,
                "GRADIENT_CLIP_NORM": 1.0,
                "NUM_WORKERS": 0,
                "PREFETCH_FACTOR": None,
                "PIN_MEMORY": False,
            })
        elif gpu_memory_gb is not None and gpu_memory_gb < 8.0:
            config.update({
                "BATCH_SIZE": 8,
                "GRADIENT_ACCUMULATION_STEPS": 4,
                "USE_MIXED_PRECISION": True,
                "USE_GRADIENT_CHECKPOINTING": True,
                "GRADIENT_CLIP_NORM": 1.0,
                "NUM_WORKERS": 0,
                "PREFETCH_FACTOR": None,
                "PIN_MEMORY": False,
            })
        else:
            config.update({
                "BATCH_SIZE": 16,
                "GRADIENT_ACCUMULATION_STEPS": 2,
                "USE_MIXED_PRECISION": True,
                "USE_GRADIENT_CHECKPOINTING": False,
                "GRADIENT_CLIP_NORM": 1.0,
                "NUM_WORKERS": 0,
                "PREFETCH_FACTOR": None,
                "PIN_MEMORY": True,
            })
    else:
        # CPU configuration - always ultra low memory
        cpu_memory_gb = _get_cpu_memory_gb()
        config["DEVICE"] = "cpu"

        if cpu_memory_gb < 4.0:
            config.update({
                "BATCH_SIZE": 2,
                "GRADIENT_ACCUMULATION_STEPS": 16,
                "USE_MIXED_PRECISION": False,
                "USE_GRADIENT_CHECKPOINTING": True,
                "GRADIENT_CLIP_NORM": 1.0,
                "NUM_WORKERS": 0,
                "PREFETCH_FACTOR": None,
                "PIN_MEMORY": False,
            })
        elif cpu_memory_gb < 8.0:
            config.update({
                "BATCH_SIZE": 4,
                "GRADIENT_ACCUMULATION_STEPS": 8,
                "USE_MIXED_PRECISION": False,
                "USE_GRADIENT_CHECKPOINTING": True,
                "GRADIENT_CLIP_NORM": 1.0,
                "NUM_WORKERS": 0,
                "PREFETCH_FACTOR": None,
                "PIN_MEMORY": False,
            })
        else:
            config.update({
                "BATCH_SIZE": 8,
                "GRADIENT_ACCUMULATION_STEPS": 4,
                "USE_MIXED_PRECISION": False,
                "USE_GRADIENT_CHECKPOINTING": True,
                "GRADIENT_CLIP_NORM": 1.0,
                "NUM_WORKERS": 0,
                "PREFETCH_FACTOR": None,
                "PIN_MEMORY": False,
            })

    return config


# Auto-detect optimal configuration
_optimal_config = _get_optimal_config()


class ModelConfig:
    """
    Model configuration with auto-detected optimal settings for the current hardware.
    """

    # Vocabulary path
    VOCAB_PATH = Path(
        "common/data/processed/vocab.json"
    )

    # Model architecture
    EMBEDDING_DIM = 128
    BLOCK_SIZE = 128
    NUM_HEADS = 4
    NUM_LAYERS = 4
    DROPOUT = 0.1

    # Device (auto-detected)
    DEVICE = _optimal_config["DEVICE"]

    # Training settings (auto-configured)
    BATCH_SIZE = _optimal_config["BATCH_SIZE"]
    GRADIENT_ACCUMULATION_STEPS = _optimal_config["GRADIENT_ACCUMULATION_STEPS"]
    LEARNING_RATE = 3e-4
    EPOCHS = 10

    # Memory optimization settings (auto-configured)
    USE_MIXED_PRECISION = _optimal_config["USE_MIXED_PRECISION"]
    USE_GRADIENT_CHECKPOINTING = _optimal_config["USE_GRADIENT_CHECKPOINTING"]
    GRADIENT_CLIP_NORM = _optimal_config["GRADIENT_CLIP_NORM"]

    # DataLoader settings (auto-configured)
    PIN_MEMORY = _optimal_config["PIN_MEMORY"]
    NUM_WORKERS = _optimal_config["NUM_WORKERS"]
    PREFETCH_FACTOR = _optimal_config.get("PREFETCH_FACTOR")

    # Checkpoint settings
    CHECKPOINT_DIR = "mini_gpt/checkpoints"
    KEEP_LAST_N_CHECKPOINTS = 3

    # Early stopping settings
    EARLY_STOPPING_PATIENCE = 5
    EARLY_STOPPING_MIN_DELTA = 0.0

    # TensorBoard settings
    TENSORBOARD_LOG_DIR = "mini_gpt/runs"
