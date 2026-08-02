"""
MiniGPT configuration - imports from the shared ModelConfig.
All settings are auto-detected based on available hardware (CPU/GPU).
"""

from common.configs.model_config import ModelConfig

# Re-export for backward compatibility
GPTConfig = ModelConfig
