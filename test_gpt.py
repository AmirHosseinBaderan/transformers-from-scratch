import torch

from common.configs.model_config import ModelConfig
from mini_gpt.model import MiniGPT
from common.data.vocabulary import Vocabulary

from common.utils.logger import logger


vocabulary = Vocabulary.load(
    ModelConfig.VOCAB_PATH
)
logger.info(
    "Vocabulary size: %s",
    vocabulary.size
)

model = MiniGPT(
    vocab_size=500,
    block_size=128,
    embedding_dim=128,
    num_heads=4,
    num_layers=4,
)


x = torch.randint(
    0,
    500,
    (32,128)
)


logits = model(x)


logger.info("Logits shape: %s", logits.shape)