import torch

from common.nn.layers.multi_head_attention import MultiHeadAttention

from common.utils.logger import logger


model = MultiHeadAttention(
    embedding_dim=128,
    num_heads=4,
)


x = torch.randn(
    32,
    128,
    128
)


y = model(x)


logger.info("Output shape: %s", y.shape)