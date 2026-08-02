import torch

from common.nn.blocks.decoder_block import DecoderBlock

from common.utils.logger import logger


block = DecoderBlock(
    embedding_dim=128,
    num_heads=4,
)


x = torch.randn(
    32,
    128,
    128
)


y = block(x)


logger.info("Output shape: %s", y.shape)