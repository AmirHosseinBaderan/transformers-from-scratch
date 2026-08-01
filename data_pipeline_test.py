from pathlib import Path

from torch.utils.data import DataLoader

from common.data.dataset import TextDataset
from common.configs.data_config import DataConfig


dataset = TextDataset(
    Path(
        "common/data/processed/train.bin"
    ),
    block_size=DataConfig.BLOCK_SIZE,
)


loader = DataLoader(
    dataset,
    batch_size=DataConfig.TRAIN_BATCH_SIZE,
    shuffle=True,
)


x, y = next(iter(loader))


print(x.shape)
print(y.shape)