from __future__ import annotations
from pathlib import Path
import pandas as pd

from common.data.readers.reader import Reader


class CSVReader(Reader):

    def __init__(
        self,
        path: Path,
        column: str = "text",
    ) -> None:
        self._path = path
        self._column = column

    def read(self) -> list[str]:
        dataframe = pd.read_csv(self._path)

        if self._column not in dataframe.columns:
            raise ValueError(
                f"Column '{self._column}' not found."
            )

        return (
            dataframe[self._column]
            .dropna()
            .astype(str)
            .tolist()
        )