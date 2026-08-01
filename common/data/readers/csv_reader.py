from __future__ import annotations

from pathlib import Path
from typing import Iterator

import csv


class CSVReader:
    """
    Streaming CSV reader.

    Reads text samples one by one without loading
    the entire dataset into memory.
    """

    def __init__(
            self,
            path: Path,
            column: str = "text",
    ):
        self._path = path
        self._column = column


    def read(self) -> Iterator[str]:
        """
        Stream text rows from CSV file.
        """

        with self._path.open(
                "r",
                encoding="utf-8",
                newline="",
        ) as file:

            reader = csv.DictReader(file)

            if self._column not in reader.fieldnames:
                raise ValueError(
                    f"Column '{self._column}' not found."
                )


            for row in reader:

                text = row.get(
                    self._column
                )

                if text is None:
                    continue

                text = text.strip()

                if not text:
                    continue

                yield text