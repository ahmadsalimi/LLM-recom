import os
from abc import ABC, abstractmethod
from typing import List

import torch


class VectorIO(ABC):

    def __init__(self, directory: str):
        self.directory = directory
        os.makedirs(self.directory, exist_ok=True)

    def initialize_read(self):
        pass

    @abstractmethod
    def get(self, id_: str, locale: str) -> torch.Tensor:
        pass

    @abstractmethod
    def write_batch(self, ids: List[str], locales: List[str], vectors: torch.Tensor):
        pass

    def flush(self):
        pass
