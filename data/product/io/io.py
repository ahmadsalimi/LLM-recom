import os
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional

import torch


class VectorIO(ABC):

    def __init__(self, directory: str, include_locale: Optional[List[str]] = None):
        self.directory = directory
        os.makedirs(self.directory, exist_ok=True)
        self.include_locale = include_locale

    def initialize_read(self):
        pass

    @abstractmethod
    def get(self, id_: str, locale: str) -> torch.Tensor:
        pass

    @abstractmethod
    def get_all_indices(self) -> List[Tuple[str, str]]:
        pass

    @abstractmethod
    def write_batch(self, ids: List[str], locales: List[str], vectors: torch.Tensor):
        pass

    def flush(self):
        pass
