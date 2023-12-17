import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch

from data.product.io.io import VectorIO


class NumpyVectorIO(VectorIO):

    def __init__(self, directory: str, dtype: np.dtype = np.float32):
        super().__init__(directory)
        self.dtype = dtype

    def get(self, id_: str, locale: str) -> torch.Tensor:
        return torch.from_numpy(np.load(f'{self.directory}/{id_}_{locale}.npy'))

    def get_all_indices(self) -> List[Tuple[str, str]]:
        return [(f.split('_')[0], f.split('_')[1].split('.')[0])
                for f in os.listdir(self.directory)]

    def write_batch(self, ids: List[str], locales: List[str], vectors: torch.Tensor):
        for id_, locale, vector in zip(ids, locales, vectors):
            np.save(f'{self.directory}/{id_}_{locale}.npy', vector.numpy().astype(self.dtype))
