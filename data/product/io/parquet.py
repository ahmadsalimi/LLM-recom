from typing import List, Tuple

import pandas as pd
import torch

from data.product.io.io import VectorIO


class ParquetVectorIO(VectorIO):

    def __init__(self, directory: str, chunk_size: int = 2048) -> None:
        super().__init__(directory)
        self.chunk_size = chunk_size
        self.current_chunk_idx = 0
        self.current_chunk = pd.DataFrame(columns=['id', 'locale', 'vector'])
        self.__data = None
        self.__pid_and_locale_to_index = None

    def initialize_read(self) -> torch.Tensor:
        if self.__data is not None:
            return
        self.__data = pd.read_parquet(self.directory)
        self.__pid_and_locale_to_index = {(pid, locale): i
                                          for i, (pid, locale)
                                          in enumerate(zip(self.__data['id'].tolist(),
                                                           self.__data['locale'].tolist()))}

    def __check_read(self) -> None:
        assert self.__data is not None, 'Call initialize_read() first'

    def get(self, id_: str, locale: str) -> torch.Tensor:
        self.__check_read()
        vector = self.__data.iloc[self.__pid_and_locale_to_index[(id_, locale)]]['vector']
        return torch.tensor(vector)

    def get_all_indices(self) -> List[Tuple[str, str]]:
        self.__check_read()
        return [(pid, locale) for pid, locale in zip(self.__data['id'].tolist(),
                                                     self.__data['locale'].tolist())]

    def __store_chunk(self):
        self.current_chunk.to_parquet(f'{self.directory}/chunk_{self.current_chunk_idx}.parquet')
        self.current_chunk_idx += 1
        self.current_chunk = pd.DataFrame(columns=['id', 'locale', 'vector'])

    def write_batch(self, ids: List[str], locales: List[str], vectors: torch.Tensor):
        self.current_chunk = self.current_chunk.append(pd.DataFrame({
            'id': ids,
            'locale': locales,
            'vector': vectors.tolist()
        }), ignore_index=True)
        if len(self.current_chunk) >= self.chunk_size:
            self.__store_chunk()

    def flush(self):
        if len(self.current_chunk) > 0:
            self.__store_chunk()
