import os
from typing import Tuple, List

import pandas as pd
import pytorch_lightning as pl
import torch


class SaveOutput(pl.callbacks.BasePredictionWriter):

    def __init__(self, directory: str, chunk_size: int = 2048) -> None:
        super().__init__(write_interval='batch')
        self.directory = directory
        self.chunk_size = chunk_size
        self.current_chunk_idx = 0
        self.current_chunk = pd.DataFrame(columns=['id', 'vector'])

    def on_predict_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self.current_chunk_idx = 0
        self.current_chunk = pd.DataFrame(columns=['id', 'vector'])

    def __store_chunk(self):
        os.makedirs(self.directory)
        self.current_chunk.to_parquet(f'{self.directory}/chunk_{self.current_chunk_idx}.parquet')
        self.current_chunk_idx += 1
        self.current_chunk = pd.DataFrame(columns=['id', 'vector'])

    def on_predict_batch_end(self,
                             trainer: pl.Trainer,
                             pl_module: pl.LightningModule,
                             outputs: torch.Tensor,
                             batch: List[Tuple[str, str]],
                             batch_idx: int,
                             dataloader_idx: int) -> None:
        ids, _ = zip(*batch)
        self.current_chunk = self.current_chunk.append(pd.DataFrame({
            'id': ids,
            'vector': outputs.tolist()
        }), ignore_index=True)
        if len(self.current_chunk) >= self.chunk_size:
            self.__store_chunk()

    def on_predict_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs: List[torch.Tensor]) -> None:
        if len(self.current_chunk) > 0:
            self.__store_chunk()
