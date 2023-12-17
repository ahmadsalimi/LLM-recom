from typing import Tuple, List

import pytorch_lightning as pl
import torch

from data.product.io.io import VectorIO


class SaveOutput(pl.callbacks.BasePredictionWriter):

    def __init__(self, vector_io: VectorIO) -> None:
        super().__init__(write_interval='batch')
        self.vector_io = vector_io

    def on_predict_batch_end(self,
                             trainer: pl.Trainer,
                             pl_module: pl.LightningModule,
                             outputs: torch.Tensor,
                             batch: List[Tuple[str, str]],
                             batch_idx: int,
                             dataloader_idx: int) -> None:
        ids, locales, _ = zip(*batch)
        self.vector_io.write_batch(ids, locales, outputs)

    def on_predict_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs: List[torch.Tensor]) -> None:
        self.vector_io.flush()
