from typing import List, Dict

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
                             batch: Dict[str, List[str]],
                             batch_idx: int,
                             dataloader_idx: int) -> None:
        self.vector_io.write_batch(batch['id'], batch['locale'], outputs)

    def on_predict_epoch_end(self, trainer: pl.Trainer,
                             pl_module: pl.LightningModule,
                             outputs: List[torch.Tensor]) -> None:
        self.vector_io.flush()
