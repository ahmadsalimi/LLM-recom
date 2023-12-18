from typing import List, Dict

import torch
from pytorch_lightning import LightningModule
from torch import nn

from data.product.io.io import VectorIO
from metric.mrr import MRR


class TrainingFreeModule(LightningModule):

    def __init__(self, model: nn.Module, vector_io: VectorIO):
        super().__init__()
        self.model = model
        vector_io.initialize_read()
        self.mrr = MRR(vector_io)

    def forward(self, batch: List[str]) -> torch.Tensor:
        """
        Arguments:
            batch: List[str], list of formatted sessions

        Returns:
            output: Tensor, shape ``[B, D]``.
        """
        return self.model(batch)

    def test_step(self, batch: Dict[str, List[str]], batch_idx: int, dataloader_idx: int = 0) -> None:
        texts, gt_ids, gt_locales = batch['text'], batch['gt_id'], batch['gt_locale']
        output = self(texts)    # [B, D]
        mrr = self.mrr(output, gt_ids, gt_locales)
        self.log('MRR', mrr, prog_bar=True)
