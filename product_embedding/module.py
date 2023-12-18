from typing import List, Dict

import torch
from pytorch_lightning import LightningModule
from torch import nn


class ProductEmbeddingModule(LightningModule):

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, batch: List[str]) -> torch.Tensor:
        """
        Arguments:
            batch: List[str], list of formatted product strings.

        Returns:
            output: Tensor, shape ``[B, D]``.
        """
        return self.model(batch)

    def predict_step(self, products: Dict[str, List[str]], batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        return self(products['text'])
