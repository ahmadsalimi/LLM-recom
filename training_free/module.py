from typing import List, Tuple

import torch
from pytorch_lightning import LightningModule
from torch import nn

from data.product.io.io import VectorIO


class TrainingFreeModule(LightningModule):

    def __init__(self, model: nn.Module, vector_io: VectorIO):
        super().__init__()
        self.model = model
        self.vector_io = vector_io
        self.vector_io.initialize_read()
        self.product_indices = self.vector_io.get_all_indices()
        self.product_vectors = torch.stack([self.vector_io.get(id_, locale)
                                            for id_, locale in self.product_indices])       # [N, D]
        self.id_to_index = {(id_, locale): i
                            for i, (id_, locale) in enumerate(self.product_indices)}

    def forward(self, batch: List[str]) -> torch.Tensor:
        """
        Arguments:
            batch: List[str], list of formatted sessions

        Returns:
            output: Tensor, shape ``[B, D]``.
        """
        return self.model(batch)

    def test_step(self, batch: List[Tuple[str, str, str]], batch_idx: int, dataloader_idx: int = 0) -> None:
        texts, gt_ids, gt_locales = zip(*batch)
        gt_indices = [self.id_to_index[(id_, locale)] for id_, locale in zip(gt_ids, gt_locales)]   # [B]
        output = self(texts)    # [B, D]

        # MRR
        output = output.unsqueeze(1)    # [B, 1, D]
        similarity = torch.cosine_similarity(output, self.product_vectors, dim=-1)    # [B, N]
        similarity = similarity.cpu().numpy()
        ranks = similarity.argsort(axis=-1, descending=True) + 1    # [B, N]
        gt_ranks = ranks[torch.arange(len(ranks)), gt_indices]    # [B]
        MRR = (1 / gt_ranks).mean()
        self.log('MRR', MRR, prog_bar=True)
