from typing import List

import torch
from torch import nn

from data.product.io.io import VectorIO


class MRR(nn.Module):

    def __init__(self, vector_io: VectorIO):
        super().__init__()
        product_indices = vector_io.get_all_indices()
        self.product_vectors = torch.stack([vector_io.get(id_, locale)
                                            for id_, locale in product_indices])       # [N, D]
        self.id_to_index = {(id_, locale): i
                            for i, (id_, locale) in enumerate(product_indices)}

    def forward(self, y_hat: torch.Tensor, gt_ids: List[str], gt_locales: List[str]) -> torch.Tensor:
        """
        Arguments:
            y_hat: Tensor, shape ``[B, D]``.
            gt_ids: List[str], list of ground truth product IDs of length ``B``.
            gt_locales: List[str], list of ground truth product locales of length ``B``.

        Returns:
            loss: Tensor, shape ``[]``.
        """
        gt_indices = [self.id_to_index[(id_, locale)] for id_, locale in zip(gt_ids, gt_locales)]
        y_hat = y_hat.unsqueeze(1)    # [B, 1, D]
        similarity = torch.cosine_similarity(y_hat, self.product_vectors, dim=-1)    # [B, N]
        similarity = similarity.cpu().numpy()
        ranks = similarity.argsort(axis=-1, descending=True) + 1    # [B, N]
        gt_ranks = ranks[torch.arange(len(ranks)), gt_indices]    # [B]
        MRR = (1 / gt_ranks).mean()
        return MRR
