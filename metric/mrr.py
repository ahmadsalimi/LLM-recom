from typing import List

import torch
from torch import nn
from tqdm import tqdm
from docarray import BaseDoc
from docarray.typing import NdArray
from docarray import DocList
from vectordb import InMemoryExactNNVectorDB

from data.product.io.io import VectorIO


class Product(BaseDoc):
    id: str = ''
    locale: str = ''
    embedding: NdArray[1024]


class MRR(nn.Module):

    def __init__(self, vector_io: VectorIO, similarity_batch_size: int = 10000):
        super().__init__()
        # self.similarity_batch_size = int(similarity_batch_size)
        product_indices = vector_io.get_all_indices()
        products = [Product(id=id_, locale=locale, embedding=vector_io.get(id_, locale).numpy())
                    for id_, locale in tqdm(product_indices, desc='Loading products',
                                            total=len(product_indices))]
        self.limit = 1000000
        self.db = InMemoryExactNNVectorDB[Product](workspace='./vectordb-workspace')
        self.db.index(inputs=DocList[Product](products))
        # self.product_vectors = torch.stack([vector_io.get(id_, locale)
        #                                     for id_, locale
        #                                     in tqdm(product_indices, desc='Loading product vectors',
        #                                             total=len(product_indices))])   # [N, D]
        # self.id_to_index = {(id_, locale): i
        #                     for i, (id_, locale) in enumerate(product_indices)}

    def forward(self, y_hat: torch.Tensor, gt_ids: List[str], gt_locales: List[str]) -> torch.Tensor:
        """
        Arguments:
            y_hat: Tensor, shape ``[B, D]``.
            gt_ids: List[str], list of ground truth product IDs of length ``B``.
            gt_locales: List[str], list of ground truth product locales of length ``B``.

        Returns:
            loss: Tensor, shape ``[]``.
        """
        queries = [Product(embedding=y_hat[i].cpu().numpy()) for i in range(len(y_hat))]
        results = self.db.search(inputs=DocList[Product](queries), limit=self.limit)
        idx_to_rank = [{
            (r.id, r.locale): i + 1
            for i, r in enumerate(result.matches)
        } for result in results]
        gt_ranks = [ranks.get((id_, locale), self.limit + 1)
                    for id_, locale, ranks in zip(gt_ids, gt_locales, idx_to_rank)]
        mrr = (1 / torch.tensor(gt_ranks, dtype=torch.float32, device=y_hat.device)).mean()
        return mrr
        #
        # gt_indices = [self.id_to_index[(id_, locale)] for id_, locale in zip(gt_ids, gt_locales)]
        # y_hat = y_hat.unsqueeze(1)  # [B, 1, D]
        # similarity = torch.tensor([], device=y_hat.device)
        # for i in range(0, len(self.product_vectors), self.similarity_batch_size):
        #     similarity = torch.cat((similarity,
        #                             torch.cosine_similarity(
        #                                 y_hat,
        #                                 self.product_vectors[i:i + self.similarity_batch_size].to(y_hat.device),
        #                                 dim=-1)), dim=1)
        # ranks = similarity.argsort(dim=-1, descending=True) + 1    # [B, N]
        # gt_ranks = ranks[torch.arange(len(ranks)), gt_indices]    # [B]
        # mrr = (1 / gt_ranks).mean()
        # return mrr
