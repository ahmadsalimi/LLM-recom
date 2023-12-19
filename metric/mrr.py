from typing import List, Optional

import numpy as np
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

    def __init__(self, vector_io: VectorIO, similarity_batch_size: int = 10000, map_vectors: Optional[nn.Module] = None,
                 device: torch.device = torch.device('cpu')):
        super().__init__()
        # self.similarity_batch_size = int(similarity_batch_size)
        product_indices = vector_io.get_all_indices()
        # TODO: this is a hack, we should not use private attributes
        vectors = np.array(vector_io._ParquetVectorIO__data['vector'].tolist())
        if map_vectors is None:
            mapped_vectors = vectors
        else:
            mapped_vectors = map_vectors(
                torch.from_numpy(vectors[:similarity_batch_size]).to(device)).detach().cpu().numpy()
            for i in tqdm(range(similarity_batch_size, len(vectors), similarity_batch_size),
                          desc='Mapping vectors',
                          total=len(vectors) // similarity_batch_size):
                mapped_vectors = np.concatenate((
                    mapped_vectors,
                    map_vectors(torch.from_numpy(vectors[i:i + similarity_batch_size]).to(device)).detach().cpu().numpy(),
                ))
        products = [Product(id=id_, locale=locale, embedding=mapped_vectors[i])
                    for i, (id_, locale) in enumerate(tqdm(product_indices, desc='Loading products',
                                                           total=len(product_indices)))]
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
        queries = [Product(embedding=y_hat[i].detach().cpu().numpy()) for i in range(len(y_hat))]
        results = self.db.search(inputs=DocList[Product](queries), limit=100)
        idx_to_rank = [{
            (r.id, r.locale): i + 1
            for i, r in enumerate(result.matches)
        } for result in results]
        gt_ranks = [ranks.get((id_, locale), float('inf'))
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
