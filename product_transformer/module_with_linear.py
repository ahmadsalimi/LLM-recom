from typing import Any, List, Dict, Union, Tuple

import numpy as np
import torch
from pytorch_lightning import LightningModule
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Subset
from transformers import get_linear_schedule_with_warmup

from data.session.vector_dataset import SessionVectorDataset
from metric.mrr import MRR
from metric.cosine import CosineSimilarityLoss
from product_transformer.model import ProductTransformer


class ProductTransformerWithLinearModule(LightningModule):

    def __init__(self, d_model: int = 768,
                 n_layers: int = 12,
                 n_head: int = 12,
                 d_hidden: int = 2048,
                 dropout: float = 0.1,
                 lr: float = 5e-4,
                 weight_decay: float = 1e-1,
                 scheduler_n_warmup: int = 1000,
                 mrr_similarity_batch_size: int = 10000,
                 triplet_margin: float = 1.2):
        super().__init__()
        self.model = ProductTransformer(d_model=d_model,
                                        n_layers=n_layers,
                                        n_head=n_head,
                                        d_hidden=d_hidden,
                                        dropout=dropout)
        self.retrieval_mapping = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_model),
        )
        self.hparams.update(d_model=d_model,
                            n_layers=n_layers,
                            n_head=n_head,
                            d_hidden=d_hidden,
                            dropout=dropout,
                            lr=lr,
                            weight_decay=weight_decay,
                            scheduler_n_warmup=scheduler_n_warmup,
                            mrr_similarity_batch_size=mrr_similarity_batch_size,
                            triplet_margin=triplet_margin)

    def forward(self, batch: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Arguments:
            batch: List[List[torch.Tensor]], list of session item vectors

        Returns:
            y_hat: Tensor, shape ``[L', D]``.
            y: Tensor, shape ``[L', D]``.
        """
        batch = [s.to(self.device) for s in batch]
        x = [s[:-1] for s in batch]
        Ls = [s.shape[0] for s in x]
        L = max(Ls)
        x = torch.stack([F.pad(s, (0, 0, 0, L - s.shape[0])) for s in x], dim=0)               # [B, L, D]
        padding_mask = ~torch.stack([F.pad(torch.ones(l, device=x.device),
                                           (0, L - l)) for l in Ls], dim=0).bool()             # [B, L]
        y = torch.cat([s[1:] for s in batch], dim=0)                                                # [L', D]
        y_prime = self.retrieval_mapping(y)                                                         # [L', D]
        y_hat = self.model(x, padding_mask)                                                         # [B, L, D]
        flat_mask = (~padding_mask).flatten(0, 1)                                                   # [BL]
        y_hat = y_hat.flatten(0, 1)[flat_mask]                                                      # [L', D]
        return y_hat, y_prime

    def __step(self, batch: List[torch.Tensor], stage: str) -> torch.Tensor:
        dataset: Subset = self.trainer.val_dataloaders[0].dataset
        y_hat, y_prime = self(batch)
        # loss = self.loss(y_hat, y_prime)
        # vectors = np.array(vector_io._ParquetVectorIO__data['vector'].tolist())
        random_negative_indices = torch.randint(0, len(dataset.dataset.vector_io._ParquetVectorIO__data), size=(y_hat.shape[0],))
        y_negative = torch.from_numpy(np.stack(
            dataset.dataset.vector_io._ParquetVectorIO__data['vector'].iloc[random_negative_indices].tolist(),
            axis=0
        )).to(y_hat.device)
        y_prime_negative = self.retrieval_mapping(y_negative)
        loss = F.triplet_margin_loss(y_hat, y_prime, y_prime_negative, margin=self.hparams['triplet_margin'])
        # positive_similarity_loss = 1 - F.cosine_similarity(y_hat, y_prime, dim=-1)
        # negative_similarity_loss = 1 - F.cosine_similarity(y_hat, y_prime_negative, dim=-1)
        # loss = F.relu(positive_similarity_loss - negative_similarity_loss + self.hparams['triplet_margin']).mean()
        self.log(f'{stage}_loss', loss, batch_size=len(batch))
        return loss

    def training_step(self, batch: Dict[str, Union[List[torch.Tensor], List[str]]], batch_idx: int) -> torch.Tensor:
        self.log('lr0', self.optimizers().param_groups[0]['lr'], prog_bar=True)
        self.log('lr1', self.optimizers().param_groups[1]['lr'], prog_bar=True)
        return self.__step(batch['vectors'], 'train')

    @torch.no_grad()
    def validation_step(self, batch: Dict[str, Union[List[torch.Tensor], List[str]]], batch_idx: int) -> torch.Tensor:
        self.__step(batch['vectors'], 'val')

    def on_test_start(self) -> None:
        dataset: SessionVectorDataset = self.trainer.test_dataloaders[0].dataset
        self.mrr = MRR(dataset.vector_io, similarity_batch_size=self.hparams['mrr_similarity_batch_size'],
                       map_vectors=self.retrieval_mapping, device=self.device)

    @torch.no_grad()
    def test_step(self, batch: Dict[str, Union[List[torch.Tensor], List[str]]], batch_idx: int) -> torch.Tensor:
        vectors, gt_ids, gt_locales = batch['vectors'], batch['gt_id'], batch['gt_locale']
        y_hat, y_prime = self(vectors)
        # loss = self.loss(y_hat, y_prime)
        # self.log('test_loss', loss, batch_size=len(vectors))
        prediction_indices = torch.cumsum(torch.tensor([len(v) - 1 for v in vectors], device=y_hat.device), dim=0) - 1
        output = y_hat[prediction_indices]    # [B, D]
        mrr = self.mrr(output, gt_ids, gt_locales)
        print(f'MRR: {mrr}')
        self.log('MRR', mrr, batch_size=len(vectors), prog_bar=True)

    def get_grouped_params(self) -> List[Dict[str, Any]]:
        params_with_wd, params_without_wd = [], []
        no_decay = ['bias', 'LayerNorm.weight']
        for name, param in self.named_parameters():
            if any(nd in name for nd in no_decay):
                params_without_wd.append(param)
            else:
                params_with_wd.append(param)
        return [{'params': params_with_wd, 'weight_decay': self.hparams['weight_decay']},
                {'params': params_without_wd, 'weight_decay': 0.0}]

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.get_grouped_params(), lr=self.hparams['lr'])
        scheduler = dict(
            scheduler=get_linear_schedule_with_warmup(optimizer,
                                                      num_warmup_steps=self.hparams['scheduler_n_warmup'],
                                                      num_training_steps=2000000,
                                                      last_epoch=self.trainer.current_epoch - 1),
            interval='step')
        return [optimizer], [scheduler]
