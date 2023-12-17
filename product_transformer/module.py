from typing import Any, List, Dict, Union, Tuple

import torch
from pytorch_lightning import LightningModule
from torch.nn import functional as F
from transformers import get_linear_schedule_with_warmup

from data.session.vector_dataset import SessionVectorDataset
from metric.mrr import MRR
from metric.cosine import CosineSimilarityLoss
from product_transformer.model import ProductTransformer


class ProductTransformerModule(LightningModule):

    def __init__(self, d_model: int = 768,
                 n_layers: int = 12,
                 n_head: int = 12,
                 d_hidden: int = 2048,
                 dropout: float = 0.1,
                 lr: float = 5e-4,
                 weight_decay: float = 1e-1,
                 scheduler_n_warmup: int = 1000):
        super().__init__()
        self.model = ProductTransformer(d_model=d_model,
                                        n_layers=n_layers,
                                        n_head=n_head,
                                        d_hidden=d_hidden,
                                        dropout=dropout)
        self.hparams.update(d_model=d_model,
                            n_layers=n_layers,
                            n_head=n_head,
                            d_hidden=d_hidden,
                            dropout=dropout,
                            lr=lr,
                            weight_decay=weight_decay,
                            scheduler_n_warmup=scheduler_n_warmup)
        self.mrr = None
        self.loss = CosineSimilarityLoss()

    def forward(self, batch: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Arguments:
            batch: List[torch.Tensor], list of session item vectors

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
        y_hat = self(x, padding_mask)                                                               # [B, L, D]
        flat_mask = (~padding_mask).flatten(0, 1)                                                   # [BL]
        y_hat = y_hat.flatten(0, 1)[flat_mask]                                                      # [L', D]
        return y_hat, y

    def __step(self, batch: List[torch.Tensor], stage: str) -> torch.Tensor:
        y_hat, y = self(batch)
        loss = self.loss(y_hat, y)
        self.log(f'{stage}_loss', loss)
        return loss

    def training_step(self, batch: Dict[str, Union[torch.Tensor, List[str]]], batch_idx: int) -> torch.Tensor:
        return self.__step(batch['vectors'], 'train')

    @torch.no_grad()
    def validation_step(self, batch: Dict[str, Union[torch.Tensor, List[str]]], batch_idx: int) -> torch.Tensor:
        self.__step(batch['vectors'], 'val')

    def on_test_start(self) -> None:
        dataset: SessionVectorDataset = self.trainer.test_dataloaders[0].dataset
        self.mrr = MRR(dataset.vector_io)

    @torch.no_grad()
    def test_step(self, batch: Dict[str, Union[torch.Tensor, List[str]]], batch_idx: int) -> torch.Tensor:
        vectors, gt_ids, gt_locales = batch['vectors'], batch['gt_id'], batch['gt_locale']
        y_hat, y = self(vectors)
        loss = self.loss(y_hat, y)
        self.log('test_loss', loss)
        output = y_hat[torch.cumsum(torch.tensor([len(v) for v in vectors]), dim=0) - 1]    # [B, D]
        mrr = self.mrr(output, gt_ids, gt_locales)
        self.log('MRR', mrr, prog_bar=True)

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
                                                      num_training_steps=len(self.trainer.train_dataloader)
                                                                         * self.trainer.max_epochs,
                                                      last_epoch=self.trainer.current_epoch - 1),
            interval='step')
        return [optimizer], [scheduler]
