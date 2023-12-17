import torch
from torch import nn
from torch.nn import functional as F


class CosineSimilarityLoss(nn.Module):

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        cosine_similarity = F.cosine_similarity(y_hat, y, dim=-1).mean()                            # []
        loss = 1 - cosine_similarity
        return loss
