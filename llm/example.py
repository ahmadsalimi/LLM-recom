from typing import List

import torch
from torch import nn


class ExampleLLM(nn.Module):
    def forward(self, products: List[str]) -> torch.Tensor:
        """
        Arguments:
            products: List[str], list of formatted product strings.

        Returns:
            output: Tensor, shape ``[B, D]``.
        """
        return torch.randn(len(products), 1024)
