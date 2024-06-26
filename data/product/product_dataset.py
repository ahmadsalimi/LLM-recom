from typing import Dict

import pandas as pd
from torch.utils.data import Dataset
from unidecode import unidecode

from data.product.format.formatter import ProductFormatter


class ProductDataset(Dataset):

    def __init__(self, file: str, formatter: ProductFormatter):
        self.file = file
        self.products = pd.read_csv(file)
        self.formatter = formatter

    def __len__(self) -> int:
        return len(self.products)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        id_ = self.products.iloc[idx]['id']
        locale = self.products.iloc[idx]['locale']
        return dict(
            id=id_,
            locale=locale,
            text=unidecode(self.formatter.format(self.products.iloc[idx].to_dict()))
        )
