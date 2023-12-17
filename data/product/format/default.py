from typing import List, Dict, Any

import pandas as pd

from data.product.format.formatter import ProductFormatter


class DefaultFormatter(ProductFormatter):

    def __init__(self, ignore_keys: List[str] = None):
        self.ignore_keys = ignore_keys or ['id']

    def format(self, product: Dict[str, Any]):
        return '\n'.join(f'{key}: {value}'
                         if key != 'price'
                         else f'{key}: {self.format_price(value, product["locale"])}'
                         for key, value in product.items()
                         if key not in self.ignore_keys and value is not None and pd.notna(value))
