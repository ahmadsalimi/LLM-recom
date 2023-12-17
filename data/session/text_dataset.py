from typing import List, Tuple, Dict

import pandas as pd
from torch.utils.data import Dataset

from data.product.format.formatter import ProductFormatter
from data.session.common import read_sessions


class SessionTextDataset(Dataset):

    def __init__(self, sessions_file: str, products_file: str, formatter: ProductFormatter):
        self.sessions_file = sessions_file
        self.products_file = products_file
        self.sessions = read_sessions(sessions_file)
        self.products = pd.read_csv(products_file)
        self.pid_and_locale_to_index = {(pid, locale): i
                                        for i, (pid, locale)
                                        in enumerate(zip(self.products['id'].tolist(),
                                                         self.products['locale'].tolist()))}
        self.formatter = formatter

    def __len__(self) -> int:
        return len(self.sessions)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        session = self.sessions.iloc[idx]
        items_ids = session['prev_items'] + [session['next_item']]
        formatted_items = [
            self.formatter.format(
                self.products.iloc[self.pid_and_locale_to_index[(id_, session['locale'])]].to_dict())
            for id_ in items_ids
        ]
        text = '\n'.join([
            f' Product {i + 1} '.center(80, '-') + '\n' + item
            for i, item in enumerate(formatted_items[:-1])
        ] + [f' Product {len(formatted_items)} '.center(80, '-')])
        gt_id = items_ids[-1]
        gt_locale = session['locale']
        return dict(
            text=text,
            gt_id=gt_id,
            gt_locale=gt_locale
        )
