import pandas as pd
from torch.utils.data import Dataset

from data.product.format.formatter import ProductFormatter


class SessionTextDataset(Dataset):

    def __init__(self, sessions_file: str, products_file: str, formatter: ProductFormatter):
        self.sessions_file = sessions_file
        self.products_file = products_file
        self.sessions = pd.read_csv(sessions_file)
        self.sessions['prev_items'] = self.sessions['prev_items'] \
            .str.replace(r"(\['|'\])", '', regex=True).str.split("' '")
        self.products = pd.read_csv(products_file)
        self.pid_and_locale_to_index = {(pid, locale): i
                                        for i, (pid, locale)
                                        in enumerate(zip(self.products['id'].tolist(),
                                                         self.products['locale'].tolist()))}
        self.formatter = formatter

    def __len__(self) -> int:
        return len(self.sessions)

    def __getitem__(self, idx: int) -> str:
        session = self.sessions.iloc[idx]
        items = session['prev_items'] + [session['next_item']]
        items = [
            self.formatter.format(
                self.products.iloc[self.pid_and_locale_to_index[(item, session['locale'])]].to_dict())
            for item in items
        ]
        items = [f' Product {i + 1} '.center(80, '-') + '\n' + item
                 for i, item in enumerate(items)]
        items += [f' Product {len(items) + 1} '.center(80, '-')]
        return '\n'.join(items)
