from typing import Union, Tuple, Optional, List

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from data.product.format.default import DefaultFormatter
from data.product.format.formatter import ProductFormatter
from data.session.text_dataset import SessionTextDataset


class SessionTextDataModule(LightningDataModule):

    def __init__(self, batch_size: int,
                 sessions_file: Union[str, Tuple[str, str]],
                 products_file: str,
                 num_workers: int = 0,
                 formatter: Optional[ProductFormatter] = None,
                 include_locale: Optional[List[str]] = None):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sessions_file = sessions_file
        self.products_file = products_file
        self.formatter = formatter or DefaultFormatter()
        self.include_locale = include_locale
        self.test_dataset = None

    def setup(self, stage: str = None) -> None:
        assert stage == 'test', f'Only "test" stage is supported, got {stage}'
        self.test_dataset = SessionTextDataset(self.sessions_file, self.products_file, self.formatter,
                                               include_locale=self.include_locale)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset,
                          shuffle=False,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)
