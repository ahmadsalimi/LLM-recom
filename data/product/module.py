from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from data.product.format.default import DefaultFormatter
from data.product.product_dataset import ProductDataset


class ProductDataModule(LightningDataModule):

    def __init__(self, batch_size: int,
                 num_workers: int = 0,
                 file: str = None,
                 formatter: str = None):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.file = file
        self.formatter = formatter or DefaultFormatter()
        self.predict_dataset = None

    def setup(self, stage: str = None) -> None:
        assert stage == 'predict', f'Only "predict" stage is supported, got {stage}'
        self.predict_dataset = ProductDataset(self.file, self.formatter)

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(self.predict_dataset,
                          shuffle=False,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)
