from typing import Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from data.product.io.io import VectorIO
from data.session.vector_dataset import SessionVectorDataset


class SessionVectorDataModule(LightningDataModule):

    def __init__(self, batch_size: int,
                 sessions_file: str,
                 num_workers: int = 0,
                 vector_io: Optional[VectorIO] = None):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sessions_file = sessions_file
        self.vector_io = vector_io
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: str = None) -> None:
        if stage == 'fit':
            self.vector_io.initialize_read()
            self.train_dataset = SessionVectorDataset(self.sessions_file, self.vector_io)
            self.val_dataset = SessionVectorDataset(self.sessions_file, self.vector_io)
        elif stage == 'test':
            self.vector_io.initialize_read()
            self.test_dataset = SessionVectorDataset(self.sessions_file, self.vector_io)
        else:
            raise ValueError(f'Unsupported stage: {stage}')

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset,
                          shuffle=True,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset,
                          shuffle=False,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset,
                          shuffle=False,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)
