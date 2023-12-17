from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from data.product.io.io import VectorIO
from data.session.vector_dataset import SessionVectorDataset


class SessionVectorDataModule(LightningDataModule):

    def __init__(self, batch_size: int,
                 sessions_file: str,
                 num_workers: int = 0,
                 train_vector_io: VectorIO = None,
                 val_vector_io: VectorIO = None,
                 test_vector_io: VectorIO = None):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sessions_file = sessions_file
        self.train_vector_io = train_vector_io
        self.val_vector_io = val_vector_io
        self.test_vector_io = test_vector_io
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: str = None) -> None:
        if stage == 'fit':
            self.train_dataset = SessionVectorDataset(self.sessions_file, self.train_vector_io)
            self.val_dataset = SessionVectorDataset(self.sessions_file, self.val_vector_io)
        elif stage == 'test':
            self.test_dataset = SessionVectorDataset(self.sessions_file, self.test_vector_io)
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
