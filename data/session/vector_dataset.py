import torch
from torch.utils.data import Dataset

from data.product.io.io import VectorIO
from data.session.common import read_sessions


class SessionVectorDataset(Dataset):
    def __init__(self, session_file: str, vector_io: VectorIO):
        self.session_file = session_file
        self.sessions = read_sessions(session_file)
        self.vector_io = vector_io
        self.vector_io.initialize_read()

    def __len__(self) -> int:
        return len(self.sessions)

    def __getitem__(self, idx: int) -> torch.Tensor:
        session = self.sessions.iloc[idx]
        items = session['prev_items'] + [session['next_item']]
        items = [self.vector_io.get(item, session['locale']) for item in items]
        return torch.stack(items, dim=0)
