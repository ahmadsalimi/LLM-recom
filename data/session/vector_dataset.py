from typing import Dict, Union, Tuple, Optional, List

import torch
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import collate, default_collate_fn_map

from data.product.io.io import VectorIO
from data.session.common import read_sessions


class SessionVectorDataset(Dataset):
    def __init__(self, session_file: Union[str, Tuple[str, str]], vector_io: VectorIO,
                 include_locale: Optional[List[str]] = None):
        self.session_file = session_file
        self.sessions = read_sessions(session_file, include_locale=include_locale)
        self.vector_io = vector_io

    def __len__(self) -> int:
        return len(self.sessions)

    def __getitem__(self, idx: int) -> Dict[str, Union[List[torch.Tensor], str]]:
        session = self.sessions.iloc[idx]
        item_ids = session['prev_items'] + [session['next_item']]
        item_vectors = [self.vector_io.get(item, session['locale']) for item in item_ids]
        gt_id = session['next_item']
        gt_locale = session['locale']
        return dict(
            vectors=torch.stack(item_vectors, dim=0),
            gt_id=gt_id,
            gt_locale=gt_locale,
        )

    @staticmethod
    def my_collate_tensor_fn(batch, *args, **kwargs):
        return batch

    @classmethod
    def my_collate(cls, batch):
        my_collate_fn_map = default_collate_fn_map.copy()
        my_collate_fn_map[torch.Tensor] = cls.my_collate_tensor_fn
        return collate(batch, collate_fn_map=my_collate_fn_map)
