from typing import Union, Tuple, Optional, List

import pandas as pd


def read_sessions(file: Union[str, Tuple[str, str]], include_locale: Optional[List[str]] = None) -> pd.DataFrame:
    if isinstance(file, str):
        sessions = pd.read_csv(file)
    else:
        sessions_file, gt_file = file
        sessions = pd.read_csv(sessions_file)
        gt = pd.read_csv(gt_file)
        sessions['next_item'] = gt['next_item']
    sessions['prev_items'] = sessions['prev_items'] \
        .str.replace(r"(\['|'\])", '', regex=True).str.split(r"'\s+'", regex=True)
    if include_locale is not None:
        sessions = sessions[sessions['locale'].isin(include_locale)]
    return sessions
