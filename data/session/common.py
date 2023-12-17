import pandas as pd


def read_sessions(file: str) -> pd.DataFrame:
    sessions = pd.read_csv(file)
    sessions['prev_items'] = sessions['prev_items'] \
        .str.replace(r"(\['|'\])", '', regex=True).str.split("' '")
    return sessions
