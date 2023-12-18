import argparse
import os

import pandas as pd
import numpy as np
from tqdm import tqdm


def store_chunk(directory: str, data: Dict[str, List[Any]], i: int):
    os.makedirs(directory, exist_ok=True)
    df = pd.DataFrame(data)
    df.to_parquet(os.path.join(directory, f'chunk_{i}.parquet'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert numpy to parquet',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('numpy_dir', type=str, help='Path to the directory with numpy files')
    parser.add_argument('parquet_dir', type=str, help='Path to the directory with parquet files')
    parser.add_argument('--chunk-size', type=int, default=1000000, help='Chunk size')
    args = parser.parse_args()

    data = dict(
        id=[],
        locale=[],
        vector=[],
    )

    files = os.listdir(args.numpy_dir)
    current_chunk = 1
    for i, file in enumerate(tqdm(files, desc='Reading numpy files', total=len(files))):
        if not file.endswith('.npy'):
            continue
        id_, locale = os.path.splitext(file)[0].split('_')
        file_path = os.path.join(args.numpy_dir, file)
        data['id'].append(id_)
        data['locale'].append(locale)
        data['vector'].append(np.load(file_path))
        if len(data['id']) >= args.chunk_size:
            store_chunk(args.parquet_dir, data, current_chunk)
            current_chunk += 1
            data = dict(
                id=[],
                locale=[],
                vector=[],
            )

    if len(data['id']) > 0:
        store_chunk(args.parquet_dir, data, current_chunk)
