import pandas as pd
import cudf

def stream_data(file_path, chunk_size=2048):
    """Yields GPU-ready chunks from a CSV."""
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        yield cudf.from_pandas(chunk)