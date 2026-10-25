"""Minimal MNIST IDX loader for BME1322 Assignment 2."""

from pathlib import Path
import gzip
import struct

import numpy as np


def _read_idx(path: str | Path) -> np.ndarray:
    """Read an IDX-format file into a NumPy array."""
    path = Path(path)
    with gzip.open(path, "rb") as f:
        magic = f.read(4)
        if len(magic) != 4:
            raise ValueError(f"Invalid IDX header in {path}")

        dtype_code = magic[2]
        ndim = magic[3]

        dtype_map = {
            0x08: np.uint8,
            0x09: np.int8,
            0x0B: np.dtype(">i2"),
            0x0C: np.dtype(">i4"),
            0x0D: np.dtype(">f4"),
            0x0E: np.dtype(">f8"),
        }
        if dtype_code not in dtype_map:
            raise ValueError(f"Unsupported IDX dtype {dtype_code:#04x} in {path}")

        dims = [struct.unpack(">I", f.read(4))[0] for _ in range(ndim)]
        data = f.read()
        if len(data) != int(np.prod(dims)):
            raise ValueError(f"Unexpected data size in {path}")
        return np.frombuffer(data, dtype=dtype_map[dtype_code]).reshape(dims)


def load_mnist(data_dir: str | Path):
    """Load the train/test splits from an MNIST data directory.

    Returns:
        (X_train, y_train, X_test, y_test)
    """
    data_dir = Path(data_dir)
    X_train = _read_idx(data_dir / "train-images-idx3-ubyte.gz")
    y_train = _read_idx(data_dir / "train-labels-idx1-ubyte.gz")
    X_test = _read_idx(data_dir / "t10k-images-idx3-ubyte.gz")
    y_test = _read_idx(data_dir / "t10k-labels-idx1-ubyte.gz")

    expected = [(60000, 28, 28), (60000,), (10000, 28, 28), (10000,)]
    for arr, shape in zip((X_train, y_train, X_test, y_test), expected):
        if arr.shape != shape:
            raise ValueError(f"Unexpected MNIST shape {arr.shape}, expected {shape}")

    return X_train, y_train.astype(np.int64), X_test, y_test.astype(np.int64)
