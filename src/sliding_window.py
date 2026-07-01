"""
sliding_window.py
Membentuk representasi deret waktu multivariat sesuai Persamaan 1 proposal:
    {(X_{t-W+1}, ..., X_t)} -> Y_{(t+1,...,t+T)}
Sliding window dibentuk TERPISAH per split (train/val/test) dan per
source_file, agar tidak ada sekuens yang melintasi batas file CSV atau
batas antar subset data (proposal Bagian 6.g.iii - iv).

Catatan: karena preprocessing.py sudah men-drop source_file_col sebelum
encoding/scaling, fungsi build_windows_for_split di bawah menerima daftar
sub-dataframe per file (caller bertanggung jawab menjaga pemisahan ini
sebelum kolom source_file di-drop).
"""
import numpy as np
import pandas as pd
from typing import List, Tuple

from config import DATA, WINDOW


def _windows_from_single_segment(
    df_segment: pd.DataFrame, feature_cols: List[str], target_col: str,
    W: int, T: int, stride: int,
) -> Tuple[np.ndarray, np.ndarray]:
    X = df_segment[feature_cols].values
    Y = df_segment[target_col].values
    n = len(df_segment)

    X_windows, Y_windows = [], []
    last_start = n - (W + T)
    for start in range(0, last_start + 1, stride):
        hist_end = start + W
        tgt_end = hist_end + T
        X_windows.append(X[start:hist_end])
        Y_windows.append(Y[hist_end:tgt_end])

    if not X_windows:
        return (
            np.empty((0, W, len(feature_cols))),
            np.empty((0, T), dtype=int),
        )
    return np.stack(X_windows), np.stack(Y_windows)


def build_windows(
    df_with_source: pd.DataFrame, feature_cols: List[str],
    target_col: str = None, W: int = None, T: int = None, stride: int = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """df_with_source harus masih memiliki kolom source_file (belum di-drop)
    dan sudah terurut kronologis di dalam tiap file."""
    target_col = target_col or DATA.target_col
    W = W or WINDOW.W
    T = T or WINDOW.T
    stride = stride or WINDOW.stride

    all_X, all_Y = [], []
    for _, seg in df_with_source.groupby(DATA.source_file_col, sort=False):
        Xw, Yw = _windows_from_single_segment(seg, feature_cols, target_col, W, T, stride)
        if len(Xw) > 0:
            all_X.append(Xw)
            all_Y.append(Yw)

    if not all_X:
        return np.empty((0, W, len(feature_cols))), np.empty((0, T), dtype=int)

    return np.concatenate(all_X, axis=0), np.concatenate(all_Y, axis=0)
