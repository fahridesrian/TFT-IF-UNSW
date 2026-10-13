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
    n = len(df_segment)
    F = len(feature_cols)
    if n < W + T:
        return (
            np.empty((0, W, F), dtype=np.float32),
            np.empty((0, T), dtype=np.int64),
        )
    # float32 sejak awal: train.make_loader (.float()) jadi no-op tanpa
    # salinan, dan ukuran array setengah dari float64. Model hanya butuh
    # presisi float32 (AMP bahkan memakai float16 di GPU).
    X = df_segment[feature_cols].to_numpy(dtype=np.float32)
    Y = df_segment[target_col].to_numpy()
    starts = np.arange(0, n - W - T + 1, stride)
    num = len(starts)
    # sliding_window_view = tampilan ber-stride tanpa menyalin; ambil
    # salinan contiguous sekali per segmen (bukan 1,4 juta array per-baris
    # seperti loop lama yang memakan memori sementara 2x lipat).
    X_view = np.lib.stride_tricks.sliding_window_view(X, W, axis=0)
    # sliding_window_view(X, W, axis=0) -> (n-W+1, F, W); transpose ke (n-W+1, W, F)
    Xw = np.ascontiguousarray(X_view[starts, :, :].transpose(0, 2, 1))
    Y_view = np.lib.stride_tricks.sliding_window_view(Y, T)
    Yw = np.ascontiguousarray(Y_view[starts + W].astype(np.int64, copy=False))
    assert Xw.shape == (num, W, F), (Xw.shape, num, W, F)
    assert Yw.shape == (num, T), (Yw.shape, num, T)
    return Xw, Yw


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
        return (
            np.empty((0, W, len(feature_cols)), dtype=np.float32),
            np.empty((0, T), dtype=np.int64),
        )

    return (
        np.concatenate(all_X, axis=0).astype(np.float32, copy=False),
        np.concatenate(all_Y, axis=0).astype(np.int64, copy=False),
    )
