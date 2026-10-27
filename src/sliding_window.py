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
import itertools
import os

import numpy as np
import pandas as pd
from typing import List, Tuple

from config import DATA, WINDOW

CHUNK = 50_000  # jendela per potongan saat mengisi larik keluaran
_COUNTER = itertools.count()


def _segment_arrays(df_segment, feature_cols, target_col, W, T, stride):
    """(X float32 (n, F), Y (n,), starts) untuk satu segmen, atau None bila n < W + T."""
    n = len(df_segment)
    if n < W + T:
        return None
    X = df_segment[feature_cols].to_numpy(dtype=np.float32)
    Y = df_segment[target_col].to_numpy()
    starts = np.arange(0, n - W - T + 1, stride)
    return X, Y, starts


def _alloc_windows(shape, memmap_dir, nama):
    """Larik keluaran. Dengan memmap_dir, larik disimpan di berkas (tidak memakai
    batas commit/pagefile Windows); tanpa itu, larik biasa di RAM."""
    if memmap_dir:
        os.makedirs(memmap_dir, exist_ok=True)
        path = os.path.join(memmap_dir, f"{nama}_{os.getpid()}_{next(_COUNTER)}.npy")
        return np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=shape)
    return np.empty(shape, dtype=np.float32)


def build_windows(
    df_with_source: pd.DataFrame, feature_cols: List[str],
    target_col: str = None, W: int = None, T: int = None, stride: int = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """df_with_source harus masih memiliki kolom source_file (belum di-drop)
    dan sudah terurut kronologis di dalam tiap file.

    Larik keluaran dialokasikan sekali dengan ukuran akhir lalu diisi per
    potongan (CHUNK jendela), sehingga memori puncak = ukuran larik akhir +
    satu potongan. Versi lama membuat salinan per segmen, salinan transpose,
    lalu np.concatenate (sekitar 2 sampai 3 kali ukuran akhir; gagal pada W=50).
    Isi larik identik dengan versi lama.

    Bila variabel lingkungan TFT_WINDOW_MEMMAP berisi nama folder, larik X
    disimpan sebagai berkas .npy di folder itu (memmap), bukan di RAM."""
    target_col = target_col or DATA.target_col
    W = W or WINDOW.W
    T = T or WINDOW.T
    stride = stride or WINDOW.stride
    F = len(feature_cols)
    memmap_dir = os.environ.get("TFT_WINDOW_MEMMAP") or None

    segs = []
    total = 0
    for _, seg in df_with_source.groupby(DATA.source_file_col, sort=False):
        a = _segment_arrays(seg, feature_cols, target_col, W, T, stride)
        if a is not None and len(a[2]) > 0:
            segs.append(a)
            total += len(a[2])

    if total == 0:
        return (
            np.empty((0, W, F), dtype=np.float32),
            np.empty((0, T), dtype=np.int64),
        )

    Xw = _alloc_windows((total, W, F), memmap_dir, "X")
    Yw = np.empty((total, T), dtype=np.int64)
    pos = 0
    for X, Y, starts in segs:
        # sliding_window_view(X, W, axis=0) -> (n-W+1, F, W): tampilan tanpa salinan
        X_view = np.lib.stride_tricks.sliding_window_view(X, W, axis=0)
        Y_view = np.lib.stride_tricks.sliding_window_view(Y, T)
        for i in range(0, len(starts), CHUNK):
            idx = starts[i:i + CHUNK]
            k = len(idx)
            Xw[pos:pos + k] = X_view[idx].transpose(0, 2, 1)
            Yw[pos:pos + k] = Y_view[idx + W].astype(np.int64, copy=False)
            pos += k
    assert pos == total, (pos, total)
    if memmap_dir:
        Xw.flush()
    return Xw, Yw


def build_last_labels(
    df_with_source: pd.DataFrame, target_col: str = None,
    W: int = None, T: int = None, stride: int = None,
) -> np.ndarray:
    """Label rekaman TERAKHIR di jendela historis tiap window, urutan sama
    dengan build_windows. Dipakai hanya untuk acuan persistensi (A16); model
    tidak pernah melihat label ini."""
    target_col = target_col or DATA.target_col
    W = W or WINDOW.W
    T = T or WINDOW.T
    stride = stride or WINDOW.stride
    out = []
    for _, seg in df_with_source.groupby(DATA.source_file_col, sort=False):
        n = len(seg)
        if n < W + T:
            continue
        y = seg[target_col].to_numpy()
        starts = np.arange(0, n - W - T + 1, stride)
        out.append(y[starts + W - 1])
    if not out:
        return np.empty((0,), dtype=np.int64)
    return np.concatenate(out).astype(np.int64, copy=False)
