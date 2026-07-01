"""
data_loading.py
Memuat keempat file UNSW-NB15_{1..4}.csv, menggabungkannya, dan menandai
setiap baris dengan file asalnya (source_file) agar batas file bisa dijaga
saat pembentukan sliding window (lihat sliding_window.py).
"""
import os
import pandas as pd
from config import DATA


def load_unsw_nb15(raw_dir: str = None) -> pd.DataFrame:
    raw_dir = raw_dir or DATA.raw_dir
    frames = []
    for fname in DATA.csv_files:
        fpath = os.path.join(raw_dir, fname)
        if not os.path.exists(fpath):
            raise FileNotFoundError(
                f"File tidak ditemukan: {fpath}. "
                f"Pastikan keempat file UNSW-NB15_1..4.csv ada di {raw_dir}."
            )
        # header=None + names=... dipakai bila CSV asli tanpa header;
        # bila CSV sudah punya header sesuai config.DATA.columns, ganti ke header=0.
        try:
            df = pd.read_csv(fpath, low_memory=False)
            if list(df.columns) != list(DATA.columns):
                # fallback: file tanpa header, kolom sesuai urutan config
                df = pd.read_csv(fpath, header=None, names=list(DATA.columns), low_memory=False)
        except Exception:
            df = pd.read_csv(fpath, header=None, names=list(DATA.columns), low_memory=False)

        df[DATA.source_file_col] = fname
        frames.append(df)

    full_df = pd.concat(frames, axis=0, ignore_index=True)
    return full_df
