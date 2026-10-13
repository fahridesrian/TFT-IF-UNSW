"""
data_loading.py
Memuat keempat file UNSW-NB15_{1..4}.csv, menggabungkannya, dan menandai
setiap baris dengan file asalnya (source_file) agar batas file bisa dijaga
saat pembentukan sliding window (lihat sliding_window.py).
"""
import os
import pandas as pd
from config import DATA


def _count_lines(fpath: str) -> int:
    # Hitung baris fisik (termasuk baris rusak) untuk laporan skip.
    # File ~700rb baris, loop Python polos masih <1 dtk per file.
    n = 0
    with open(fpath, "r", encoding="utf-8-sig", errors="replace") as fh:
        for _ in fh:
            n += 1
    return n


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
        # File asli UNSW-NB15 tidak punya header dan mengandung belasan
        # baris rusak (contoh: UNSW-NB15_4.csv baris ~100041-100579:
        # baris terpotong / dua record menempel jadi 46/61 kolom, bukan 49).
        # C engine tanpa on_bad_lines='skip' langsung ParserError.
        # Dummy test (tests/make_dummy_data.py) justru MENULIS header.
        # Strateginya: selalu baca header=None + skip baris rusak, lalu
        # deteksi header secara eksplisit dari isi baris pertama.
        total_lines = _count_lines(fpath)
        df = pd.read_csv(
            fpath, header=None, low_memory=False,
            encoding="utf-8-sig",
            on_bad_lines="skip",
        )
        # Deteksi header: baris pertama sama persis dengan DATA.columns
        # (perbandingan case-insensitive setelah strip).
        first = [str(v).strip().lower() for v in df.iloc[0].tolist()]
        expected = [str(c).strip().lower() for c in DATA.columns]
        has_header = (first == expected)
        if has_header:
            df = df.iloc[1:].reset_index(drop=True)
        if df.shape[1] != len(DATA.columns):
            raise ValueError(
                f"{fname}: jumlah kolom terbaca {df.shape[1]}, "
                f"harusnya {len(DATA.columns)} ({list(DATA.columns)[:3]}...). "
                "Cek format CSV."
            )
        df.columns = list(DATA.columns)
        # Baris yang KEPENDEKAN (<49 kolom, mis. baris terpotong di
        # UNSW-NB15_4.csv) TIDAK di-skip oleh C engine — melainkan diisi
        # NaN di kolom-kolom trailing. Baris valid selalu punya `label`
        # 0/1, jadi baris sampah terdeteksi tepat via label NaN.
        # (File ini tidak mengandung '"' sama sekali, jadi hitungan koma
        # = hitungan kolom; 8 baris pendek semacam ini ada di file 4.)
        n_short = int(df[DATA.binary_label_col].isna().sum())
        if n_short:
            df = df.dropna(subset=[DATA.binary_label_col]).reset_index(drop=True)
        n_skipped = total_lines - len(df) - (1 if has_header else 0)
        if n_skipped:
            print(
                f"    Peringatan {fname}: {n_skipped} baris rusak "
                f"dilewati (dari {total_lines} baris fisik)."
            )

        df[DATA.source_file_col] = fname
        frames.append(df)

    full_df = pd.concat(frames, axis=0, ignore_index=True)
    return full_df
