"""
preprocessing.py
Tahap praproses sesuai Bagian 6.d proposal:
  1. Filter 4 kelas (Normal, Generic, Exploits, Fuzzers)
  2. Cleaning null/inf/duplikat
  3. Sort kronologis (stime)
  4. Split kronologis train/val/test TANPA shuffle
  5. Encoding kategorikal (fit hanya di train)
  6. Min-Max scaling (fit hanya di train)
Semua parameter preprocessing (encoder, scaler) hanya di-fit pada data latih
untuk mencegah data leakage (proposal Bagian 6.d.vii).
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Tuple
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from config import DATA, PREPROC

# Kolom yang MEMANG bertipe string (jangan dipaksa numerik).
_NON_NUMERIC_COLS = set(DATA.categorical_cols) | {
    DATA.target_col, DATA.source_file_col, "srcip", "dstip",
}


@dataclass
class PreprocArtifacts:
    encoders: Dict[str, LabelEncoder]
    scaler: MinMaxScaler
    feature_cols: list
    label_encoder: LabelEncoder  # untuk attack_cat -> int


def coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """CSV UNSW-NB15 asli dikenal memiliki kolom yang seharusnya numerik
    tetapi berisi string kosong/spasi atau nilai heksadesimal pada sebagian
    baris (mis. ct_flw_http_mthd, is_ftp_login, ct_ftp_cmd, sport, dsport),
    sehingga pandas membaca seluruh kolom sebagai object/string. Fungsi ini
    memaksa kolom-kolom non-kategorikal & non-target menjadi numerik;
    nilai yang tidak bisa dikonversi (mis. ' ', '-', string hex) menjadi
    NaN dan akan diimputasi di clean_data()."""
    df = df.copy()
    for c in df.columns:
        if c in _NON_NUMERIC_COLS:
            continue
        # Tidak digantungkan pada pengecekan dtype (object/str/StringDtype
        # berbeda-beda antar versi pandas) — pd.to_numeric pada kolom yang
        # sudah numerik hanya no-op murah, jadi selalu dipaksa di sini.
        df[c] = pd.to_numeric(df[c].astype(str).str.strip(), errors="coerce")
    return df


def normalize_string_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Trim whitespace & samakan casing pada kolom kategorikal dan target,
    karena UNSW-NB15 asli dikenal memiliki inkonsistensi seperti
    ' Fuzzers' / 'Fuzzers ' atau variasi kapitalisasi pada attack_cat,
    proto, state, service."""
    df = df.copy()
    cols_to_normalize = list(DATA.categorical_cols) + [DATA.target_col]
    for c in cols_to_normalize:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    return df


def filter_classes(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_string_columns(df)
    df[DATA.target_col] = df[DATA.target_col].replace(
        {"nan": "Normal", "": "Normal", "None": "Normal"}
    )
    df[DATA.target_col] = df[DATA.target_col].fillna("Normal")
    # Cocokkan case-insensitive terhadap classes_used agar variasi kapitalisasi
    # pada attack_cat (mis. 'fuzzers', 'FUZZERS') tetap tertangkap.
    canonical = {c.lower(): c for c in DATA.classes_used}
    df[DATA.target_col] = df[DATA.target_col].apply(
        lambda v: canonical.get(str(v).lower(), v)
    )
    df = df[df[DATA.target_col].isin(DATA.classes_used)].reset_index(drop=True)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = coerce_numeric_columns(df)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # inf -> nan lalu imputasi
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    if PREPROC.fillna_strategy == "median":
        for c in numeric_cols:
            if df[c].isna().any():
                df[c] = df[c].fillna(df[c].median())
    else:
        df[numeric_cols] = df[numeric_cols].fillna(0)

    non_numeric_cols = [c for c in df.columns if c not in numeric_cols]
    for c in non_numeric_cols:
        df[c] = df[c].fillna("unknown")

    df = df.drop_duplicates().reset_index(drop=True)
    return df


def sort_chronologically(df: pd.DataFrame) -> pd.DataFrame:
    sort_col = DATA.time_cols[0] if DATA.time_cols[0] in df.columns else None
    if sort_col is None:
        return df.reset_index(drop=True)
    df = df.sort_values(by=[DATA.source_file_col, sort_col]).reset_index(drop=True)
    return df


def chronological_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split per source_file secara kronologis agar proporsi tiap file terjaga
    dan tidak ada leakage antar waktu (proposal Bagian 6.d.iv)."""
    train_parts, val_parts, test_parts = [], [], []
    for _, g in df.groupby(DATA.source_file_col, sort=False):
        n = len(g)
        n_train = int(n * PREPROC.train_ratio)
        n_val = int(n * PREPROC.val_ratio)
        train_parts.append(g.iloc[:n_train])
        val_parts.append(g.iloc[n_train:n_train + n_val])
        test_parts.append(g.iloc[n_train + n_val:])
    train_df = pd.concat(train_parts).reset_index(drop=True)
    val_df = pd.concat(val_parts).reset_index(drop=True)
    test_df = pd.concat(test_parts).reset_index(drop=True)
    return train_df, val_df, test_df


def encode_and_scale(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, PreprocArtifacts]:
    train_df, val_df, test_df = train_df.copy(), val_df.copy(), test_df.copy()

    # source_file_col SENGAJA dipertahankan (bukan di-drop) karena dibutuhkan
    # oleh sliding_window.py untuk menjaga batas antar file CSV saat
    # pembentukan sekuens. Kolom ini dikeluarkan dari feature_cols di bawah.
    drop_cols = list(DATA.id_cols_to_drop) + list(DATA.time_cols) + [
        DATA.binary_label_col
    ]
    for d in [train_df, val_df, test_df]:
        for c in drop_cols:
            if c in d.columns:
                d.drop(columns=c, inplace=True)

    # Encode kategorikal, fit hanya di train
    encoders = {}
    for c in DATA.categorical_cols:
        if c not in train_df.columns:
            continue
        le = LabelEncoder()
        le.fit(train_df[c].astype(str))
        encoders[c] = le
        for d in [train_df, val_df, test_df]:
            known = set(le.classes_)
            d[c] = d[c].astype(str).apply(lambda v: v if v in known else le.classes_[0])
            d[c] = le.transform(d[c])

    # Encode target attack_cat -> integer sesuai urutan DATA.classes_used
    label_encoder = LabelEncoder()
    label_encoder.fit(list(DATA.classes_used))
    for d in [train_df, val_df, test_df]:
        d[DATA.target_col] = label_encoder.transform(d[DATA.target_col])

    feature_cols = [
        c for c in train_df.columns
        if c not in (DATA.target_col, DATA.source_file_col)
    ]

    scaler = MinMaxScaler()
    scaler.fit(train_df[feature_cols])
    for d in [train_df, val_df, test_df]:
        d[feature_cols] = scaler.transform(d[feature_cols])

    artifacts = PreprocArtifacts(
        encoders=encoders, scaler=scaler, feature_cols=feature_cols,
        label_encoder=label_encoder,
    )
    return train_df, val_df, test_df, artifacts


def run_preprocessing(raw_df: pd.DataFrame):
    df = filter_classes(raw_df)
    df = clean_data(df)
    df = sort_chronologically(df)
    train_df, val_df, test_df = chronological_split(df)
    train_df, val_df, test_df, artifacts = encode_and_scale(train_df, val_df, test_df)
    return train_df, val_df, test_df, artifacts
