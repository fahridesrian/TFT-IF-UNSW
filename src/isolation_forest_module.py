"""
isolation_forest_module.py
Isolation Forest (Liu et al., 2008, 2012) digunakan HANYA sebagai penghasil
skor anomali (anomaly scorer), bukan classifier akhir (proposal Bagian 5.3
dan 6.f). IF dilatih hanya pada subset kelas Normal di data latih, sehingga
skor merepresentasikan derajat penyimpangan terhadap pola lalu lintas normal.
Skor dinormalisasi ke (0,1) memakai statistik train saja (hindari leakage),
lalu ditempel sebagai fitur tambahan ke-(F+1).
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from config import DATA, IF


def fit_isolation_forest(train_df: pd.DataFrame, feature_cols: list) -> IsolationForest:
    normal_class_idx = list(DATA.classes_used).index("Normal")
    normal_mask = train_df[DATA.target_col] == normal_class_idx
    X_normal = train_df.loc[normal_mask, feature_cols].values

    iso = IsolationForest(
        n_estimators=IF.n_estimators,
        max_samples=IF.max_samples,
        contamination=IF.contamination,
        random_state=IF.random_state,
        n_jobs=-1,
    )
    iso.fit(X_normal)
    return iso


def score_and_attach(
    iso: IsolationForest, df: pd.DataFrame, feature_cols: list,
    score_min: float = None, score_max: float = None,
):
    """Menghitung skor anomali (semakin besar = semakin anomali, mengikuti
    konvensi s(x,n) pada Persamaan 2 proposal) dan menempelkannya sebagai
    kolom 'if_score' baru. Normalisasi min-max memakai score_min/score_max
    dari TRAIN (dihitung di luar & dipakai ulang untuk val/test)."""
    raw_scores = iso.decision_function(df[feature_cols].values)  # makin besar = makin normal
    anomaly_scores = -raw_scores  # balik tanda: makin besar = makin anomali

    if score_min is None:
        score_min = anomaly_scores.min()
    if score_max is None:
        score_max = anomaly_scores.max()
    denom = (score_max - score_min) if (score_max - score_min) != 0 else 1e-8
    norm_scores = (anomaly_scores - score_min) / denom
    norm_scores = np.clip(norm_scores, 0.0, 1.0)

    out_df = df.copy()
    out_df["if_score"] = norm_scores
    return out_df, score_min, score_max


def integrate_isolation_forest(train_df, val_df, test_df, feature_cols: list):
    iso = fit_isolation_forest(train_df, feature_cols)

    train_df2, smin, smax = score_and_attach(iso, train_df, feature_cols)
    val_df2, _, _ = score_and_attach(iso, val_df, feature_cols, smin, smax)
    test_df2, _, _ = score_and_attach(iso, test_df, feature_cols, smin, smax)

    new_feature_cols = feature_cols + ["if_score"]
    return train_df2, val_df2, test_df2, new_feature_cols, iso
