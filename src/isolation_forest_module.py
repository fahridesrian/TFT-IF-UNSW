"""
isolation_forest_module.py
Isolation Forest (Liu et al., 2008, 2012) digunakan HANYA sebagai penghasil
skor anomali (anomaly scorer), bukan classifier akhir (proposal Bagian 5.3
dan 6.f). IF dilatih hanya pada subset kelas Normal di data latih, sehingga
skor merepresentasikan derajat penyimpangan terhadap pola lalu lintas normal.

Skor yang ditempel adalah s(x, n) LITERAL dari Persamaan 2 proposal
(keputusan terkunci A11/2026-09-27):

    s(x, n) = 2^(-E[h(x)] / c(n)),   c(n) = 2H(n-1) - 2(n-1)/n

sklearn `IsolationForest.score_samples` mengembalikan -s(x) persis
(implementasinya mengikuti Algorithm 3 Liu et al., termasuk penyesuaian
c(Size) pada external node), sehingga s = -score_samples berada di (0,1):
mendekati 1 berarti anomali, sekitar 0.5 ambigu, mendekati 0 normal.
Tidak ada normalisasi ulang (min-max dihapus) agar rumus yang dipaparkan
di skripsi identik dengan angka yang mengalir ke model.
"""
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


def score_and_attach(iso: IsolationForest, df: pd.DataFrame, feature_cols: list):
    """Hitung skor anomali literal s(x,n) = -score_samples (Liu et al. 2008,
    Eq. 2), tanpa normalisasi tambahan; makin besar = makin anomali. Ditempel
    sebagai kolom 'if_score' (fitur ke-(F+1))."""
    anomaly_scores = -iso.score_samples(df[feature_cols].values)

    out_df = df.copy()
    out_df["if_score"] = anomaly_scores
    return out_df


def integrate_isolation_forest(train_df, val_df, test_df, feature_cols: list):
    iso = fit_isolation_forest(train_df, feature_cols)

    train_df2 = score_and_attach(iso, train_df, feature_cols)
    val_df2 = score_and_attach(iso, val_df, feature_cols)
    test_df2 = score_and_attach(iso, test_df, feature_cols)

    new_feature_cols = feature_cols + ["if_score"]
    return train_df2, val_df2, test_df2, new_feature_cols, iso
