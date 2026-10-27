"""
statistik_skor_if.py
Menghitung statistik skor anomali Isolation Forest (if_score) per kelas dan per
bagian data untuk Subbab 3.6, serta ROC-AUC dan PR-AUC skor itu sendiri
(Normal melawan serangan). Hanya membaca; tidak mengubah berkas apa pun di repositori.

Jalankan dari ROOT repositori (folder yang berisi main.py dan data/raw):

    python statistik_skor_if.py
    python statistik_skor_if.py --raw-dir D:/data/unsw

Keluaran: layar dan logs/statistik_skor_if.md. Kirim berkas .md itu ke Claude.
Data, split, skaler, dan IF dibangun persis seperti main.py (deterministik).
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))
from config import DATA, IF  # noqa: E402
from data_loading import load_unsw_nb15  # noqa: E402
from preprocessing import run_preprocessing  # noqa: E402
from isolation_forest_module import integrate_isolation_forest  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--raw-dir", type=str, default=DATA.raw_dir)
a = p.parse_args()

OUT = []


def say(s=""):
    print(s)
    OUT.append(s)


raw_df = load_unsw_nb15(a.raw_dir)
train_df, val_df, test_df, art = run_preprocessing(raw_df)
feature_cols = art.feature_cols
tr, va, te, cols_if, iso = integrate_isolation_forest(train_df, val_df, test_df, feature_cols)
names = list(DATA.classes_used)
normal = names.index("Normal")

say("# Statistik skor anomali Isolation Forest")
say()
say(f"Banyak fitur masukan IF: {len(feature_cols)}. Pohon: {iso.n_estimators}. "
    f"max_samples efektif (psi): {iso.max_samples_}. random_state: {IF.random_state}.")
n_norm_tr = int((tr[DATA.target_col] == normal).sum())
say(f"Rekaman Normal data latih (dipakai melatih IF): {n_norm_tr}. "
    f"Batas atas rekaman yang pernah masuk satu pohon: {iso.n_estimators * iso.max_samples_} "
    f"({100 * iso.n_estimators * iso.max_samples_ / n_norm_tr:.2f}%).")
say()
say("| Bagian | Kelas | Jumlah | Rata-rata | Median | Persentil 5 | Persentil 95 |")
say("|---|---|---|---|---|---|---|")
for nm, df in [("Latih", tr), ("Validasi", va), ("Uji", te)]:
    for k, kn in enumerate(names):
        s = df.loc[df[DATA.target_col] == k, "if_score"].values
        if len(s) == 0:
            continue
        say(f"| {nm} | {kn} | {len(s)} | {s.mean():.4f} | {np.median(s):.4f} | "
            f"{np.percentile(s, 5):.4f} | {np.percentile(s, 95):.4f} |")
say()
say("Kemampuan skor itu sendiri membedakan Normal (0) dan serangan (1):")
say()
say("| Bagian | ROC-AUC | PR-AUC | Proporsi serangan |")
say("|---|---|---|---|")
for nm, df in [("Latih", tr), ("Validasi", va), ("Uji", te)]:
    y = (df[DATA.target_col] != normal).astype(int).values
    s = df["if_score"].values
    say(f"| {nm} | {roc_auc_score(y, s):.4f} | {average_precision_score(y, s):.4f} | {y.mean():.4f} |")

os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
with open(os.path.join(ROOT, "logs", "statistik_skor_if.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(OUT) + "\n")
print("\nTertulis: logs/statistik_skor_if.md")