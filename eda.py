"""
eda.py
Analisis eksplorasi data (Exploratory Data Analysis) sesuai proposal
Bagian 6.c: karakterisasi dataset UNSW-NB15 sebelum pemodelan.

Output (default di logs/eda/):
  - class_distribution.png        : bar chart distribusi kelas + imbalance ratio
  - histograms/<fitur>.png        : histogram + density per fitur numerik
  - histograms_grid.png           : semua histogram dalam satu grid
  - boxplots/<fitur>_by_class.png : boxplot per kelas untuk tiap fitur
  - correlation_heatmap.png       : heatmap korelasi Pearson antar fitur
  - correlation_matrix.csv        : matriks korelasi lengkap
  - describe_overall.csv          : statistik deskriptif keseluruhan
  - describe_per_class.csv        : mean/std/median per kelas
  - data_quality_report.csv       : laporan null/inf per kolom (pra-cleaning)
  - summary.md                    : ringkasan temuan utama

Catatan: statistik & plot dihitung SETELAH filter kelas & cleaning tetapi
SEBELUM imputasi, agar kondisi "kekasaran" data tetap terlihat; baris NaN
di-drop per fitur saat plotting saja.

Jalankan: python eda.py [--raw-dir data/raw] [--out logs/eda] [--sample 200000]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import DATA
from data_loading import load_unsw_nb15
from preprocessing import filter_classes, clean_data, coerce_numeric_columns


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def main():
    parser = argparse.ArgumentParser(description="EDA UNSW-NB15 (proposal 6.c)")
    parser.add_argument("--raw-dir", type=str, default=DATA.raw_dir)
    parser.add_argument("--out", type=str, default=os.path.join("logs", "eda"))
    parser.add_argument("--sample", type=int, default=200_000,
                        help="ukuran sampel untuk plot (statistik tetap pakai data penuh)")
    args = parser.parse_args()

    _ensure_dir(args.out)
    hist_dir = _ensure_dir(os.path.join(args.out, "histograms"))
    box_dir = _ensure_dir(os.path.join(args.out, "boxplots"))

    print("[EDA] Memuat data mentah ...")
    raw_df = load_unsw_nb15(args.raw_dir)
    n_raw = len(raw_df)

    # --- Laporan kualitas data (sebelum cleaning) ---
    coerced = coerce_numeric_columns(raw_df)
    numeric_raw = coerced.select_dtypes(include=[np.number]).columns.tolist()
    nan_count = coerced[numeric_raw].isna().sum()
    inf_count = pd.Series(
        np.isinf(coerced[numeric_raw].to_numpy(dtype=float)).sum(axis=0),
        index=numeric_raw,
    )
    dup_count = int(raw_df.duplicated().sum())
    quality = pd.DataFrame({
        "null_count": nan_count,
        "null_pct": (nan_count / n_raw * 100).round(3),
        "inf_count": inf_count,
    }).sort_values("null_count", ascending=False)
    quality.loc["__total__"] = [
        int(nan_count.sum()), round(nan_count.sum() / n_raw * 100, 3), int(inf_count.sum()),
    ]
    quality.to_csv(os.path.join(args.out, "data_quality_report.csv"))
    print(f"[EDA] baris mentah: {n_raw:,} | duplikat: {dup_count:,} "
          f"| total null numerik: {int(nan_count.sum()):,}")

    # --- Filter 4 kelas + cleaning (NaN numerik dibiarkan untuk EDA) ---
    df = filter_classes(raw_df)
    df = clean_data(df)
    n_kept = len(df)
    print(f"[EDA] baris setelah filter {list(DATA.classes_used)} + cleaning: {n_kept:,}")

    # --- Distribusi kelas & imbalance ratio (Bagian 6.c.i) ---
    counts = df[DATA.target_col].value_counts().reindex(DATA.classes_used).fillna(0).astype(int)
    ir = counts.max() / max(counts.min(), 1)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(counts.index, counts.values,
                  color=["#4C72B0", "#DD8452", "#55A868", "#C44E52"])
    for b, c in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, c, f"{c:,}\n({c / n_kept * 100:.1f}%)",
                ha="center", va="bottom", fontsize=9)
    ax.set_title(f"Distribusi kelas (imbalance ratio max:min = {ir:.1f}:1)")
    ax.set_ylabel("Jumlah baris")
    ax.set_ylim(0, counts.max() * 1.20)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "class_distribution.png"), dpi=150)
    plt.close(fig)

    # --- Statistik deskriptif (Bagian 6.c.ii) ---
    plot_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in set(DATA.time_cols) | {DATA.target_col, DATA.binary_label_col}
    ]
    df[plot_cols].describe().T.to_csv(os.path.join(args.out, "describe_overall.csv"))
    df.groupby(DATA.target_col)[plot_cols].agg(["mean", "std", "median"]).to_csv(
        os.path.join(args.out, "describe_per_class.csv")
    )

    # --- Histogram + density & boxplot per kelas (Bagian 6.c.iii; sampel) ---
    plot_df = df.sample(n=min(args.sample, len(df)), random_state=42)
    for c in plot_cols:
        vals = plot_df[c].dropna()
        if len(vals) < 10:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(vals, bins=50, density=True, alpha=0.6, color="#4C72B0")
        try:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(vals)
            xs = np.linspace(vals.min(), vals.max(), 200)
            ax.plot(xs, kde(xs), color="#C44E52", lw=1.5, label="density")
            ax.legend(fontsize=8)
        except Exception:
            pass  # fitur konstan/varian-nol tidak bisa di-KDE; histogram tetap digambar
        ax.set_title(f"Distribusi {c}")
        fig.tight_layout()
        fig.savefig(os.path.join(hist_dir, f"{c}.png"), dpi=120)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 4))
        data_by_class = [
            plot_df.loc[plot_df[DATA.target_col] == cls, c].dropna()
            for cls in DATA.classes_used
        ]
        ax.boxplot(data_by_class, showfliers=False)
        ax.set_xticklabels(list(DATA.classes_used), fontsize=8)
        ax.set_title(f"{c} per kelas (outlier disembunyikan)")
        fig.tight_layout()
        fig.savefig(os.path.join(box_dir, f"{c}_by_class.png"), dpi=120)
        plt.close(fig)

    # Grid histogram ringkas (satu gambar untuk semua fitur)
    n = len(plot_cols)
    ncols = 6
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 2.4 * nrows))
    for ax, c in zip(np.ravel(axes), plot_cols):
        ax.hist(plot_df[c].dropna(), bins=40, color="#4C72B0")
        ax.set_title(c, fontsize=8)
    for ax in np.ravel(axes)[n:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "histograms_grid.png"), dpi=110)
    plt.close(fig)

    # --- Heatmap korelasi Pearson (Bagian 6.c.iv) ---
    corr = df[plot_cols].corr()
    fig, ax = plt.subplots(figsize=(13, 11))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(plot_cols)))
    ax.set_yticks(range(len(plot_cols)))
    ax.set_xticklabels(plot_cols, rotation=90, fontsize=6)
    ax.set_yticklabels(plot_cols, fontsize=6)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Korelasi Pearson antar fitur")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "correlation_heatmap.png"), dpi=150)
    plt.close(fig)
    corr.to_csv(os.path.join(args.out, "correlation_matrix.csv"))

    # --- Ringkasan markdown ---
    with open(os.path.join(args.out, "summary.md"), "w", encoding="utf-8") as f:
        f.write("# Ringkasan EDA UNSW-NB15\n\n")
        f.write(f"- Baris mentah: **{n_raw:,}**; duplikat: **{dup_count:,}**; "
                f"setelah filter {list(DATA.classes_used)} & cleaning: **{n_kept:,}**\n")
        f.write("- Statistik dihitung sebelum imputasi (NaN tidak diisi) agar "
                "kondisi mentah tetap terlihat.\n\n")
        f.write("## Distribusi kelas\n\n| Kelas | Jumlah | Persen |\n|---|---:|---:|\n")
        for cls, c in counts.items():
            f.write(f"| {cls} | {c:,} | {c / n_kept * 100:.2f}% |\n")
        f.write(f"\nImbalance ratio (max:min) = **{ir:.1f}**\n\n")
        f.write("## Kolom dengan null terbanyak (sebelum imputasi)\n\n```\n")
        f.write(quality.head(10).to_string())
        f.write("\n```\n\n## Output\n\n- histograms/, boxplots/, histograms_grid.png, "
                "correlation_heatmap.png, describe_overall.csv, describe_per_class.csv, "
                "data_quality_report.csv, correlation_matrix.csv\n")
    print(f"[EDA] Selesai. Semua output di {args.out}")


if __name__ == "__main__":
    main()
