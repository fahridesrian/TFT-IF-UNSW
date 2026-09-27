"""
main.py
Orkestrator penuh pipeline penelitian, mengikuti diagram alur (Gambar 2)
dan langkah kerja (Bagian 6) proposal:

  load data -> praproses -> analisis/penanganan imbalance (class weighting
  saat training) -> integrasi Isolation Forest -> sliding window
  -> (opsional) hyperparameter tuning -> latih TFT-base & TFT-IF secara
  paralel -> evaluasi & bandingkan -> interpretasi (VSN + attention)
  -> simpan hasil ke logs/results_summary.json

Jalankan: python main.py [--tune] [--epochs N] [--raw-dir path]
"""
import argparse
import copy
import json
import os
import sys

# Memungkinkan `python main.py` dijalankan langsung dari root repo tanpa
# perlu `pip install -e .` lebih dulu (modul flat di src/).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import numpy as np
import torch

from config import DATA, WINDOW, MODEL, TRAIN, PATHS
from data_loading import load_unsw_nb15
from preprocessing import run_preprocessing
from isolation_forest_module import integrate_isolation_forest
from sliding_window import build_windows
from train import train_model
from evaluate import evaluate_model
from interpret import extract_interpretation, summarize_if_contribution
from hyperparameter_tuning import run_tuning


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", type=str, default=DATA.raw_dir)
    p.add_argument("--tune", action="store_true", help="Jalankan Optuna hyperparameter tuning")
    p.add_argument("--n-trials", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--window", type=int, default=None,
                   help="Override WINDOW.W (untuk ablasi W/T, proposal 6.g.v)")
    p.add_argument("--horizon", type=int, default=None,
                   help="Override WINDOW.T (untuk ablasi W/T, proposal 6.g.v)")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(PATHS.logs_dir, exist_ok=True)
    os.makedirs(PATHS.checkpoints_dir, exist_ok=True)

    verbose = not args.quiet
    train_cfg = copy.deepcopy(TRAIN)
    if args.epochs:
        train_cfg.epochs = args.epochs
    # Override W/T untuk ablasi tanpa menyentuh config.py (WINDOW adalah
    # singleton config yang juga dibaca sliding_window.build_windows).
    if args.window:
        WINDOW.W = args.window
    if args.horizon:
        WINDOW.T = args.horizon
    if verbose and (args.window or args.horizon):
        print(f"    Override window: W={WINDOW.W}, T={WINDOW.T}")

    # 1. Load data (Bagian 6.b)
    if verbose:
        print("[1/8] Memuat data mentah UNSW-NB15 ...")
    raw_df = load_unsw_nb15(args.raw_dir)

    # 2. Praproses (Bagian 6.d): filter kelas, cleaning, split kronologis, encode+scale
    if verbose:
        print("[2/8] Praproses data ...")
    train_df, val_df, test_df, artifacts = run_preprocessing(raw_df)
    feature_cols = artifacts.feature_cols  # belum termasuk if_score
    cat_features = artifacts.categorical_feature_indices or {}

    # Laporan distribusi kelas per split (keputusan A14c/2026-09-27):
    # transparansi imbalance sejak awal pipeline.
    split_class_distribution = {}
    for name, d in [("train", train_df), ("val", val_df), ("test", test_df)]:
        counts = d[DATA.target_col].value_counts().sort_index()
        split_class_distribution[name] = {
            DATA.classes_used[int(cls)]: int(cnt) for cls, cnt in counts.items()
        }
    if verbose:
        print("    Distribusi kelas per split:")
        for name, dist in split_class_distribution.items():
            print(f"      {name}: {dist}")

    # 3. Integrasi Isolation Forest (Bagian 6.f)
    if verbose:
        print("[3/8] Melatih Isolation Forest & menghitung skor anomali ...")
    train_if, val_if, test_if, feature_cols_if, iso_model = integrate_isolation_forest(
        train_df, val_df, test_df, feature_cols
    )

    # 4. Sliding window (Bagian 6.g) -- dibentuk dua kali: dengan & tanpa if_score
    if verbose:
        print("[4/8] Membentuk sliding window ...")
    Xtr_base, Ytr = build_windows(train_df, feature_cols)
    Xval_base, Yval = build_windows(val_df, feature_cols)
    Xte_base, Yte = build_windows(test_df, feature_cols)

    Xtr_if, Ytr_if = build_windows(train_if, feature_cols_if)
    Xval_if, Yval_if = build_windows(val_if, feature_cols_if)
    Xte_if, Yte_if = build_windows(test_if, feature_cols_if)

    if verbose:
        print(f"    TFT-base X_train shape: {Xtr_base.shape}, Y_train shape: {Ytr.shape}")
        print(f"    TFT-IF   X_train shape: {Xtr_if.shape}, Y_train shape: {Ytr_if.shape}")

    if len(Xtr_base) == 0 or len(Xtr_if) == 0:
        raise RuntimeError(
            "Sliding window menghasilkan 0 sampel. Cek ukuran data vs "
            "parameter WINDOW.W/WINDOW.T di config.py."
        )

    model_cfg = copy.deepcopy(MODEL)

    # 5. (Opsional) hyperparameter tuning dengan Optuna, dipakai untuk TFT-IF
    if args.tune:
        if verbose:
            print("[5/8] Menjalankan hyperparameter tuning (Optuna) pada TFT-IF ...")
        best_params, _ = run_tuning(
            Xtr_if, Ytr_if, Xval_if, Yval_if, num_features=Xtr_if.shape[-1],
            n_trials=args.n_trials, categorical_features=cat_features,
        )
        if verbose:
            print(f"    Best params: {best_params}")
        model_cfg.hidden_size = best_params["hidden_size"]
        model_cfg.dropout = best_params["dropout"]
        model_cfg.attn_heads = best_params[f"attn_heads_{best_params['hidden_size']}"]
        train_cfg.learning_rate = best_params["learning_rate"]
    else:
        if verbose:
            print("[5/8] Melewati tuning (gunakan --tune untuk mengaktifkan).")

    # 6. Latih TFT-base (kondisi kontrol) dan TFT-IF (dengan skor anomali)
    if verbose:
        print("[6/8] Melatih TFT-base ...")
    model_base, hist_base = train_model(
        Xtr_base, Ytr, Xval_base, Yval, num_features=Xtr_base.shape[-1],
        model_cfg=model_cfg, train_cfg=train_cfg, verbose=verbose,
        desc="TFT-base", show_progress=verbose,
        categorical_features=cat_features,
    )

    if verbose:
        print("[6/8] Melatih TFT-IF ...")
    model_if, hist_if = train_model(
        Xtr_if, Ytr_if, Xval_if, Yval_if, num_features=Xtr_if.shape[-1],
        model_cfg=model_cfg, train_cfg=train_cfg, verbose=verbose,
        desc="TFT-IF", show_progress=verbose,
        categorical_features=cat_features,
    )

    torch.save(model_base.state_dict(), os.path.join(PATHS.checkpoints_dir, "tft_base.pt"))
    torch.save(model_if.state_dict(), os.path.join(PATHS.checkpoints_dir, "tft_if.pt"))

    # 7. Evaluasi & bandingkan (Bagian 6.i)
    if verbose:
        print("[7/8] Mengevaluasi model pada data uji ...")
    results_base = evaluate_model(model_base, Xte_base, Yte)
    results_if = evaluate_model(model_if, Xte_if, Yte_if)

    # Metrik validasi untuk seleksi ablasi W/T (selektor = F1-macro validasi,
    # proposal 6.g.v; data uji tidak boleh dipakai memilih konfigurasi).
    val_results_base = evaluate_model(model_base, Xval_base, Yval)
    val_results_if = evaluate_model(model_if, Xval_if, Yval_if)

    comparison = {
        metric: {"TFT_base": results_base[metric], "TFT_IF": results_if[metric]}
        for metric in ["accuracy", "precision_macro", "recall_macro", "f1_macro",
                        "precision_weighted", "recall_weighted", "f1_weighted",
                        "false_alarm_rate"]
    }
    if verbose:
        print("    Perbandingan metrik (TFT-base vs TFT-IF):")
        for m, v in comparison.items():
            print(f"      {m}: base={v['TFT_base']:.4f}  if={v['TFT_IF']:.4f}")

    # 8. Interpretasi (Bagian 6.j) -- pakai subset data uji TFT-IF
    if verbose:
        print("[8/8] Mengekstraksi interpretabilitas (VSN + attention) dari TFT-IF ...")
    sample_n = min(256, len(Xte_if))
    interp = extract_interpretation(
        model_if, Xte_if[:sample_n], feature_names=feature_cols_if,
    )
    if_contribution = summarize_if_contribution(interp)

    summary = {
        "config": {
            "W": WINDOW.W, "T": WINDOW.T,
            "hidden_size": model_cfg.hidden_size,
            "dropout": model_cfg.dropout,
            "attn_heads": model_cfg.attn_heads,
            "learning_rate": train_cfg.learning_rate,
            "epochs_ran_base": len(hist_base["train_loss"]),
            "epochs_ran_if": len(hist_if["train_loss"]),
            "num_features_base": Xtr_base.shape[-1],
            "num_features_if": Xtr_if.shape[-1],
            "categorical_cardinalities": artifacts.categorical_cardinalities,
        },
        "split_class_distribution": split_class_distribution,
        "results_TFT_base": results_base,
        "results_TFT_IF": results_if,
        "results_val_TFT_base": val_results_base,
        "results_val_TFT_IF": val_results_if,
        "comparison": comparison,
        "top10_feature_importance_TFT_IF": interp["feature_importance_ranked"][:10],
        "if_score_feature_rank": if_contribution,
    }

    with open(PATHS.results_file, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    if verbose:
        print(f"\nSelesai. Ringkasan hasil disimpan di {PATHS.results_file}")
        print(f"Peringkat pentingnya fitur if_score: {if_contribution}")

    return summary


if __name__ == "__main__":
    main()
