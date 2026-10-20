"""Hitung ROC-AUC / PR-AUC dari checkpoint run yang SUDAH ADA (tanpa training
ulang). Data, split, skaler, dan Isolation Forest direkonstruksi persis seperti
main.py (semuanya deterministik), lalu tft_base.pt dan tft_if.pt dimuat dari
folder run. Keluaran: <run_dir>/threshold_free.json.

Contoh:
    python eval_auc_from_run.py logs/runs/20261020-121115_W10_T3_seed42 \\
        logs/runs/20261020-175821_W10_T3_seed43
Semua run yang diberikan harus memakai W dan T yang sama (data dibangun sekali).
Skrip menampilkan selisih F1 macro dengan results_summary.json sebagai cek
bahwa rekonstruksi identik (harus ~0).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import torch

from config import DATA, WINDOW, MODEL, TRAIN
from data_loading import load_unsw_nb15
from preprocessing import run_preprocessing
from isolation_forest_module import integrate_isolation_forest
from sliding_window import build_windows
from tft_model import TemporalFusionTransformer
from evaluate import evaluate_model

p = argparse.ArgumentParser()
p.add_argument("run_dirs", nargs="+")
p.add_argument("--raw-dir", type=str, default=DATA.raw_dir)
a = p.parse_args()

cfgs = []
for d in a.run_dirs:
    with open(os.path.join(d, "results_summary.json")) as f:
        cfgs.append(json.load(f))
WT = {(c["config"]["W"], c["config"]["T"]) for c in cfgs}
if len(WT) != 1:
    sys.exit(f"Semua run harus memakai W,T yang sama, ditemukan: {WT}")
WINDOW.W, WINDOW.T = next(iter(WT))

print("Membangun data ...", flush=True)
raw_df = load_unsw_nb15(a.raw_dir)
train_df, val_df, test_df, art = run_preprocessing(raw_df)
feature_cols = art.feature_cols
cat_features = art.categorical_feature_indices or {}
train_if, val_if, test_if, feature_cols_if, _ = integrate_isolation_forest(
    train_df, val_df, test_df, feature_cols
)
data = {
    "TFT_base": (build_windows(val_df, feature_cols), build_windows(test_df, feature_cols), len(feature_cols)),
    "TFT_IF": (build_windows(val_if, feature_cols_if), build_windows(test_if, feature_cols_if), len(feature_cols_if)),
}
device = torch.device(TRAIN.device if torch.cuda.is_available() else "cpu")

for d, summ in zip(a.run_dirs, cfgs):
    c = summ["config"]
    out = {"run_id": c.get("run_id", os.path.basename(d)), "W": c["W"], "T": c["T"],
           "seed": c.get("seed")}
    for name, ckpt, key in [("TFT_base", "tft_base.pt", "results_TFT_base"),
                            ("TFT_IF", "tft_if.pt", "results_TFT_IF")]:
        (Xv, Yv), (Xt, Yt), nfeat = data[name]
        model = TemporalFusionTransformer(
            num_features=nfeat, hidden_size=c["hidden_size"],
            num_classes=MODEL.num_classes, T=c["T"], lstm_layers=MODEL.lstm_layers,
            attn_heads=c["attn_heads"], dropout=c["dropout"],
            categorical_features=cat_features,
        ).to(device)
        model.load_state_dict(torch.load(os.path.join(d, ckpt), map_location=device))
        res_t = evaluate_model(model, Xt, Yt)
        res_v = evaluate_model(model, Xv, Yv)
        diff = abs(res_t["f1_macro"] - summ[key]["f1_macro"])
        print(f"{out['run_id']} {name}: cek F1 macro uji selisih = {diff:.2e}"
              + ("  <-- TIDAK IDENTIK, periksa rekonstruksi" if diff > 1e-3 else ""), flush=True)
        out[name] = {"test": res_t["threshold_free"], "val": res_v["threshold_free"],
                     "f1_macro_check_diff": diff}
    with open(os.path.join(d, "threshold_free.json"), "w") as f:
        json.dump(out, f, indent=2)
    m = {n: out[n]["test"]["macro"] for n in ("TFT_base", "TFT_IF")}
    nv = {n: out[n]["test"]["normal_vs_attack"] for n in ("TFT_base", "TFT_IF")}
    print(f"  uji ROC-AUC makro: base={m['TFT_base']['roc_auc']:.4f} IF={m['TFT_IF']['roc_auc']:.4f} | "
          f"PR-AUC makro: base={m['TFT_base']['pr_auc']:.4f} IF={m['TFT_IF']['pr_auc']:.4f} | "
          f"PR-AUC Normal-vs-serangan: base={nv['TFT_base']['pr_auc']:.4f} IF={nv['TFT_IF']['pr_auc']:.4f}", flush=True)
