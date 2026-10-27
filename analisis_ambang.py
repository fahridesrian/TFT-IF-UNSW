"""
analisis_ambang.py
Analisis ambang keputusan TANPA melatih ulang. Memuat checkpoint run yang sudah
ada (tft_base.pt, tft_if.pt), lalu membandingkan pada data uji:

  argmax   : prediksi apa adanya (yang dilaporkan sekarang).
  prior    : skor kelas k = P(k) * pi_k^alpha, pi_k = proporsi kelas pada data latih;
             alpha dipilih di VALIDASI (F1 macro), lalu dipakai di uji.
  ambang   : serangan bila 1 - P(Normal) > tau; kelas serangan = argmax di antara
             tiga kelas serangan; tau dipilih di VALIDASI (F1 macro), lalu dipakai di uji.

Pembanding skor Isolation Forest mentah (hanya biner Normal vs serangan):
  skor rekaman TERAKHIR di jendela, dibandingkan dengan label rekaman BERIKUTNYA
  (peramalan, informasi yang sama dengan model) dan dengan label rekaman itu sendiri
  (deteksi, bukan tugas model; hanya untuk konteks).

Skrip ini hanya membaca. Tidak ada berkas repositori yang diubah atau ditimpa;
keluaran memakai nama unik berstempel waktu di logs/.

Taruh berkas ini di ROOT repositori (sejajar eval_auc_from_run.py), lalu:

    python analisis_ambang.py logs/runs/<run_seed42> logs/runs/<run_seed43> ...

Semua run harus memakai W dan T yang sama. Kirim logs/analisis_ambang_*.md ke Claude.
"""
import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
from sklearn.metrics import (average_precision_score, f1_score, roc_auc_score)

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

from config import DATA  # noqa: E402

NAMES = list(DATA.classes_used)
NORMAL = NAMES.index("Normal")
C = len(NAMES)


def binary_metrics(y_true, y_pred):
    """Normal (0) melawan serangan (1) dari label empat kelas."""
    yt = (y_true != NORMAL)
    yp = (y_pred != NORMAL)
    tp = int((yt & yp).sum())
    fp = int((~yt & yp).sum())
    fn = int((yt & ~yp).sum())
    tn = int((~yt & ~yp).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    far = fp / (fp + tn) if fp + tn else float("nan")
    return {"prec_serangan": prec, "recall_serangan": rec, "f1_serangan": f1, "FAR": far}


def four_class_metrics(y_true, y_pred):
    out = {
        "f1_macro": float(f1_score(y_true, y_pred, labels=list(range(C)), average="macro", zero_division=0)),
        "acc": float((y_true == y_pred).mean()),
    }
    out.update(binary_metrics(y_true, y_pred))
    return out


def predict_prior(p, pi, alpha):
    return (p * (pi[None, :] ** alpha)).argmax(axis=1)


def predict_tau(p, tau):
    attack_arg = p[:, 1:].argmax(axis=1) + 1
    return np.where(1.0 - p[:, NORMAL] > tau, attack_arg, NORMAL)


def analyze(p_val, y_val, p_te, y_te, pi):
    """p_*: (n, C) peluang; y_*: (n,) label. Parameter dipilih di validasi."""
    res = {}
    res["argmax"] = {"param": None,
                     "val": four_class_metrics(y_val, p_val.argmax(1)),
                     "test": four_class_metrics(y_te, p_te.argmax(1))}
    best = max(np.linspace(0, 1, 11),
               key=lambda a: f1_score(y_val, predict_prior(p_val, pi, a), labels=list(range(C)),
                                      average="macro", zero_division=0))
    res["prior"] = {"param": {"alpha": float(best)},
                    "val": four_class_metrics(y_val, predict_prior(p_val, pi, best)),
                    "test": four_class_metrics(y_te, predict_prior(p_te, pi, best))}
    taus = np.linspace(0.01, 0.99, 99)
    bt = max(taus, key=lambda t: f1_score(y_val, predict_tau(p_val, t), labels=list(range(C)),
                                          average="macro", zero_division=0))
    res["ambang"] = {"param": {"tau": float(bt)},
                     "val": four_class_metrics(y_val, predict_tau(p_val, bt)),
                     "test": four_class_metrics(y_te, predict_tau(p_te, bt))}
    return res


def if_baseline(s_val, y_val, s_te, y_te):
    """Skor IF mentah, ambang tunggal dipilih di validasi (F1 serangan)."""
    yb_val = (y_val != NORMAL).astype(int)
    yb_te = (y_te != NORMAL).astype(int)
    cands = np.unique(np.quantile(s_val, np.linspace(0.5, 0.999, 200)))
    def f1b(t):
        return binary_metrics(y_val, np.where(s_val > t, 1, 0))["f1_serangan"]
    t = max(cands, key=f1b)
    out = {"ambang": float(t),
           "val": binary_metrics(y_val, np.where(s_val > t, 1, 0)),
           "test": binary_metrics(y_te, np.where(s_te > t, 1, 0))}
    out["val"]["ROC-AUC"] = float(roc_auc_score(yb_val, s_val))
    out["test"]["ROC-AUC"] = float(roc_auc_score(yb_te, s_te))
    out["test"]["PR-AUC"] = float(average_precision_score(yb_te, s_te))
    return out


def fmt(x):
    return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.4f}"


def main():
    import torch
    from config import WINDOW, MODEL, TRAIN
    from data_loading import load_unsw_nb15
    from preprocessing import run_preprocessing
    from isolation_forest_module import integrate_isolation_forest
    from sliding_window import build_windows, build_last_labels
    from tft_model import TemporalFusionTransformer
    from evaluate import predict_proba

    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--raw-dir", type=str, default=DATA.raw_dir)
    a = ap.parse_args()

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
        train_df, val_df, test_df, feature_cols)
    counts = np.bincount(train_df[DATA.target_col].to_numpy().astype(int), minlength=C).astype(float)
    pi = counts / counts.sum()

    data = {
        "TFT_base": (build_windows(val_df, feature_cols), build_windows(test_df, feature_cols), len(feature_cols)),
        "TFT_IF": (build_windows(val_if, feature_cols_if), build_windows(test_if, feature_cols_if), len(feature_cols_if)),
    }
    last_val = build_last_labels(val_df)
    last_te = build_last_labels(test_df)
    device = torch.device(TRAIN.device if torch.cuda.is_available() else "cpu")

    all_out = {"W": WINDOW.W, "T": WINDOW.T, "pi_latih": pi.tolist(), "runs": {}}
    md = ["# Analisis ambang keputusan (tanpa melatih ulang)", "",
          f"W={WINDOW.W}, T={WINDOW.T}. Proporsi kelas data latih pi = "
          + ", ".join(f"{n} {v:.4f}" for n, v in zip(NAMES, pi)) + ".",
          "Parameter (alpha, tau, ambang IF) dipilih di VALIDASI, angka di bawah adalah UJI kecuali disebut lain.", ""]

    # Skor IF mentah: rekaman terakhir di jendela, kolom terakhir pada TFT_IF
    (Xv_if, Yv), (Xt_if, Yt), _ = data["TFT_IF"]
    yv, yt = Yv.reshape(-1), Yt.reshape(-1)
    T = WINDOW.T
    s_val = np.repeat(Xv_if[:, -1, -1], T)
    s_te = np.repeat(Xt_if[:, -1, -1], T)
    ifb = if_baseline(s_val, yv, s_te, yt)
    # konteks: skor rekaman terakhir vs label rekaman itu sendiri (deteksi, bukan tugas model)
    nowc = float(roc_auc_score((last_te != NORMAL).astype(int), Xt_if[:, -1, -1]))
    persist = binary_metrics(yt, np.repeat(last_te, T))
    all_out["if_mentah"] = ifb
    all_out["if_deteksi_auc_uji"] = nowc

    md += ["## Skor IF mentah (biner Normal vs serangan, uji)", "",
           "| Cara | ROC-AUC | Presisi | Recall | F1 serangan | FAR |", "|---|---|---|---|---|---|",
           f"| Skor rekaman terakhir vs label BERIKUTNYA (peramalan), ambang {ifb['ambang']:.4f} | "
           f"{fmt(ifb['test']['ROC-AUC'])} | {fmt(ifb['test']['prec_serangan'])} | {fmt(ifb['test']['recall_serangan'])} | "
           f"{fmt(ifb['test']['f1_serangan'])} | {fmt(ifb['test']['FAR'])} |",
           f"| Skor rekaman terakhir vs label rekaman ITU SENDIRI (deteksi, konteks) | {fmt(nowc)} | - | - | - | - |",
           f"| Persistence (label terakhir sebagai ramalan) | - | {fmt(persist['prec_serangan'])} | "
           f"{fmt(persist['recall_serangan'])} | {fmt(persist['f1_serangan'])} | {fmt(persist['FAR'])} |", ""]

    for d, summ in zip(a.run_dirs, cfgs):
        c = summ["config"]
        rid = c.get("run_id", os.path.basename(d))
        all_out["runs"][rid] = {}
        md += [f"## Run {rid} (seed {c.get('seed')})", "",
               "| Model | Cara | Parameter | F1 macro | Akurasi | Presisi serangan | Recall serangan | F1 serangan | FAR |",
               "|---|---|---|---|---|---|---|---|---|"]
        for name, ckpt in [("TFT_base", "tft_base.pt"), ("TFT_IF", "tft_if.pt")]:
            (Xv, Yv_), (Xt, Yt_), nfeat = data[name]
            model = TemporalFusionTransformer(
                num_features=nfeat, hidden_size=c["hidden_size"], num_classes=MODEL.num_classes,
                T=c["T"], lstm_layers=MODEL.lstm_layers, attn_heads=c["attn_heads"],
                dropout=c["dropout"], categorical_features=cat_features).to(device)
            model.load_state_dict(torch.load(os.path.join(d, ckpt), map_location=device))
            pv = predict_proba(model, Xv).reshape(-1, C)
            pt = predict_proba(model, Xt).reshape(-1, C)
            res = analyze(pv, Yv_.reshape(-1), pt, Yt_.reshape(-1), pi)
            all_out["runs"][rid][name] = res
            for cara in ("argmax", "prior", "ambang"):
                m = res[cara]["test"]
                par = "-" if res[cara]["param"] is None else ", ".join(f"{k}={v:.2f}" for k, v in res[cara]["param"].items())
                md.append(f"| {name} | {cara} | {par} | {fmt(m['f1_macro'])} | {fmt(m['acc'])} | "
                          f"{fmt(m['prec_serangan'])} | {fmt(m['recall_serangan'])} | {fmt(m['f1_serangan'])} | {fmt(m['FAR'])} |")
            print(f"{rid} {name}: argmax F1={res['argmax']['test']['f1_macro']:.4f} "
                  f"prior F1={res['prior']['test']['f1_macro']:.4f} ambang F1={res['ambang']['test']['f1_macro']:.4f}", flush=True)
        md.append("")

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
    base = os.path.join(ROOT, "logs", f"analisis_ambang_W{WINDOW.W}_T{WINDOW.T}_{stamp}")
    with open(base + ".json", "w") as f:
        json.dump(all_out, f, indent=2)
    with open(base + ".md", "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print("Tertulis:", base + ".md")


if __name__ == "__main__":
    main()
