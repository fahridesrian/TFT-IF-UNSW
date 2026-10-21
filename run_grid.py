"""Grid (W,T) pada KEDUA model (A3). Menjalankan main.py per kombinasi yang
BELUM ada untuk seed yang sama; kombinasi yang sudah punya run (mis. W=10,
T=3 seed 42) dilewati kecuali --force. Tiap run punya folder sendiri:
logs/runs/<tanggal-jam>_W{W}_T{T}_seed{S}_<tag>/ (results_summary.json,
tft_base.pt, tft_if.pt, run.log); tidak ada yang tertimpa.

Ringkasan membaca SEMUA run di logs/runs untuk seed itu (apa pun tag-nya),
mengambil yang terbaru per (W,T), lalu memilih (W,T) terbaik tiap model dari
F1-macro VALIDASI (bukan data uji). Bila run memuat metrik tanpa ambang,
PR-AUC makro validasi ikut ditampilkan sebagai pembanding (tidak dipakai memilih).

Contoh:
    python run_grid.py --windows 10 25 --horizons 1 3 --seed 42 --epochs 30
    python run_grid.py --summarize-only --windows 10 25 --horizons 1 3
"""
import argparse
import glob
import json
import os
import subprocess
import sys
from datetime import datetime

p = argparse.ArgumentParser()
p.add_argument("--windows", type=int, nargs="+", default=[10, 25])
p.add_argument("--horizons", type=int, nargs="+", default=[1, 3])
p.add_argument("--seed", type=int, default=42)
p.add_argument("--epochs", type=int, default=None)
p.add_argument("--raw-dir", type=str, default=None)
p.add_argument("--batch-size", type=int, default=None,
               help="Diteruskan ke main.py. Memori aktivasi naik linear dengan W x batch: "
                    "bila W=25 kehabisan memori GPU pada 4096, turunkan (mis. 2048)")
p.add_argument("--tag", type=str, default=None,
               help="Label akhir nama folder run (default: grid<tanggal-jam>)")
p.add_argument("--force", action="store_true",
               help="Jalankan ulang kombinasi yang sudah punya run")
p.add_argument("--summarize-only", action="store_true")
a = p.parse_args()

tag = a.tag or f"grid{datetime.now().strftime('%Y%m%d-%H%M%S')}"


def load_runs():
    """{(W, T): (started_at, folder, ringkasan)} untuk seed terpilih; terbaru menang."""
    found = {}
    for f in glob.glob(os.path.join("logs", "runs", "*", "results_summary.json")):
        try:
            d = json.load(open(f))
            c = d["config"]
        except (OSError, ValueError, KeyError):
            continue
        if c.get("seed") != a.seed or "results_val_TFT_base" not in d:
            continue
        key = (c["W"], c["T"])
        stamp = c.get("started_at", "")
        if key not in found or stamp > found[key][0]:
            found[key] = (stamp, os.path.dirname(f), d)
    return found


if not a.summarize_only:
    existing = load_runs()
    for W in a.windows:
        for T in a.horizons:
            if (W, T) in existing and not a.force:
                print(f"== W={W}, T={T}, seed {a.seed}: sudah ada "
                      f"({os.path.basename(existing[(W, T)][1])}), dilewati", flush=True)
                continue
            cmd = [sys.executable, "main.py", "--window", str(W),
                   "--horizon", str(T), "--seed", str(a.seed), "--tag", tag]
            if a.epochs:
                cmd += ["--epochs", str(a.epochs)]
            if a.raw_dir:
                cmd += ["--raw-dir", a.raw_dir]
            if a.batch_size:
                cmd += ["--batch-size", str(a.batch_size)]
            print(">>", " ".join(cmd), flush=True)
            subprocess.run(cmd, check=True)

runs = load_runs()
rows = []
for W in a.windows:
    for T in a.horizons:
        if (W, T) not in runs:
            continue
        d = runs[(W, T)][2]
        tf = d["results_val_TFT_base"].get("threshold_free"), d["results_val_TFT_IF"].get("threshold_free")
        pr = [x["macro"]["pr_auc"] if x else None for x in tf]
        rows.append((W, T, d["results_val_TFT_base"]["f1_macro"],
                     d["results_val_TFT_IF"]["f1_macro"], pr[0], pr[1]))

fmt = lambda v: f"{v:.4f}" if v is not None else "-"
print(f"\nSeed {a.seed}, data VALIDASI")
print("W   T   F1macro base  F1macro IF   PR-AUC makro base  PR-AUC makro IF")
for W, T, b, i, pb, pi in rows:
    print(f"{W:<3} {T:<3} {b:<13.4f} {i:<12.4f} {fmt(pb):<18} {fmt(pi)}")
if rows:
    bb = max(rows, key=lambda r: r[2]); bi = max(rows, key=lambda r: r[3])
    print(f"\nTerbaik TFT-base: W={bb[0]}, T={bb[1]} (val F1 macro {bb[2]:.4f})")
    print(f"Terbaik TFT-IF  : W={bi[0]}, T={bi[1]} (val F1 macro {bi[3]:.4f})")
    print("Catatan: T berbeda = tugas berbeda; bandingkan antarmodel pada T sama. "
          "F1 macro satu seed berfluktuasi ±0,01 antarseed: selisih lebih kecil "
          "dari itu bukan pembeda yang andal.")
else:
    print("Belum ada run untuk kombinasi ini.")
