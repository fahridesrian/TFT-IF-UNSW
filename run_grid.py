"""Grid (W,T) pada KEDUA model (A3). Menjalankan main.py per kombinasi;
tiap run punya folder sendiri: logs/runs/<tanggal-jam>_W{W}_T{T}_seed{S}_<grid-id>/
(results_summary.json, tft_base.pt, tft_if.pt, run.log); tidak ada yang tertimpa.
Setelah selesai, pilih (W,T) terbaik tiap model dari F1-macro VALIDASI
(results_val_TFT_*), bukan dari data uji.

Contoh: python run_grid.py --epochs 30
        python run_grid.py --windows 10 25 --horizons 1 3 --seed 42
        python run_grid.py --summarize-only --grid-id grid20261020-103000
"""
import argparse
import glob
import json
import os
import subprocess
import sys
from datetime import datetime

p = argparse.ArgumentParser()
p.add_argument("--windows", type=int, nargs="+", default=[10, 25, 50])
p.add_argument("--horizons", type=int, nargs="+", default=[1, 3])
p.add_argument("--seed", type=int, default=42)
p.add_argument("--epochs", type=int, default=None)
p.add_argument("--raw-dir", type=str, default=None)
p.add_argument("--grid-id", type=str, default=None,
               help="ID grid (default: grid<tanggal-jam> baru). Dipakai sebagai "
                    "--tag tiap run, juga untuk --summarize-only grid lama")
p.add_argument("--summarize-only", action="store_true")
a = p.parse_args()

grid_id = a.grid_id or f"grid{datetime.now().strftime('%Y%m%d-%H%M%S')}"
if a.summarize_only and not a.grid_id:
    sys.exit("--summarize-only butuh --grid-id (lihat nama folder di logs/runs/)")
print(f"Grid ID: {grid_id}", flush=True)

if not a.summarize_only:
    for W in a.windows:
        for T in a.horizons:
            cmd = [sys.executable, "main.py", "--window", str(W),
                   "--horizon", str(T), "--seed", str(a.seed),
                   "--tag", grid_id]
            if a.epochs:
                cmd += ["--epochs", str(a.epochs)]
            if a.raw_dir:
                cmd += ["--raw-dir", a.raw_dir]
            print(">>", " ".join(cmd), flush=True)
            subprocess.run(cmd, check=True)

rows = []
for f in sorted(glob.glob(os.path.join(
        "logs", "runs", f"*_{grid_id}*", "results_summary.json"))):
    d = json.load(open(f))
    c = d["config"]
    rows.append((c["W"], c["T"],
                 d["results_val_TFT_base"]["f1_macro"],
                 d["results_val_TFT_IF"]["f1_macro"]))
print("\nW   T   val F1macro base   val F1macro IF")
for W, T, b, i in rows:
    print(f"{W:<3} {T:<3} {b:<18.4f} {i:.4f}")
if rows:
    bb = max(rows, key=lambda r: r[2]); bi = max(rows, key=lambda r: r[3])
    print(f"\nTerbaik TFT-base: W={bb[0]}, T={bb[1]} (val F1 macro {bb[2]:.4f})")
    print(f"Terbaik TFT-IF  : W={bi[0]}, T={bi[1]} (val F1 macro {bi[3]:.4f})")
    print("Catatan: T berbeda = tugas berbeda; bandingkan antarmodel pada T sama.")
