"""Grid (W,T) pada KEDUA model (A3). Menjalankan main.py per kombinasi;
hasil tersimpan per kombinasi di logs/results_summary_W{W}_T{T}_seed{S}.json.
Setelah selesai, pilih (W,T) terbaik tiap model dari F1-macro VALIDASI
(results_val_TFT_*), bukan dari data uji.

Contoh: python run_grid.py --epochs 30
        python run_grid.py --windows 10 25 --horizons 1 3 --seed 42
"""
import argparse
import glob
import json
import subprocess
import sys

p = argparse.ArgumentParser()
p.add_argument("--windows", type=int, nargs="+", default=[10, 25, 50])
p.add_argument("--horizons", type=int, nargs="+", default=[1, 3])
p.add_argument("--seed", type=int, default=42)
p.add_argument("--epochs", type=int, default=None)
p.add_argument("--raw-dir", type=str, default=None)
p.add_argument("--summarize-only", action="store_true")
a = p.parse_args()

if not a.summarize_only:
    for W in a.windows:
        for T in a.horizons:
            cmd = [sys.executable, "main.py", "--window", str(W),
                   "--horizon", str(T), "--seed", str(a.seed)]
            if a.epochs:
                cmd += ["--epochs", str(a.epochs)]
            if a.raw_dir:
                cmd += ["--raw-dir", a.raw_dir]
            print(">>", " ".join(cmd), flush=True)
            subprocess.run(cmd, check=True)

rows = []
for f in sorted(glob.glob(f"logs/results_summary_W*_T*_seed{a.seed}.json")):
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
