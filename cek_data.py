"""
cek_data_asli.py
Pemeriksaan data UNSW-NB15 asli sebelum bimbingan. Jalankan dari ROOT repositori
(folder yang berisi main.py dan data/raw):

    python cek_data_asli.py            # memakai data/raw dan src/config.py
    python cek_data_asli.py --raw-dir D:/data/unsw

Yang diperiksa:
  1. Rentang waktu (stime) tiap berkas dan apakah keempat berkas berurutan atau tumpang tindih.
  2. Jumlah rekaman per kelas (empat kelas) per berkas, dengan aturan pembersihan label yang sama dengan kode.
  3. Pembagian 70:15:15 per berkas (meniru preprocessing.chronological_split): jumlah per kelas
     dan rentang waktu tiap split, serta apakah data uji berkas manapun lebih awal daripada data latih berkas lain.
  4. Persistensi label: berapa persen pasangan rekaman berurutan (urut stime) bersama kelas yang sama,
     panjang rata-rata run, dan akurasi dua pembanding sederhana (kelas mayoritas, "sama dengan rekaman terakhir").
Keluaran: layar dan logs/cek_data_asli.md.

Catatan: pembersihan duplikat penuh (seluruh 49 kolom) tidak ditiru agar skrip ringan; angka bisa
sedikit berbeda dari keluaran main.py setelah drop_duplicates.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from src.config import DATA, PREPROC  # noqa: E402

KELAS = list(DATA.classes_used)
OUT = []


def say(s=""):
    print(s)
    OUT.append(s)


def muat(raw_dir):
    frames = []
    for f in DATA.csv_files:
        d = pd.read_csv(os.path.join(raw_dir, f), header=None, names=list(DATA.columns),
                        usecols=["stime", "attack_cat", "label"], low_memory=False,
                        encoding="utf-8-sig", on_bad_lines="skip")
        # Jika berkas ternyata punya baris header, baris itu jadi NaN pada stime dan dibuang.
        d["stime"] = pd.to_numeric(d["stime"], errors="coerce")
        d = d.dropna(subset=["stime"]).copy()
        cat = d["attack_cat"].fillna("").astype(str).str.strip()
        lab = pd.to_numeric(d["label"], errors="coerce")
        kosong = cat.isin(["nan", "", "None"])
        salah = int((kosong & (lab == 1)).sum())
        if salah:
            say(f"PERINGATAN {f}: {salah} baris attack_cat kosong tetapi label=1")
        cat = cat.where(~kosong, "Normal")
        kanon = {c.lower(): c for c in KELAS}
        cat = cat.map(lambda v: kanon.get(v.lower(), v))
        d["kelas"] = cat
        d["berkas"] = f
        frames.append(d[["berkas", "stime", "kelas"]])
    return frames


def persist(segmen):
    """Pembanding 'sama dengan rekaman terakhir' pada daftar segmen label (tiap segmen berurutan waktu).
    Prediksi untuk rekaman ke-i+1 adalah kelas rekaman ke-i, tidak melintasi batas segmen."""
    yt = np.concatenate([y[1:] for y in segmen if len(y) > 1])
    yp = np.concatenate([y[:-1] for y in segmen if len(y) > 1])
    return float(np.mean(yt == yp)), float(f1_score(yt, yp, labels=KELAS, average="macro", zero_division=0))


def tgl(t):
    return pd.to_datetime(t, unit="s").strftime("%Y-%m-%d %H:%M:%S")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=DATA.raw_dir)
    args = ap.parse_args()
    frames = muat(args.raw_dir)

    say("# Hasil cek data UNSW-NB15\n")
    say("## 1. Rentang waktu per berkas (UTC, semua kelas)")
    say("| Berkas | Rekaman | stime min | stime max | sudah terurut |")
    say("|---|---|---|---|---|")
    rentang = {}
    for d in frames:
        f = d["berkas"].iloc[0]
        rentang[f] = (d["stime"].min(), d["stime"].max())
        urut = bool(d["stime"].is_monotonic_increasing)
        say(f"| {f} | {len(d):,} | {tgl(d['stime'].min())} | {tgl(d['stime'].max())} | {'ya' if urut else 'tidak'} |")
    fl = list(rentang)
    say("")
    for i in range(len(fl) - 1):
        a, b = rentang[fl[i]], rentang[fl[i + 1]]
        tumpang = a[1] > b[0]
        say(f"- {fl[i]} dan {fl[i+1]}: " + ("TUMPANG TINDIH" if tumpang else "berurutan")
            + f" (selisih {(b[0]-a[1])/3600:.1f} jam antara akhir yang pertama dan awal yang kedua)")

    say("\n## 2. Jumlah rekaman empat kelas per berkas")
    say("| Berkas | " + " | ".join(KELAS) + " | kelas lain (dibuang) |")
    say("|---|" + "---|" * (len(KELAS) + 1))
    empat = []
    for d in frames:
        f = d["berkas"].iloc[0]
        vc = d["kelas"].value_counts()
        say(f"| {f} | " + " | ".join(f"{int(vc.get(k, 0)):,}" for k in KELAS)
            + f" | {int(vc.drop([k for k in KELAS if k in vc.index]).sum()):,} |")
        e = d[d["kelas"].isin(KELAS)].sort_values("stime", kind="stable").reset_index(drop=True)
        empat.append(e)

    say("\n## 3. Pembagian 70:15:15 per berkas (meniru kode)")
    say("| Berkas | Split | Rekaman | " + " | ".join(KELAS) + " | stime min | stime max |")
    say("|---|---|---|" + "---|" * len(KELAS) + "---|---|")
    bagian = {"train": [], "val": [], "test": []}
    for e in empat:
        n = len(e)
        ntr, nva = int(n * PREPROC.train_ratio), int(n * PREPROC.val_ratio)
        bagian["train"].append(e.iloc[:ntr])
        bagian["val"].append(e.iloc[ntr:ntr + nva])
        bagian["test"].append(e.iloc[ntr + nva:])
        for nama in ("train", "val", "test"):
            g = bagian[nama][-1]
            vc = g["kelas"].value_counts()
            say(f"| {e['berkas'].iloc[0]} | {nama} | {len(g):,} | "
                + " | ".join(f"{int(vc.get(k, 0)):,}" for k in KELAS)
                + f" | {tgl(g['stime'].min())} | {tgl(g['stime'].max())} |")
    say("")
    tot = {k: pd.concat(v) for k, v in bagian.items()}
    say("Total per split:")
    for nama, g in tot.items():
        vc = g["kelas"].value_counts()
        say(f"- {nama}: {len(g):,} rekaman; " + ", ".join(f"{k}={int(vc.get(k, 0)):,}" for k in KELAS))
    sedikit = [(n, k, int(g['kelas'].value_counts().get(k, 0))) for n, g in tot.items() for k in KELAS
               if g['kelas'].value_counts().get(k, 0) < 1000]
    say("- Kelas dengan kurang dari 1.000 rekaman pada suatu split: "
        + (", ".join(f"{n}/{k}={c}" for n, k, c in sedikit) if sedikit else "tidak ada"))
    tmax_latih = max(g["stime"].max() for g in bagian["train"])
    tmin_uji = min(g["stime"].min() for g in bagian["test"])
    say(f"- Data latih paling akhir: {tgl(tmax_latih)}; data uji paling awal: {tgl(tmin_uji)}. "
        + ("Ada data uji yang LEBIH AWAL daripada sebagian data latih (urutan global tidak terjaga)."
           if tmin_uji < tmax_latih else "Seluruh data uji lebih akhir daripada seluruh data latih."))

    say("\n## 4. Persistensi label pada urutan stime (empat kelas, per berkas)")
    say("| Berkas | Pasangan berurutan sekelas | Panjang run rata-rata | Akurasi kelas mayoritas | Akurasi 'sama dengan rekaman terakhir' |")
    say("|---|---|---|---|---|")
    for e in empat:
        y = e["kelas"].values
        sama = float(np.mean(y[1:] == y[:-1]))
        run = len(y) / (1 + int(np.sum(y[1:] != y[:-1])))
        mayor = float(e["kelas"].value_counts(normalize=True).iloc[0])
        say(f"| {e['berkas'].iloc[0]} | {sama*100:.2f}% | {run:.2f} | {mayor*100:.2f}% | {sama*100:.2f}% |")
    say("\nBacaan: jika 'sama dengan rekaman terakhir' jauh di atas kelas mayoritas, label sangat persisten "
        "dan perlu dibahas bahwa pembanding tersebut harus dikalahkan model.")

    say("\n## 5. Pembanding persistensi pada data uji (skema per berkas yang dipakai kode)")
    say("| Split | Akurasi 'sama dengan rekaman terakhir' | F1 macro pembanding itu |")
    say("|---|---|---|")
    for nama in ("train", "val", "test"):
        a, f = persist([g["kelas"].values for g in bagian[nama]])
        say(f"| {nama} | {a*100:.2f}% | {f:.4f} |")
    say("\nModel harus mengalahkan angka F1 macro pembanding ini pada data uji agar klaim berarti.")

    say("\n## 6. Alternatif: pembagian kronologis global (semua berkas digabung menurut stime, lalu dipotong 70:15:15)")
    gab = pd.concat(empat).sort_values("stime", kind="stable").reset_index(drop=True)
    n = len(gab)
    ntr, nva = int(n * PREPROC.train_ratio), int(n * PREPROC.val_ratio)
    glob = {"train": gab.iloc[:ntr], "val": gab.iloc[ntr:ntr + nva], "test": gab.iloc[ntr + nva:]}
    say("| Split | Rekaman | " + " | ".join(KELAS) + " | stime min | stime max | berkas yang terlibat |")
    say("|---|---|" + "---|" * len(KELAS) + "---|---|---|")
    for nama, g in glob.items():
        vc = g["kelas"].value_counts()
        say(f"| {nama} | {len(g):,} | " + " | ".join(f"{int(vc.get(k, 0)):,}" for k in KELAS)
            + f" | {tgl(g['stime'].min())} | {tgl(g['stime'].max())} | "
            + ", ".join(sorted(g['berkas'].unique())) + " |")
    segs = {nama: [x["kelas"].values for _, x in g.groupby("berkas", sort=False)] for nama, g in glob.items()}
    a, f = persist(segs["test"])
    say(f"\nPembanding persistensi pada data uji global: akurasi {a*100:.2f}%, F1 macro {f:.4f}.")

    say("\n## 7. Celah waktu terbesar dalam tiap berkas (stime berurutan)")
    say("| Berkas | Celah terbesar (jam) | Dari | Sampai |")
    say("|---|---|---|---|")
    for e in empat:
        t = np.sort(e["stime"].values)
        dif = np.diff(t)
        idx = np.argsort(dif)[::-1][:3]
        for i in idx:
            say(f"| {e['berkas'].iloc[0]} | {dif[i]/3600:.2f} | {tgl(t[i])} | {tgl(t[i+1])} |")
    say("\nJendela yang melintasi celah besar menggabungkan dua sesi perekaman yang berbeda; jumlahnya hanya sekitar W+T-1 per celah.")
    os.makedirs("logs", exist_ok=True)
    with open("logs/cek_data_asli.md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(OUT) + "\n")
    print("\nTersimpan: logs/cek_data_asli.md")


if __name__ == "__main__":
    main()
