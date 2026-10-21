"""
cuplikan_praproses.py
Menjalankan praproses (src/preprocessing.py) langkah demi langkah pada data asli
dan mencetak angka serta cuplikan sebelum dan sesudah tiap langkah, untuk Subbab 3.4
(Praproses Data) dan contoh perhitungan manual (3.5). Skrip ini hanya membaca;
tidak mengubah berkas apa pun di repositori.

Jalankan dari ROOT repositori (folder yang berisi main.py dan data/raw):

    python cuplikan_praproses.py
    python cuplikan_praproses.py --raw-dir D:/data/unsw

Keluaran: layar dan logs/cuplikan_praproses.md. Kirim berkas .md itu ke Claude.

Empat rekaman cuplikan sama dengan Tabel cuplikan data mentah di Bab 2 (berkas
UNSW-NB15_1.csv). Jika salah satu tidak ditemukan atau terbuang sebagai duplikat,
skrip mencatatnya.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))
from config import DATA, PREPROC  # noqa: E402
from data_loading import load_unsw_nb15  # noqa: E402
import preprocessing as pp  # noqa: E402

OUT = []

# (srcip, dstip, stime, sbytes, dbytes, nama)
SAMPEL = [
    ("59.166.0.0", "149.171.126.6", 1421927414, 132, 164, "Rekaman 1 (Normal)"),
    ("175.45.176.2", "149.171.126.16", 1421927416, 918, 25552, "Rekaman 2 (Exploits)"),
    ("175.45.176.0", "149.171.126.14", 1421927420, 844, 268, "Rekaman 3 (Generic)"),
    ("175.45.176.2", "149.171.126.11", 1421927596, 1040, 0, "Rekaman 4 (Fuzzers)"),
]
KOLOM_TAMPIL = ["proto", "state", "service", "dur", "sbytes", "sttl", "dttl",
                "ct_srv_dst", "stime", "attack_cat"]


def say(s=""):
    print(s)
    OUT.append(s)


def tabel(df: pd.DataFrame, judul: str = ""):
    if judul:
        say(f"**{judul}**")
        say()
    say("```")
    say(df.to_string())
    say("```")
    say()


def cari(df: pd.DataFrame, nama_kolom_ip=True):
    """Indeks baris sampel pada df yang masih memuat srcip/dstip. None bila tak ada."""
    hasil = []
    for src, dst, st, sb, db, nama in SAMPEL:
        m = ((df["srcip"].astype(str).str.strip() == src)
             & (df["dstip"].astype(str).str.strip() == dst)
             & (pd.to_numeric(df["stime"], errors="coerce") == st)
             & (pd.to_numeric(df["sbytes"], errors="coerce") == sb)
             & (pd.to_numeric(df["dbytes"], errors="coerce") == db))
        idx = df.index[m]
        hasil.append(int(idx[0]) if len(idx) else None)
    return hasil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=None)
    args = ap.parse_args()

    say("# Cuplikan dan angka praproses (untuk Subbab 3.4)")
    say()

    # ---------------------------------------------------------------- 0. mentah
    say("## 0. Data mentah (setelah dibaca dari empat CSV)")
    raw = load_unsw_nb15(args.raw_dir)
    say(f"- Bentuk: {raw.shape[0]:,} baris, {raw.shape[1]} kolom (49 kolom data + `source_file`).")
    say("- Rekaman per berkas: " + ", ".join(
        f"{k}={v:,}" for k, v in raw["source_file"].value_counts().sort_index().items()))
    say("- Nilai unik `attack_cat` (apa adanya, tanda kutip menunjukkan spasi/kosong):")
    say("  " + repr(sorted(raw[DATA.target_col].astype(str).unique().tolist(), key=str)))
    say()
    idx0 = cari(raw)
    baris0 = raw.loc[[i for i in idx0 if i is not None], KOLOM_TAMPIL].copy()
    baris0[DATA.target_col] = baris0[DATA.target_col].map(repr)
    tabel(baris0, "Cuplikan mentah (nilai `attack_cat` ditampilkan dengan repr)")
    for (n, i) in zip([s[5] for s in SAMPEL], idx0):
        if i is None:
            say(f"- {n}: TIDAK ditemukan di data mentah.")
    say()

    # --------------------------------------------------------- 1. filter kelas
    say("## 1. Penyaringan empat kelas")
    f = pp.filter_classes(raw)
    say(f"- Sebelum: {len(raw):,} baris. Sesudah: {len(f):,} baris. Dibuang: {len(raw) - len(f):,} baris.")
    say("- Rekaman per kelas sesudah penyaringan: " + ", ".join(
        f"{k}={v:,}" for k, v in f[DATA.target_col].value_counts().items()))
    sebelum_kat = raw[DATA.target_col].astype(str).str.strip().replace({"nan": "", "None": ""})
    say("- Rekaman per kategori di data mentah (setelah trim spasi; kosong = Normal):")
    say("  " + ", ".join(f"{(k or '(kosong)')}={v:,}" for k, v in sebelum_kat.value_counts().items()))
    idx1 = cari(f)
    tabel(f.loc[[i for i in idx1 if i is not None], [DATA.target_col, "stime"]],
          "Cuplikan sesudah penyaringan (label sudah rapi dan Normal terisi)")

    # ------------------------------------------------------------- 2. bersih
    say("## 2. Pembersihan data")
    kon = pp.coerce_numeric_columns(f)
    kolom_num = kon.select_dtypes(include=[np.number]).columns.tolist()
    nan_per_kolom = kon[kolom_num].isna().sum()
    nan_per_kolom = nan_per_kolom[nan_per_kolom > 0].sort_values(ascending=False)
    say("- Nilai hilang (NaN) sesudah konversi numerik, per kolom (hanya yang > 0):")
    say("  " + (", ".join(f"{k}={v:,}" for k, v in nan_per_kolom.items()) or "tidak ada"))
    n_inf = int(np.isinf(kon[kolom_num].to_numpy(dtype=float)).sum())
    say(f"- Nilai tak hingga (inf/-inf): {n_inf:,} sel.")
    c = pp.clean_data(f)
    say(f"- Duplikat identik dibuang: {len(f) - len(c):,} baris ({len(f):,} menjadi {len(c):,}).")
    say("- Rekaman per kelas sebelum -> sesudah dedup: " + ", ".join(
        f"{k}: {f[DATA.target_col].value_counts().get(k, 0):,} -> {c[DATA.target_col].value_counts().get(k, 0):,}"
        for k in DATA.classes_used))
    idx2 = cari(c)
    for (n, i) in zip([s[5] for s in SAMPEL], idx2):
        if i is None:
            say(f"- {n}: tidak ada lagi sesudah pembersihan (terbuang sebagai duplikat).")
    say()

    # ------------------------------------------------------------- 3. urut
    say("## 3. Pengurutan kronologis")
    s = pp.sort_chronologically(c)
    n_unik = s["stime"].nunique()
    sama = s["stime"].duplicated(keep=False).sum()
    say(f"- `stime` unik: {n_unik:,} dari {len(s):,} baris; baris yang berbagi `stime` dengan baris lain: {sama:,} ({sama / len(s) * 100:.1f}%).")
    say("- Aturan pemecah seri: `stime`, lalu berkas asal (`source_file`), lalu urutan baris asli (pengurutan stabil).")
    say(f"- Rentang waktu: {pd.to_datetime(s['stime'].min(), unit='s')} sampai {pd.to_datetime(s['stime'].max(), unit='s')} (UTC).")
    say()
    idx3 = cari(s)
    tabel(s.loc[[i for i in idx3 if i is not None], ["source_file", "stime", DATA.target_col]]
          .assign(posisi_urut=[i for i in idx3 if i is not None]),
          "Posisi cuplikan pada urutan global")

    # ------------------------------------------------------------- 4. split
    say("## 4. Pembagian 70:15:15")
    tr, va, te = pp.chronological_split(s)
    for nama, d in (("latih", tr), ("validasi", va), ("uji", te)):
        say(f"- {nama}: {len(d):,} baris; {pd.to_datetime(d['stime'].min(), unit='s')} sampai {pd.to_datetime(d['stime'].max(), unit='s')}; "
            + ", ".join(f"{k}={v:,}" for k, v in d[DATA.target_col].value_counts().items())
            + f"; berkas asal: " + ", ".join(f"{k.split('_')[-1][0]}={v:,}" for k, v in d['source_file'].value_counts().sort_index().items()))
    say()
    n_tr, n_va = len(tr), len(va)

    def lokasi(g):
        if g is None:
            return None
        if g < n_tr:
            return ("latih", g)
        if g < n_tr + n_va:
            return ("validasi", g - n_tr)
        return ("uji", g - n_tr - n_va)

    lok = [lokasi(g) for g in idx3]
    for n, l in zip([x[5] for x in SAMPEL], lok):
        say(f"- {n}: " + (f"data {l[0]}, baris ke-{l[1]}" if l else "tidak ada"))
    say()

    # ------------------------------------------------------------ 5. imputasi
    say("## 5. Imputasi median (fit pada data latih)")
    tr_i, va_i, te_i = pp.impute_missing(tr, va, te)
    kn = tr.select_dtypes(include=[np.number]).columns.tolist()
    med = tr[kn].median()
    baris = []
    for k in kn:
        a, b, d = int(tr[k].isna().sum()), int(va[k].isna().sum()), int(te[k].isna().sum())
        if a + b + d:
            baris.append((k, a, b, d, med[k]))
    if baris:
        say("| kolom | NaN latih | NaN validasi | NaN uji | median latih |")
        say("|---|---|---|---|---|")
        for k, a, b, d, m in baris:
            say(f"| {k} | {a:,} | {b:,} | {d:,} | {m} |")
    else:
        say("- Tidak ada nilai hilang pada data latih, validasi, maupun uji pada tahap ini.")
    say()

    # --------------------------------------------------- 6-7. encode + scale
    say("## 6. Encoding kategori dan 7. Normalisasi Min-Max")
    trf, vaf, tef, art = pp.encode_and_scale(tr_i, va_i, te_i)
    say(f"- Fitur akhir: {len(art.feature_cols)} (kolom: {', '.join(art.feature_cols)})")
    say(f"- Kategori per fitur (termasuk token 'unknown'): {art.categorical_cardinalities}")
    for c_ in DATA.categorical_cols:
        enc = art.encoders[c_]
        say(f"- `{c_}`: {len(enc.classes_)} nilai; contoh urutan kode: " + ", ".join(
            f"{k}={i}" for i, k in enumerate(list(enc.classes_)[:12])) + (" ..." if len(enc.classes_) > 12 else ""))
        for nama, d in (("validasi", va_i), ("uji", te_i)):
            baru = (~d[c_].astype(str).isin(set(enc.classes_))).sum()
            say(f"  - baris {nama} dengan nilai belum pernah muncul di data latih (dipetakan ke 'unknown'): {int(baru):,}")
    say()
    kontinu = [k for k in art.feature_cols if k not in DATA.categorical_cols]
    mn, mx = art.scaler.data_min_, art.scaler.data_max_
    pilih = ["dur", "sbytes", "sttl", "dttl", "ct_srv_dst"]
    say("Minimum dan maksimum data latih untuk fitur contoh:")
    say("| fitur | min latih | maks latih |")
    say("|---|---|---|")
    for k in pilih:
        if k in kontinu:
            j = kontinu.index(k)
            say(f"| {k} | {mn[j]} | {mx[j]} |")
    say()

    # ------------------------------------------- cuplikan sebelum / sesudah
    say("## Cuplikan akhir: nilai asli, sesudah imputasi, dan sesudah encoding + Min-Max")
    peta = {"latih": (tr_i, trf), "validasi": (va_i, vaf), "uji": (te_i, tef)}
    kolom_akhir = ["proto", "state", "service", "dur", "sbytes", "sttl", "dttl", "ct_srv_dst"]
    for n, l in zip([x[5] for x in SAMPEL], lok):
        if not l:
            say(f"- {n}: tidak ada.")
            continue
        sblm, ssdh = peta[l[0]]
        a = sblm.loc[l[1], kolom_akhir]
        b = ssdh.loc[l[1], kolom_akhir]
        t = pd.DataFrame({"sebelum (setelah imputasi)": a, "sesudah (encoding + Min-Max)": b})
        t["sesudah (encoding + Min-Max)"] = t["sesudah (encoding + Min-Max)"].apply(
            lambda v: f"{v:.6f}" if isinstance(v, (float, np.floating)) else v)
        say(f"**{n}**, data {l[0]}, label akhir = {int(ssdh.loc[l[1], DATA.target_col])}")
        say()
        say("```")
        say(t.to_string())
        say("```")
        say()

    os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
    out = os.path.join(ROOT, "logs", "cuplikan_praproses.md")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(OUT))
    print(f"\nTersimpan: {out}")


if __name__ == "__main__":
    main()
