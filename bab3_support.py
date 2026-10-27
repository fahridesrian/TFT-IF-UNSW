"""
bab3_support.py
Satu skrip untuk menghasilkan angka dan cuplikan pendukung Bab 3 (Subbab 3.1 sampai 3.11).
HANYA MEMBACA: tidak mengubah berkas repositori dan tidak melatih ulang. Setiap
pemanggilan menulis SATU berkas baru berstempel waktu di logs/ (tidak ada yang tertimpa).

Taruh berkas ini di ROOT repositori (sejajar main.py), lalu jalankan:

    python bab3_support.py <perintah> [opsi]

Perintah (Subbab yang didukung):
  kerangka   3.1   inventaris run, hasil seleksi grid, angka tiap tahap (untuk diagram alur)
  lingkungan 3.2   versi pustaka, GPU, RAM, CPU, dan semua parameter di config.py
  data       3.3   statistik per berkas, kategori mentah, jumlah kelas, pembagian latih/validasi/uji
  praproses  3.4   angka dan cuplikan sebelum/sesudah tiap langkah praproses (4 rekaman contoh)
  manual     3.5   bahan hitung manual: Min-Max, lintasan tiap pohon IF, bobot kelas, (opsional) keluaran TFT
  skorif     3.6   statistik skor Isolation Forest dan AUC skor itu sendiri
  window     3.7   panjang segmen, jumlah jendela, ukuran memori, (opsional) verifikasi bentuk larik
  model      3.8   hyperparameter, jumlah parameter TFT-base dan TFT-IF (butuh PyTorch)
  pelatihan  3.9   bobot kelas dari target jendela latih, ringkasan epoch tiap run
  skenario   3.10  tabel grid (W,T), lima seed, acuan persistence dan selalu-Normal
  evaluasi   3.11  hitung ulang metrik dari matriks konfusi, AUC/PR-AUC, selisih TFT-IF dan TFT-base
  blok       3.11  uji dibagi blok berurutan, metrik tiap blok (butuh --run-dir)
  uji        3.11  uji selisih TFT-IF vs TFT-base: uji-t berpasangan antar seed dan bootstrap blok (butuh --run-dir)
  ambang     3.11  analisis ambang keputusan dari checkpoint (butuh PyTorch dan --run-dir)
  semua            kerangka, lingkungan, data, praproses, skorif, window, pelatihan, skenario, evaluasi

Opsi umum:
  --raw-dir D          folder CSV mentah (default dari config)
  --runs-dir D         folder run (default logs/runs)
  --W N --T N          konfigurasi jendela (default 25 dan 1)
  --run-dir D [D ...]  folder run untuk perintah manual/ambang (semua W,T sama)
  --fitur a,b,c        fitur untuk hitung manual (default dur,sbytes,sttl,ct_srv_dst)
  --bangun             pada 'window': benar-benar bangun jendela validasi untuk memeriksa bentuk larik

Contoh:
    python bab3_support.py lingkungan
    python bab3_support.py praproses
    python bab3_support.py skenario --W 25 --T 1
    python bab3_support.py manual --run-dir logs/runs/<run_seed42>
    python bab3_support.py ambang --W 25 --T 1 --run-dir logs/runs/<seed42> logs/runs/<seed43>
    python bab3_support.py semua

Kirim berkas logs/bab3_<perintah>_<waktu>.md yang dihasilkan ke Claude.
"""
import argparse
import glob
import importlib
import json
import math
import os
import platform
import sys
from dataclasses import asdict
from datetime import datetime

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

from config import DATA, IF, MODEL, PREPROC, TRAIN, WINDOW  # noqa: E402

OUT = []
NAMES = list(DATA.classes_used)
NORMAL = NAMES.index("Normal")
C = len(NAMES)

# (srcip, dstip, stime, sbytes, dbytes, nama, kelas): sama dengan Tabel cuplikan Bab 2
SAMPEL = [
    ("59.166.0.0", "149.171.126.6", 1421927414, 132, 164, "Rekaman 1 (Normal)", "Normal"),
    ("175.45.176.2", "149.171.126.16", 1421927416, 918, 25552, "Rekaman 2 (Exploits)", "Exploits"),
    ("175.45.176.0", "149.171.126.14", 1421927420, 844, 268, "Rekaman 3 (Generic)", "Generic"),
    ("175.45.176.2", "149.171.126.11", 1421927596, 1040, 0, "Rekaman 4 (Fuzzers)", "Fuzzers"),
]
KOLOM_TAMPIL = ["proto", "state", "service", "dur", "sbytes", "sttl", "dttl",
                "ct_srv_dst", "stime", "attack_cat"]


# --------------------------------------------------------------------------- util
def say(s=""):
    print(s)
    OUT.append(str(s))


def kode(df, judul=""):
    if judul:
        say(f"**{judul}**")
        say()
    say("```")
    say(df.to_string() if hasattr(df, "to_string") else str(df))
    say("```")
    say()


def md_table(header, rows):
    say("| " + " | ".join(header) + " |")
    say("|" + "---|" * len(header))
    for r in rows:
        say("| " + " | ".join(str(x) for x in r) + " |")
    say()


def f4(x):
    return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.4f}"


def ribu(n):
    return f"{int(n):,}".replace(",", ".")


def simpan(nama):
    os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
    out = os.path.join(ROOT, "logs", f"bab3_{nama}_{datetime.now().strftime('%Y%m%d-%H%M%S')}.md")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(OUT) + "\n")
    print(f"\nTertulis: {out}")


# --------------------------------------------------------------------------- konteks data
class Ctx:
    """Membangun data sekali (malas) dan menyimpannya untuk semua perintah."""

    def __init__(self, raw_dir):
        self.raw_dir = raw_dir
        self._c = {}

    def _get(self, k, fn):
        if k not in self._c:
            self._c[k] = fn()
        return self._c[k]

    @property
    def pp(self):
        import preprocessing as pp
        return pp

    @property
    def raw(self):
        from data_loading import load_unsw_nb15
        return self._get("raw", lambda: load_unsw_nb15(self.raw_dir))

    @property
    def filt(self):
        return self._get("filt", lambda: self.pp.filter_classes(self.raw))

    @property
    def clean(self):
        return self._get("clean", lambda: self.pp.clean_data(self.filt))

    @property
    def sorted(self):
        return self._get("sorted", lambda: self.pp.sort_chronologically(self.clean))

    @property
    def split_raw(self):
        return self._get("split_raw", lambda: self.pp.chronological_split(self.sorted))

    @property
    def imputed(self):
        return self._get("imputed", lambda: self.pp.impute_missing(*self.split_raw))

    @property
    def final(self):
        """(train_df, val_df, test_df, artifacts) sesudah encoding dan Min-Max."""
        return self._get("final", lambda: self.pp.encode_and_scale(*self.imputed))

    @property
    def with_if(self):
        def f():
            from isolation_forest_module import integrate_isolation_forest
            tr, va, te, art = self.final
            return integrate_isolation_forest(tr, va, te, art.feature_cols)
        return self._get("with_if", f)


# --------------------------------------------------------------------------- run
def baca_runs(runs_dir):
    runs = []
    for p in sorted(glob.glob(os.path.join(runs_dir, "*", "results_summary.json"))):
        try:
            with open(p) as f:
                s = json.load(f)
        except Exception:
            continue
        s["_dir"] = os.path.dirname(p)
        runs.append(s)
    return runs


def terbaru_per(runs, kunci):
    """Run terbaru (menurut started_at/run_id) untuk tiap kunci."""
    out = {}
    for r in sorted(runs, key=lambda r: r.get("config", {}).get("run_id", r["_dir"])):
        out[kunci(r)] = r
    return out


def metrik(r, model, split="test", k="f1_macro"):
    key = ("results_" if split == "test" else "results_val_") + model
    return r.get(key, {}).get(k)


def tfree(r, model, jenis="normal_vs_attack", k="pr_auc"):
    t = r.get("results_" + model, {}).get("threshold_free")
    if not t:
        return None
    if jenis == "macro":
        return t.get("macro", {}).get(k)
    return t.get(jenis, {}).get(k)


def mean_sd(v):
    v = [x for x in v if x is not None]
    if not v:
        return "-"
    return f"{np.mean(v):.4f} ± {np.std(v, ddof=1) if len(v) > 1 else 0.0:.4f}"


# --------------------------------------------------------------------------- 3.1 kerangka
def cmd_kerangka(a, ctx):
    say("# 3.1 Kerangka: inventaris run dan hasil seleksi grid")
    say()
    runs = baca_runs(a.runs_dir)
    say(f"Folder run: `{a.runs_dir}`; run ditemukan: {len(runs)}.")
    say()
    if not runs:
        say("Tidak ada run. Jalankan dari root repositori setelah training.")
        return
    rows = []
    for r in runs:
        c = r.get("config", {})
        rows.append((c.get("run_id", os.path.basename(r["_dir"])), c.get("W"), c.get("T"), c.get("seed"),
                     c.get("batch_size"), c.get("epochs_ran_base"), c.get("epochs_ran_if"),
                     f4(metrik(r, "TFT_base", "val")), f4(metrik(r, "TFT_IF", "val"))))
    md_table(["run", "W", "T", "seed", "batch", "epoch base", "epoch IF", "F1 macro val base", "F1 macro val IF"], rows)
    say("## Seleksi (W,T) per model: F1 macro validasi, seed 42, run terbaru per kombinasi")
    say()
    s42 = {k: v for k, v in terbaru_per([r for r in runs if r["config"].get("seed") == 42],
                                       lambda r: (r["config"]["W"], r["config"]["T"])).items()}
    for m in ("TFT_base", "TFT_IF"):
        if s42:
            best = max(s42, key=lambda k: metrik(s42[k], m, "val") or -1)
            say(f"- {m}: terbaik W={best[0]}, T={best[1]} (F1 macro validasi {f4(metrik(s42[best], m, 'val'))})")
    say()
    say("Gunakan angka ini untuk menyesuaikan teks langkah 5 pada diagram 3.1 (grid pada kedua model, (W,T) terbaik per model).")
    say()
    if ctx is not None:
        say("## Angka tiap tahap (untuk diagram)")
        say()
        md_table(["tahap", "rekaman"],
                 [("mentah (dibaca)", ribu(len(ctx.raw))), ("sesudah penyaringan empat kelas", ribu(len(ctx.filt))),
                  ("sesudah pembersihan", ribu(len(ctx.clean))),
                  ("latih / validasi / uji", " / ".join(ribu(len(d)) for d in ctx.split_raw))])
        art = ctx.final[3]
        say(f"Fitur masukan TFT-base: {len(art.feature_cols)}; TFT-IF: {len(art.feature_cols) + 1}.")
        say()


# --------------------------------------------------------------------------- 3.2 lingkungan
def _ver(nama):
    try:
        from importlib import metadata
        return metadata.version(nama)
    except Exception:
        return "tidak terpasang"


def _ram_gb():
    try:
        import psutil
        return psutil.virtual_memory().total / 1024 ** 3
    except Exception:
        pass
    try:
        with open("/proc/meminfo") as f:
            for ln in f:
                if ln.startswith("MemTotal"):
                    return int(ln.split()[1]) / 1024 ** 2
    except Exception:
        pass
    try:
        import ctypes

        class MS(ctypes.Structure):
            _fields_ = [("l", ctypes.c_ulong), ("m", ctypes.c_ulong), ("tp", ctypes.c_ulonglong),
                        ("ap", ctypes.c_ulonglong), ("tpf", ctypes.c_ulonglong), ("apf", ctypes.c_ulonglong),
                        ("tv", ctypes.c_ulonglong), ("av", ctypes.c_ulonglong), ("ae", ctypes.c_ulonglong)]
        s = MS()
        s.l = ctypes.sizeof(MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))
        return s.tp / 1024 ** 3
    except Exception:
        return None


def cmd_lingkungan(a, ctx):
    say("# 3.2 Lingkungan dan perangkat penelitian")
    say()
    say("## Perangkat lunak")
    say()
    rows = [("Python", platform.python_version()), ("Sistem operasi", platform.platform())]
    for n in ("torch", "scikit-learn", "numpy", "pandas", "optuna", "scipy"):
        rows.append((n, _ver(n)))
    md_table(["komponen", "versi"], rows)
    say("## Perangkat keras")
    say()
    rows = [("CPU", platform.processor() or platform.machine()), ("Jumlah inti logis", os.cpu_count())]
    ram = _ram_gb()
    rows.append(("RAM", f"{ram:.1f} GB" if ram else "tidak terbaca"))
    try:
        import torch
        rows.append(("CUDA tersedia", torch.cuda.is_available()))
        if torch.cuda.is_available():
            pr = torch.cuda.get_device_properties(0)
            rows += [("GPU", pr.name), ("Memori GPU", f"{pr.total_memory / 1024 ** 3:.1f} GB"),
                     ("Versi CUDA (PyTorch)", torch.version.cuda)]
    except Exception:
        rows.append(("PyTorch", "tidak terpasang"))
    md_table(["komponen", "nilai"], rows)
    say("## Parameter di config.py (nilai default; nilai tiap run ada di results_summary.json)")
    say()
    for nama, obj in (("DATA", DATA), ("PREPROC", PREPROC), ("IF", IF), ("WINDOW", WINDOW), ("MODEL", MODEL), ("TRAIN", TRAIN)):
        d = asdict(obj)
        if nama == "DATA":
            d = {k: v for k, v in d.items() if k not in ("columns",)}
        md_table([nama + ": parameter", "nilai"], [(k, v) for k, v in d.items()])


# --------------------------------------------------------------------------- 3.3 data
def _kategori_mentah(raw):
    k = raw[DATA.target_col].astype(str).str.strip().replace({"nan": "", "None": ""})
    return k.replace({"": "Normal"})


def cmd_data(a, ctx):
    say("# 3.3 Data penelitian")
    say()
    raw = ctx.raw
    say(f"Bentuk data mentah: {ribu(raw.shape[0])} baris, {raw.shape[1]} kolom (termasuk `source_file`).")
    say()
    say("## Per berkas (rekaman, rentang waktu UTC, celah terbesar antar rekaman berurutan)")
    say()
    rows = []
    for sf, g in raw.groupby(DATA.source_file_col, sort=True):
        st = pd.to_numeric(g["stime"], errors="coerce").dropna().sort_values()
        gap = (st.diff().max() / 3600.0) if len(st) > 1 else 0.0
        rows.append((sf, ribu(len(g)), str(pd.to_datetime(st.min(), unit="s")), str(pd.to_datetime(st.max(), unit="s")), f"{gap:.1f} jam"))
    md_table(["berkas", "rekaman", "mulai", "selesai", "celah terbesar"], rows)
    say("## Seluruh kategori serangan (data mentah)")
    say()
    km = _kategori_mentah(raw)
    vc = km.value_counts()
    md_table(["kategori", "rekaman", "% dari mentah"], [(k, ribu(v), f"{v / len(raw) * 100:.2f}") for k, v in vc.items()])
    say(f"Rasio kelas terbesar : terkecil (seluruh kategori): {vc.max() / vc.min():.1f}")
    say()
    say("## Empat kelas: sebelum dan sesudah pembersihan (Tabel jumlah kelas)")
    say()
    f, c = ctx.filt, ctx.clean
    rows = []
    for k in NAMES:
        a0 = int((f[DATA.target_col] == k).sum())
        a1 = int((c[DATA.target_col] == k).sum())
        rows.append((k, ribu(a0), ribu(a1), f"{(a0 - a1) / a0 * 100:.1f}" if a0 else "-", f"{a1 / len(c) * 100:.2f}"))
    rows.append(("Total", ribu(len(f)), ribu(len(c)), f"{(len(f) - len(c)) / len(f) * 100:.1f}", "100"))
    md_table(["kelas", "sebelum", "sesudah", "% dibuang", "% sesudah"], rows)
    n_k = [int((c[DATA.target_col] == k).sum()) for k in NAMES]
    say(f"Rasio ketidakseimbangan (terbesar : terkecil) setelah pembersihan: {max(n_k) / min(n_k):.1f}")
    say()
    say("## Pembagian 70:15:15 (Tabel distribusi split dan rentang split)")
    say()
    tr, va, te = ctx.split_raw
    rows = []
    for k in NAMES:
        r = [k]
        for d in (tr, va, te):
            n = int((d[DATA.target_col] == k).sum())
            r += [ribu(n), f"{n / len(d) * 100:.2f}"]
        rows.append(r)
    rows.append(["Total"] + sum([[ribu(len(d)), "100"] for d in (tr, va, te)], []))
    md_table(["kelas", "latih", "%", "validasi", "%", "uji", "%"], rows)
    for nm, d in (("latih", tr), ("validasi", va), ("uji", te)):
        ser = 100 * (1 - (d[DATA.target_col] == "Normal").mean())
        say(f"- Persentase serangan pada data {nm}: {ser:.2f}%")
    say()
    rows = []
    for nm, d in (("Latih", tr), ("Validasi", va), ("Uji", te)):
        berkas = ", ".join(f"{str(k).split('_')[-1].split('.')[0]}: {ribu(v)}" for k, v in d[DATA.source_file_col].value_counts().sort_index().items())
        rows.append((nm, ribu(len(d)), str(pd.to_datetime(d['stime'].min(), unit='s')), str(pd.to_datetime(d['stime'].max(), unit='s')), berkas))
    md_table(["bagian", "rekaman", "mulai", "selesai", "berkas asal (rekaman)"], rows)
    say("## Nilai hilang pada data mentah (setelah konversi numerik, empat kelas)")
    say()
    kon = ctx.pp.coerce_numeric_columns(f)
    num = kon.select_dtypes(include=[np.number]).columns
    na = kon[num].isna().sum()
    na = na[na > 0].sort_values(ascending=False)
    md_table(["kolom", "NaN", "% dari rekaman"], [(k, ribu(v), f"{v / len(kon) * 100:.1f}") for k, v in na.items()] or [("(tidak ada)", 0, 0)])


# --------------------------------------------------------------------------- 3.4 praproses
def _cari(df, sampel=SAMPEL):
    hasil = []
    for src, dst, st, sb, db, nama, kelas in sampel:
        kat = df[DATA.target_col].fillna("").astype(str).str.strip().str.lower()
        kat = kat.where(~kat.isin(["", "nan", "none"]), "normal")
        m = ((kat == kelas.lower())
             & (df["srcip"].astype(str).str.strip() == src)
             & (df["dstip"].astype(str).str.strip() == dst)
             & (pd.to_numeric(df["stime"], errors="coerce") == st)
             & (pd.to_numeric(df["sbytes"], errors="coerce") == sb)
             & (pd.to_numeric(df["dbytes"], errors="coerce") == db))
        idx = df.index[m]
        hasil.append(int(idx[0]) if len(idx) else None)
    return hasil


def _lokasi(g, n_tr, n_va):
    if g is None:
        return None
    if g < n_tr:
        return ("latih", g)
    if g < n_tr + n_va:
        return ("validasi", g - n_tr)
    return ("uji", g - n_tr - n_va)


def cmd_praproses(a, ctx):
    pp = ctx.pp
    say("# 3.4 Angka dan cuplikan praproses")
    say()
    raw = ctx.raw
    say("## 0. Data mentah")
    say(f"- Bentuk: {ribu(raw.shape[0])} baris, {raw.shape[1]} kolom (49 kolom data + `source_file`).")
    say("- Rekaman per berkas: " + ", ".join(f"{k}={ribu(v)}" for k, v in raw["source_file"].value_counts().sort_index().items()))
    say()
    idx0 = _cari(raw)
    b0 = raw.loc[[i for i in idx0 if i is not None], KOLOM_TAMPIL].copy()
    b0[DATA.target_col] = b0[DATA.target_col].map(repr)
    kode(b0, "Cuplikan mentah (nilai `attack_cat` dengan repr)")
    for n, i in zip([s[5] for s in SAMPEL], idx0):
        if i is None:
            say(f"- {n}: TIDAK ditemukan di data mentah.")
    f = ctx.filt
    say("## 1. Penyaringan empat kelas")
    say(f"- Sebelum {ribu(len(raw))}; sesudah {ribu(len(f))}; dibuang {ribu(len(raw) - len(f))}.")
    say()
    kon = pp.coerce_numeric_columns(f)
    kn = kon.select_dtypes(include=[np.number]).columns.tolist()
    nanc = kon[kn].isna().sum()
    nanc = nanc[nanc > 0].sort_values(ascending=False)
    say("## 2. Pembersihan")
    say("- NaN sesudah konversi numerik: " + (", ".join(f"{k}={ribu(v)}" for k, v in nanc.items()) or "tidak ada"))
    say(f"- Nilai tak hingga: {int(np.isinf(kon[kn].to_numpy(dtype=float)).sum())} sel.")
    c = ctx.clean
    say(f"- Duplikat dibuang: {ribu(len(f) - len(c))} ({ribu(len(f))} menjadi {ribu(len(c))}).")
    say()
    s = ctx.sorted
    say("## 3. Pengurutan kronologis")
    say(f"- `stime` unik: {ribu(s['stime'].nunique())} dari {ribu(len(s))}; baris berbagi `stime`: {ribu(s['stime'].duplicated(keep=False).sum())}.")
    say()
    idx3 = _cari(s)
    tr, va, te = ctx.split_raw
    n_tr, n_va = len(tr), len(va)
    lok = [_lokasi(g, n_tr, n_va) for g in idx3]
    for n, l in zip([x[5] for x in SAMPEL], lok):
        say(f"- {n}: " + (f"data {l[0]}, baris ke-{l[1]}" if l else "tidak ada"))
    say()
    say("## 4. Imputasi median (fit pada data latih)")
    tr_i, va_i, te_i = ctx.imputed
    med = tr[kn].median() if all(k in tr.columns for k in kn) else None
    rows = []
    for k in kn:
        if k in tr.columns:
            n1, n2, n3 = int(tr[k].isna().sum()), int(va[k].isna().sum()), int(te[k].isna().sum())
            if n1 + n2 + n3:
                rows.append((k, ribu(n1), ribu(n2), ribu(n3), med[k]))
    md_table(["kolom", "NaN latih", "NaN validasi", "NaN uji", "median latih"], rows or [("(tidak ada)", 0, 0, 0, "-")])
    say("## 5. Encoding dan Min-Max")
    trf, vaf, tef, art = ctx.final
    say(f"- Fitur akhir: {len(art.feature_cols)}; kardinalitas kategorik: {art.categorical_cardinalities}")
    for c_ in DATA.categorical_cols:
        enc = art.encoders[c_]
        for nm, d in (("validasi", va_i), ("uji", te_i)):
            baru = int((~d[c_].astype(str).isin(set(enc.classes_))).sum())
            say(f"  - `{c_}`: baris {nm} bernilai baru (menjadi 'unknown'): {ribu(baru)}")
    kontinu = [k for k in art.feature_cols if k not in DATA.categorical_cols]
    rows = []
    for k in ["dur", "sbytes", "sttl", "dttl", "ct_srv_dst"]:
        if k in kontinu:
            j = kontinu.index(k)
            rows.append((k, art.scaler.data_min_[j], art.scaler.data_max_[j]))
    md_table(["fitur", "min latih", "maks latih"], rows)
    say("## Cuplikan akhir: sesudah imputasi dan sesudah encoding + Min-Max")
    peta = {"latih": (tr_i, trf), "validasi": (va_i, vaf), "uji": (te_i, tef)}
    kolom_akhir = ["proto", "state", "service", "dur", "sbytes", "sttl", "dttl", "ct_srv_dst"]
    for n, l in zip([x[5] for x in SAMPEL], lok):
        if not l:
            say(f"- {n}: tidak ada.")
            continue
        sblm, ssdh = peta[l[0]]
        t = pd.DataFrame({"sebelum": sblm.loc[l[1], kolom_akhir], "sesudah": ssdh.loc[l[1], kolom_akhir]})
        t["sesudah"] = t["sesudah"].apply(lambda v: f"{v:.6f}" if isinstance(v, (float, np.floating)) else v)
        kode(t, f"{n}, data {l[0]}, label akhir = {int(ssdh.loc[l[1], DATA.target_col])}")


# --------------------------------------------------------------------------- 3.5 manual
def _c_n(n):
    if n <= 1:
        return 0.0
    if n == 2:
        return 1.0
    return 2.0 * (math.log(n - 1) + 0.5772156649015329) - 2.0 * (n - 1) / n


def jalur_pohon(iso, X):
    """Untuk tiap pohon: kedalaman daun l, banyak sampel daun m, h = l + c(m), per baris X."""
    n = X.shape[0]
    L = np.zeros((len(iso.estimators_), n))
    M = np.zeros_like(L)
    for t, (est, feats) in enumerate(zip(iso.estimators_, iso.estimators_features_)):
        Xt = X[:, feats]
        L[t] = np.asarray(est.decision_path(Xt).sum(axis=1)).ravel() - 1
        leaf = est.apply(Xt)
        M[t] = est.tree_.n_node_samples[leaf]
    H = L + np.vectorize(_c_n)(M)
    return L, M, H


def cmd_manual(a, ctx):
    say("# 3.5 Bahan hitung manual")
    say()
    trf, vaf, tef, art = ctx.final
    tr_i, va_i, te_i = ctx.imputed
    s = ctx.sorted
    tr, va, te = ctx.split_raw
    n_tr, n_va = len(tr), len(va)
    lok = [_lokasi(g, n_tr, n_va) for g in _cari(s)]
    peta = {"latih": (tr_i, trf), "validasi": (va_i, vaf), "uji": (te_i, tef)}
    fitur = [x.strip() for x in a.fitur.split(",") if x.strip()]
    kontinu = [k for k in art.feature_cols if k not in DATA.categorical_cols]
    tf, vf, ef, _cols, iso = ctx.with_if
    pf = {"latih": tf, "validasi": vf, "uji": ef}
    psi = iso.max_samples_
    c_psi = _c_n(psi)
    say(f"## A. Min-Max: f(x) = (x - m) / (M - m), m dan M dari data latih (fitur: {', '.join(fitur)})")
    say()
    rows = []
    for n, l in zip([x[5] for x in SAMPEL], lok):
        if not l:
            continue
        sblm, ssdh = peta[l[0]]
        for k in fitur:
            if k not in kontinu:
                say(f"- {k}: bukan fitur kontinu, dilewati.")
                continue
            j = kontinu.index(k)
            m, M = art.scaler.data_min_[j], art.scaler.data_max_[j]
            x = float(sblm.loc[l[1], k])
            hand = (x - m) / (M - m)
            rows.append((n, k, x, m, M, f"{hand:.10f}", f"{float(ssdh.loc[l[1], k]):.10f}",
                         f"{abs(hand - float(ssdh.loc[l[1], k])):.1e}"))
    md_table(["rekaman", "fitur", "x", "m", "M", "(x-m)/(M-m)", "nilai pada data", "selisih"], rows)
    say(f"## B. Skor Isolation Forest: psi = {psi}, c(psi) = {c_psi:.6f}, pohon = {len(iso.estimators_)}")
    say()
    for n, l in zip([x[5] for x in SAMPEL], lok):
        if not l:
            continue
        d = pf[l[0]]
        X = d.loc[[l[1]], art.feature_cols].to_numpy(dtype=float)
        L, M, H = jalur_pohon(iso, X)
        eh = float(H.mean())
        s_hand = 2.0 ** (-eh / c_psi)
        s_data = float(d.loc[l[1], "if_score"])
        say(f"**{n}** (data {l[0]}): E[h] = {eh:.6f}; s = 2^(-E[h]/c(psi)) = {s_hand:.8f}; if_score pada data = {s_data:.8f}; selisih = {abs(s_hand - s_data):.1e}")
        say()
        rows = [(t + 1, int(L[t, 0]), int(M[t, 0]), f"{_c_n(int(M[t, 0])):.4f}", f"{H[t, 0]:.4f}") for t in range(min(10, L.shape[0]))]
        md_table(["pohon", "kedalaman l", "sampel daun m", "c(m)", "h = l + c(m)"], rows)
    say("## C. Bobot kelas balanced dari target jendela latih")
    say()
    cnt = _target_counts(tf, a.W, a.T)
    N = cnt.sum()
    w = N / (C * cnt)
    md_table(["kelas", "n_k", "w_k = N / (K n_k)"], [(NAMES[k], ribu(cnt[k]), f"{w[k]:.6f}") for k in range(C)])
    say(f"N = {ribu(N)}, K = {C}, W = {a.W}, T = {a.T}.")
    say()
    if a.run_dir:
        _manual_model(a, ctx, lok, peta, w)


def _tft_mini(model, cols, x, nama_rekaman, fitur):
    """Bagian E: potongan TFT yang dapat dihitung tangan, dengan angka dari model terlatih.
    E1 lapisan linear 1 -> d untuk satu fitur (3 dimensi pertama dari d),
    E2 softmax bobot VSN atas semua fitur pada titik waktu terakhir jendela,
    E3 satu langkah attention: skor q.k / sqrt(d_attn) untuk posisi target dan softmax-nya (head 1).
    x: tensor (1, W, F) jendela yang berakhir tepat sebelum rekaman contoh."""
    import torch
    cap = {}
    h1 = model.vsn.flattened_grn.register_forward_hook(lambda m, i, o: cap.__setitem__("vsn_logit", o.detach()))
    h2 = model.attention.register_forward_hook(lambda m, i, o: cap.__setitem__("attn_in", [t.detach() for t in i[:3]]))
    try:
        with torch.no_grad():
            model(x)
    finally:
        h1.remove()
        h2.remove()
    F_ = len(cols)
    W_ = x.shape[1]
    say(f"## E. Potongan TFT-base untuk hitung tangan ({nama_rekaman}; jendela W = {W_}, F = {F_}, d = {model.hidden_size})")
    say()
    say("Angka berikut dari model terlatih (mode evaluasi, tanpa dropout). Hanya sebagian kecil model yang dihitung; seluruh LSTM dan GRN tidak dihitung tangan.")
    say()
    # E1
    say("### E1. Transformasi linear tiap fitur kontinu: h_j = w_j * x + b_j (3 dimensi pertama dari d)")
    say()
    rows = []
    for f in fitur:
        if f not in cols:
            continue
        i = cols.index(f)
        enc = model.vsn.feature_encoders[i]
        if not hasattr(enc, "weight") or enc.weight.shape[1] != 1:
            continue
        xv = float(x[0, -1, i])
        for j in range(3):
            wj = float(enc.weight[j, 0].detach())
            bj = float(enc.bias[j].detach())
            rows.append([f, f"{xv:.10f}", j + 1, f"{wj:.6f}", f"{bj:.6f}", f"{wj * xv + bj:.6f}"])
    md_table(["fitur", "x (posisi terakhir jendela)", "j", "w_j", "b_j", "h_j = w_j x + b_j (kunci)"], rows)
    say()
    # E2
    lg = cap["vsn_logit"][0, -1]
    wt = torch.softmax(lg, dim=-1)
    den = float(torch.exp(lg).sum())
    say("### E2. Bobot VSN: u_i = exp(v_i) / sum_k exp(v_k) pada posisi terakhir jendela (v = keluaran GRN penggabung, F nilai)")
    say()
    say(f"Penyebut sum_k exp(v_k) = {den:.6f}.")
    say()
    md_table(["no.", "fitur", "v_i", "exp(v_i)", "u_i (kunci)"],
             [[i + 1, c, f"{float(lg[i]):.6f}", f"{math.exp(float(lg[i])):.6f}", f"{float(wt[i]):.6f}"] for i, c in enumerate(cols)])
    say()
    say(f"Jumlah u_i = {float(wt.sum()):.6f}. Hitung u_i untuk fitur contoh ({', '.join(fitur)}) dari v_i dan penyebut di atas.")
    say()
    # E3
    q_in, k_in, _v_in = cap["attn_in"]
    att = model.attention
    d_a = att.d_attn
    with torch.no_grad():
        q = att.w_q[0](q_in)[0, -1]
        kk = att.w_k[0](k_in)[0]
        sc = (kk @ q) / math.sqrt(d_a)
        pr = torch.softmax(sc, dim=-1)
    say(f"### E3. Attention head 1, posisi target (urutan {sc.shape[0]} dari {sc.shape[0]}): skor s_i = q . k_i / sqrt(d_attn), d_attn = {d_a}")
    say()
    say("Vektor kueri q (posisi target) dan kunci k pada posisi target (dua vektor, masing-masing %d nilai):" % d_a)
    say()
    md_table(["dim", "q", "k (posisi target)"],
             [[m + 1, f"{float(q[m]):.6f}", f"{float(kk[-1, m]):.6f}"] for m in range(d_a)])
    say()
    say(f"Tugas E3a: hitung s_{sc.shape[0]} = (q . k_{sc.shape[0]}) / sqrt({d_a}). Kunci: {float(sc[-1]):.6f}.")
    say()
    say("Skor posisi lain (semua posisi sebelum target boleh dilihat; mask kausal hanya menutup posisi sesudah target):")
    say()
    sden = float(torch.exp(sc).sum())
    md_table(["posisi i", "s_i", "exp(s_i)", "bobot a_i (kunci)"],
             [[i + 1, f"{float(sc[i]):.6f}", f"{math.exp(float(sc[i])):.6f}", f"{float(pr[i]):.6f}"] for i in range(sc.shape[0])])
    say()
    say(f"Penyebut sum exp(s_i) = {sden:.6f}; jumlah a_i = {float(pr.sum()):.6f}. Tugas E3b: hitung a_i untuk tiga posisi terakhir.")
    say()
    say("Catatan: matriks attention yang dilaporkan model adalah rata-rata empat head; potongan ini memakai head 1 saja.")
    say()


def _manual_model(a, ctx, lok, peta, w):
    try:
        import torch
        from config import MODEL as M_
        from tft_model import TemporalFusionTransformer
    except Exception as e:
        say(f"(Bagian D dilewati: PyTorch tidak dapat dimuat: {e})")
        return
    if a.T != 1:
        say("(Bagian D hanya untuk T = 1.)")
        return
    d = a.run_dir[0]
    with open(os.path.join(d, "results_summary.json")) as f:
        cfg = json.load(f)["config"]
    tr_f, va_f, te_f, art = ctx.final
    tf, vf, ef, _cols, _iso = ctx.with_if
    pf = {"latih": tf, "validasi": vf, "uji": ef}
    say(f"## D. Keluaran model pada jendela yang berakhir tepat sebelum rekaman contoh (run {os.path.basename(d)})")
    say()
    cat = art.categorical_feature_indices or {}
    mini_args = None  # bagian E dicetak sesudah seluruh daftar D
    for nm, ckpt, frame, cols in (("TFT_base", "tft_base.pt", None, art.feature_cols),
                                  ("TFT_IF", "tft_if.pt", None, art.feature_cols + ["if_score"])):
        model = TemporalFusionTransformer(
            num_features=len(cols), hidden_size=cfg["hidden_size"], num_classes=M_.num_classes, T=cfg["T"],
            lstm_layers=M_.lstm_layers, attn_heads=cfg["attn_heads"], dropout=cfg["dropout"],
            categorical_features=cat)
        model.load_state_dict(torch.load(os.path.join(d, ckpt), map_location="cpu"))
        model.eval()
        for n, l in zip([x[5] for x in SAMPEL], lok):
            if not l or l[1] < a.W:
                continue
            seg = pf[l[0]]
            win = seg.iloc[l[1] - a.W:l[1]]
            if win[DATA.source_file_col].nunique() != 1:
                say(f"- {n}: jendela melintasi dua berkas, dilewati.")
                continue
            x = torch.tensor(win[cols].to_numpy(dtype=np.float32))[None]
            with torch.no_grad():
                p = torch.softmax(model(x), dim=-1)[0, 0].numpy()
            y = int(seg.iloc[l[1]][DATA.target_col])
            ce = -w[y] * math.log(max(p[y], 1e-12))
            say(f"- {nm}, {n}: P = " + ", ".join(f"{NAMES[k]} {p[k]:.4f}" for k in range(C))
                + f"; label benar = {NAMES[y]}; loss berbobot = w_y * (-ln p_y) = {w[y]:.4f} * {-math.log(max(p[y], 1e-12)):.4f} = {ce:.4f}")
            if nm == "TFT_base" and mini_args is None:
                mini_args = (model, cols, x, n, [f for f in a.fitur.split(",") if f.strip()])
    say()
    if mini_args is not None:
        _tft_mini(*mini_args)


# --------------------------------------------------------------------------- 3.6 skorif
def cmd_skorif(a, ctx):
    from sklearn.metrics import average_precision_score, roc_auc_score
    say("# 3.6 Statistik skor anomali Isolation Forest")
    say()
    tr, va, te, cols, iso = ctx.with_if
    say(f"Fitur masukan IF: {len(cols) - 1}. Pohon: {iso.n_estimators}. psi efektif: {iso.max_samples_}. random_state: {IF.random_state}.")
    n_norm = int((tr[DATA.target_col] == NORMAL).sum())
    say(f"Normal latih: {ribu(n_norm)}; batas atas rekaman yang pernah masuk satu pohon: {ribu(iso.n_estimators * iso.max_samples_)} ({100 * iso.n_estimators * iso.max_samples_ / n_norm:.2f}%).")
    say()
    rows = []
    for nm, d in (("Latih", tr), ("Validasi", va), ("Uji", te)):
        for k, kn in enumerate(NAMES):
            v = d.loc[d[DATA.target_col] == k, "if_score"].to_numpy()
            if len(v):
                rows.append((nm, kn, ribu(len(v)), f"{v.mean():.4f}", f"{np.median(v):.4f}", f"{np.percentile(v, 5):.4f}", f"{np.percentile(v, 95):.4f}"))
    md_table(["bagian", "kelas", "jumlah", "rata-rata", "median", "persentil 5", "persentil 95"], rows)
    rows = []
    for nm, d in (("Latih", tr), ("Validasi", va), ("Uji", te)):
        y = (d[DATA.target_col] != NORMAL).astype(int).to_numpy()
        sc = d["if_score"].to_numpy()
        rows.append((nm, f4(roc_auc_score(y, sc)), f4(average_precision_score(y, sc)), f4(y.mean())))
    md_table(["bagian", "ROC-AUC", "PR-AUC", "proporsi serangan"], rows)


# --------------------------------------------------------------------------- 3.7 window
def _segmen(d):
    return {str(k): len(g) for k, g in d.groupby(DATA.source_file_col, sort=True)}


def _target_counts(df, W, T):
    """Hitung kelas pada semua target jendela (diratakan) tanpa membangun larik masukan."""
    cnt = np.zeros(C, dtype=np.int64)
    for _, seg in df.groupby(DATA.source_file_col, sort=False):
        n = len(seg)
        if n < W + T:
            continue
        y = seg[DATA.target_col].to_numpy().astype(np.int64)
        starts = np.arange(0, n - W - T + 1)
        Yv = np.lib.stride_tricks.sliding_window_view(y, T)[starts + W]
        cnt += np.bincount(Yv.reshape(-1), minlength=C)
    return cnt


def cmd_window(a, ctx):
    W, T = a.W, a.T
    say(f"# 3.7 Sliding window (W={W}, T={T})")
    say()
    trf, vaf, tef, art = ctx.final
    F = len(art.feature_cols)
    rows, tot = [], {}
    for nm, d in (("Latih", trf), ("Validasi", vaf), ("Uji", tef)):
        seg = _segmen(d)
        nj = sum(max(0, n - W - T + 1) for n in seg.values())
        tot[nm] = nj
        rows.append((nm, ribu(len(d)), len(seg), "; ".join(f"{k.split('_')[-1].split('.')[0]}: {ribu(v)}" for k, v in seg.items()), ribu(nj), ribu(len(d) - nj)))
    md_table(["bagian", "rekaman", "segmen", "panjang segmen", "jendela", "rekaman - jendela"], rows)
    say(f"Selisih rekaman - jendela seharusnya (W + T - 1) x segmen = {W + T - 1} x banyaknya segmen.")
    say()
    rows = []
    for nm, nj in tot.items():
        rows.append((nm, f"{nj * W * F * 4 / 1e9:.2f}", f"{nj * W * (F + 1) * 4 / 1e9:.2f}"))
    tt = sum(tot.values()) * W * (2 * F + 1) * 4 / 1e9
    md_table(["bagian", f"GB TFT-base (F={F})", f"GB TFT-IF (F={F + 1})"], rows)
    say(f"Total keenam larik float32: {tt:.1f} GB (main.py membangun semuanya sekaligus).")
    say()
    say("## Jendela yang melintasi celah waktu terbesar (> 1 jam) di dalam satu berkas, data latih")
    say()
    g_ = ctx.split_raw[0]
    for sf, g in g_.groupby(DATA.source_file_col, sort=True):
        st = pd.to_numeric(g["stime"], errors="coerce").to_numpy()
        if len(st) < 2:
            continue
        dif = np.diff(st)
        j = int(dif.argmax())
        if dif[j] / 3600 > 1:
            say(f"- {sf}: celah {dif[j] / 3600:.1f} jam; jendela yang melintasinya paling banyak {W + T - 1} dari {ribu(len(g) - W - T + 1)} jendela berkas ini.")
    say()
    say("## Banyak target per kelas pada jendela latih")
    say()
    cnt = _target_counts(trf, W, T)
    md_table(["kelas", "target"], [(NAMES[k], ribu(cnt[k])) for k in range(C)])
    if a.bangun:
        from sliding_window import build_windows
        WINDOW.W, WINDOW.T = W, T
        X, Y = build_windows(vaf, art.feature_cols)
        say(f"Verifikasi (bangun nyata, validasi): X {X.shape}, Y {Y.shape}; jumlah jendela hitungan = {tot['Validasi']} -> {'cocok' if X.shape[0] == tot['Validasi'] else 'TIDAK COCOK'}.")
        say()


# --------------------------------------------------------------------------- 3.8 model
def cmd_model(a, ctx):
    import torch
    from tft_model import TemporalFusionTransformer
    say("# 3.8 Rancangan model")
    say()
    art = ctx.final[3]
    cat = art.categorical_feature_indices or {}
    md_table(["hyperparameter", "nilai"], [("hidden_size", MODEL.hidden_size), ("lstm_layers", MODEL.lstm_layers),
                                         ("attn_heads", MODEL.attn_heads), ("dropout", MODEL.dropout),
                                         ("num_classes", MODEL.num_classes), ("T (keluaran)", a.T),
                                         ("kardinalitas kategorik", art.categorical_cardinalities)])
    rows = []
    for nm, nf in (("TFT-base", len(art.feature_cols)), ("TFT-IF", len(art.feature_cols) + 1)):
        m = TemporalFusionTransformer(num_features=nf, hidden_size=MODEL.hidden_size, num_classes=MODEL.num_classes,
                                      T=a.T, lstm_layers=MODEL.lstm_layers, attn_heads=MODEL.attn_heads,
                                      dropout=MODEL.dropout, categorical_features=cat)
        tot = sum(p.numel() for p in m.parameters())
        x = torch.zeros(2, a.W, nf)
        with torch.no_grad():
            m.eval()
            out = m(x)
        rows.append((nm, nf, ribu(tot), tuple(x.shape), tuple(out.shape)))
        if nm == "TFT-IF":
            sub = [(n_, ribu(sum(p.numel() for p in mod.parameters()))) for n_, mod in m.named_children()]
    md_table(["model", "fitur", "parameter", "bentuk masukan", "bentuk keluaran (logit)"], rows)
    say("Parameter per modul (TFT-IF):")
    say()
    md_table(["modul", "parameter"], sub)


# --------------------------------------------------------------------------- 3.9 pelatihan
def cmd_pelatihan(a, ctx):
    say(f"# 3.9 Pelatihan dan penanganan ketidakseimbangan (W={a.W}, T={a.T})")
    say()
    trf = ctx.final[0]
    cnt = _target_counts(trf, a.W, a.T)
    N = cnt.sum()
    w = N / (C * cnt)
    md_table(["kelas", "n_k (target jendela latih)", "w_k = N / (K n_k)"], [(NAMES[k], ribu(cnt[k]), f"{w[k]:.4f}") for k in range(C)])
    say(f"N = {ribu(N)}. Bobot terbesar : terkecil = {w.max() / w.min():.1f}. Rumus sama dengan `compute_class_weight('balanced')` pada train.py.")
    say()
    md_table(["pengaturan", "nilai"], [("optimizer", "Adam"), ("learning_rate", TRAIN.learning_rate), ("weight_decay", TRAIN.weight_decay),
                                      ("epochs maksimum", TRAIN.epochs), ("patience early stopping", TRAIN.patience),
                                      ("scheduler patience / factor", f"{TRAIN.lr_scheduler_patience} / {TRAIN.lr_scheduler_factor}"),
                                      ("batch_size default (run grid dan lima seed memakai 2048)", TRAIN.batch_size)])
    runs = baca_runs(a.runs_dir)
    sel = [r for r in runs if r["config"].get("W") == a.W and r["config"].get("T") == a.T]
    rows = [(r["config"].get("run_id"), r["config"].get("seed"), r["config"].get("batch_size"),
             r["config"].get("epochs_ran_base"), r["config"].get("epochs_ran_if")) for r in sel]
    say("## Epoch yang dijalankan sebelum early stopping")
    say()
    md_table(["run", "seed", "batch", "epoch base", "epoch IF"], rows or [("(tidak ada run)", "", "", "", "")])
    if rows:
        say(f"Rata-rata epoch: base {np.mean([x[3] for x in rows if x[3]]):.1f}, IF {np.mean([x[4] for x in rows if x[4]]):.1f} (maksimum {TRAIN.epochs}).")
        say()


# --------------------------------------------------------------------------- 3.10 skenario
def cmd_skenario(a, ctx):
    say("# 3.10 Skenario eksperimen: grid dan lima seed")
    say()
    runs = baca_runs(a.runs_dir)
    if not runs:
        say("Tidak ada run.")
        return
    per = {}
    for r in runs:
        c = r["config"]
        per.setdefault((c["W"], c["T"], c.get("seed")), []).append(r)
    last = {k: sorted(v, key=lambda r: r["config"].get("run_id", ""))[-1] for k, v in per.items()}
    say("## Grid (seed 42), F1 macro; seleksi memakai validasi")
    say()
    rows = []
    for (W, T, sd), r in sorted(last.items()):
        if sd != 42:
            continue
        rows.append((f"W{W} T{T}", r["config"].get("batch_size"), f4(metrik(r, "TFT_base", "val")), f4(metrik(r, "TFT_IF", "val")),
                     f4(metrik(r, "TFT_base")), f4(metrik(r, "TFT_IF")),
                     f4(metrik(r, "TFT_base", "test", "false_alarm_rate")), f4(metrik(r, "TFT_IF", "test", "false_alarm_rate")),
                     f4(r.get("baselines", {}).get("persistence_test", {}).get("f1_macro")),
                     f4(r.get("baselines", {}).get("persistence_test", {}).get("accuracy"))))
    md_table(["(W,T)", "batch", "val base", "val IF", "test base", "test IF", "FAR base", "FAR IF", "persistence F1", "persistence acc"], rows)
    say(f"## Lima seed pada W={a.W}, T={a.T}")
    say()
    sel = sorted([r for (W, T, sd), r in last.items() if W == a.W and T == a.T], key=lambda r: r["config"].get("seed"))
    rows = [(r["config"].get("seed"), f4(metrik(r, "TFT_base")), f4(metrik(r, "TFT_IF")),
             f4(metrik(r, "TFT_base", "test", "false_alarm_rate")), f4(metrik(r, "TFT_IF", "test", "false_alarm_rate")),
             f4(tfree(r, "TFT_base", "macro", "roc_auc")), f4(tfree(r, "TFT_IF", "macro", "roc_auc")),
             f4(tfree(r, "TFT_base", "normal_vs_attack", "pr_auc")), f4(tfree(r, "TFT_IF", "normal_vs_attack", "pr_auc"))) for r in sel]
    md_table(["seed", "F1 base", "F1 IF", "FAR base", "FAR IF", "ROC makro base", "ROC makro IF", "PR N-vs-A base", "PR N-vs-A IF"], rows)
    if sel:
        def col(m, k, split="test"):
            return [metrik(r, m, split, k) for r in sel]
        md_table(["ringkasan (n=" + str(len(sel)) + ")", "TFT-base", "TFT-IF"],
                 [("F1 macro uji", mean_sd(col("TFT_base", "f1_macro")), mean_sd(col("TFT_IF", "f1_macro"))),
                  ("FAR uji", mean_sd(col("TFT_base", "false_alarm_rate")), mean_sd(col("TFT_IF", "false_alarm_rate"))),
                  ("akurasi uji", mean_sd(col("TFT_base", "accuracy")), mean_sd(col("TFT_IF", "accuracy"))),
                  ("ROC-AUC makro uji", mean_sd([tfree(r, "TFT_base", "macro", "roc_auc") for r in sel]), mean_sd([tfree(r, "TFT_IF", "macro", "roc_auc") for r in sel])),
                  ("PR-AUC Normal-vs-serangan uji", mean_sd([tfree(r, "TFT_base", "normal_vs_attack", "pr_auc") for r in sel]), mean_sd([tfree(r, "TFT_IF", "normal_vs_attack", "pr_auc") for r in sel]))])
        b = sel[0].get("baselines", {})
        say(f"Acuan (uji): persistence F1 macro {f4(b.get('persistence_test', {}).get('f1_macro'))}, akurasi {f4(b.get('persistence_test', {}).get('accuracy'))}, FAR {f4(b.get('persistence_test', {}).get('false_alarm_rate'))}; "
            f"selalu-Normal F1 macro {f4(b.get('always_normal_test', {}).get('f1_macro'))}, akurasi {f4(b.get('always_normal_test', {}).get('accuracy'))}.")
        say()
        dif = [metrik(r, "TFT_IF") - metrik(r, "TFT_base") for r in sel if metrik(r, "TFT_IF") is not None and metrik(r, "TFT_base") is not None]
        if not dif:
            return
        say(f"Selisih F1 macro uji (IF - base) per seed: {', '.join(f'{x:+.4f}' for x in dif)}; rata-rata {np.mean(dif):+.4f}; IF lebih tinggi pada {sum(x > 0 for x in dif)} dari {len(dif)} seed.")
        say()
        pk = [r.get("if_score_feature_rank", {}).get("rank") for r in sel]
        say(f"Peringkat `if_score` pada VSN per seed: {pk}")
        say()


# --------------------------------------------------------------------------- 3.11 evaluasi
def _dari_konfusi(cm):
    cm = np.asarray(cm, dtype=float)
    tp = np.diag(cm)
    prec = tp / np.maximum(cm.sum(0), 1)
    rec = tp / np.maximum(cm.sum(1), 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    normal_row = cm[NORMAL]
    far = (normal_row.sum() - normal_row[NORMAL]) / normal_row.sum()
    return prec, rec, f1, far, tp.sum() / cm.sum()


def cmd_evaluasi(a, ctx):
    say(f"# 3.11 Evaluasi: pemeriksaan metrik (W={a.W}, T={a.T})")
    say()
    runs = [r for r in baca_runs(a.runs_dir) if r["config"].get("W") == a.W and r["config"].get("T") == a.T]
    if not runs:
        say("Tidak ada run untuk (W,T) ini.")
        return
    r = sorted(runs, key=lambda r: r["config"].get("run_id", ""))[-1]
    say(f"Run contoh: {r['config'].get('run_id')}")
    say()
    for m in ("TFT_base", "TFT_IF"):
        res = r["results_" + m]
        prec, rec, f1, far, acc = _dari_konfusi(res["confusion_matrix"])
        say(f"## {m}: dihitung ulang dari matriks konfusi (uji)")
        say()
        md_table(["kelas", "presisi", "recall", "F1"], [(NAMES[k], f4(prec[k]), f4(rec[k]), f4(f1[k])) for k in range(C)])
        md_table(["metrik", "dari matriks", "tersimpan", "selisih"],
                 [("F1 macro", f4(f1.mean()), f4(res["f1_macro"]), f"{abs(f1.mean() - res['f1_macro']):.1e}"),
                  ("akurasi", f4(acc), f4(res["accuracy"]), f"{abs(acc - res['accuracy']):.1e}"),
                  ("FAR", f4(far), f4(res["false_alarm_rate"]), f"{abs(far - res['false_alarm_rate']):.1e}")])
        say("Matriks konfusi uji (baris = kelas benar, kolom = prediksi):")
        say()
        kode(pd.DataFrame(res["confusion_matrix"], index=NAMES, columns=NAMES))
    for m in ("TFT_base", "TFT_IF"):
        t = r["results_" + m].get("threshold_free")
        if t:
            say(f"## {m}: ROC-AUC dan PR-AUC (uji)")
            say()
            rows = [(k, f4(v["roc_auc"]), f4(v["pr_auc"]), f4(v["prevalence"])) for k, v in t["per_class"].items()]
            rows.append(("makro", f4(t["macro"]["roc_auc"]), f4(t["macro"]["pr_auc"]), f4(t["macro"]["prevalence"])))
            rows.append(("Normal vs serangan", f4(t["normal_vs_attack"]["roc_auc"]), f4(t["normal_vs_attack"]["pr_auc"]), f4(t["normal_vs_attack"]["prevalence"])))
            md_table(["kelas", "ROC-AUC", "PR-AUC", "prevalensi"], rows)
    say("Catatan: ROC-AUC/PR-AUC mulai tersedia pada run setelah patch AUC. Run lama tidak memuatnya (gunakan eval_auc_from_run.py).")
    say()


# --------------------------------------------------------------------------- ambang
def _binary(y_true, y_pred):
    yt, yp = y_true != NORMAL, y_pred != NORMAL
    tp, fp, fn, tn = int((yt & yp).sum()), int((~yt & yp).sum()), int((yt & ~yp).sum()), int((~yt & ~yp).sum())
    pr = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    return {"prec_serangan": pr, "recall_serangan": rc, "f1_serangan": 2 * pr * rc / (pr + rc) if pr + rc else 0.0,
            "FAR": fp / (fp + tn) if fp + tn else float("nan")}


def _empat(y, p):
    from sklearn.metrics import f1_score
    o = {"f1_macro": float(f1_score(y, p, labels=list(range(C)), average="macro", zero_division=0)), "acc": float((y == p).mean())}
    o.update(_binary(y, p))
    return o


def _pred_prior(p, pi, al):
    return (p * (pi[None, :] ** al)).argmax(1)


def _pred_tau(p, tau):
    return np.where(1.0 - p[:, NORMAL] > tau, p[:, 1:].argmax(1) + 1, NORMAL)


def analisis_ambang(pv, yv, pt, yt, pi):
    from sklearn.metrics import f1_score
    f = lambda y, p: f1_score(y, p, labels=list(range(C)), average="macro", zero_division=0)
    res = {"argmax": (None, _empat(yv, pv.argmax(1)), _empat(yt, pt.argmax(1)))}
    al = max(np.linspace(0, 1, 11), key=lambda x: f(yv, _pred_prior(pv, pi, x)))
    res["prior"] = (f"alpha={al:.2f}", _empat(yv, _pred_prior(pv, pi, al)), _empat(yt, _pred_prior(pt, pi, al)))
    ta = max(np.linspace(0.01, 0.99, 99), key=lambda x: f(yv, _pred_tau(pv, x)))
    res["ambang"] = (f"tau={ta:.2f}", _empat(yv, _pred_tau(pv, ta)), _empat(yt, _pred_tau(pt, ta)))
    return res


def cmd_ambang(a, ctx):
    import torch
    from evaluate import predict_proba
    from sliding_window import build_windows, build_last_labels
    from tft_model import TemporalFusionTransformer
    from sklearn.metrics import roc_auc_score
    if not a.run_dir:
        sys.exit("Perintah 'ambang' memerlukan --run-dir.")
    W, T = a.W, a.T
    WINDOW.W, WINDOW.T = W, T
    say(f"# 3.11 Analisis ambang keputusan (W={W}, T={T}); parameter dipilih di validasi, angka = uji")
    say()
    trf, vaf, tef, art = ctx.final
    tf, vf, ef, cols_if, _ = ctx.with_if
    cat = art.categorical_feature_indices or {}
    pi = np.bincount(trf[DATA.target_col].to_numpy().astype(int), minlength=C).astype(float)
    pi /= pi.sum()
    data = {"TFT_base": (build_windows(vaf, art.feature_cols), build_windows(tef, art.feature_cols), len(art.feature_cols)),
            "TFT_IF": (build_windows(vf, cols_if), build_windows(ef, cols_if), len(cols_if))}
    last_te = build_last_labels(tef)
    (Xv, Yv), (Xt, Yt), _ = data["TFT_IF"]
    yv, yt = Yv.reshape(-1), Yt.reshape(-1)
    s_v, s_t = np.repeat(Xv[:, -1, -1], T), np.repeat(Xt[:, -1, -1], T)
    cand = np.unique(np.quantile(s_v, np.linspace(0.5, 0.999, 200)))
    tb = max(cand, key=lambda t: _binary(yv, (s_v > t).astype(int))["f1_serangan"])
    ib = _binary(yt, (s_t > tb).astype(int))
    pers = _binary(yt, np.repeat(last_te, T))
    md_table(["pembanding (uji, biner)", "ROC-AUC", "presisi", "recall", "F1 serangan", "FAR"],
             [("skor IF rekaman terakhir vs label berikutnya", f4(roc_auc_score((yt != NORMAL).astype(int), s_t)), f4(ib["prec_serangan"]), f4(ib["recall_serangan"]), f4(ib["f1_serangan"]), f4(ib["FAR"])),
              ("persistence", "-", f4(pers["prec_serangan"]), f4(pers["recall_serangan"]), f4(pers["f1_serangan"]), f4(pers["FAR"]))])
    device = torch.device(TRAIN.device if torch.cuda.is_available() else "cpu")
    agg = {}
    for d in a.run_dir:
        with open(os.path.join(d, "results_summary.json")) as f:
            c = json.load(f)["config"]
        for nm, ck in (("TFT_base", "tft_base.pt"), ("TFT_IF", "tft_if.pt")):
            (Xva, Yva), (Xte, Yte), nf = data[nm]
            m = TemporalFusionTransformer(num_features=nf, hidden_size=c["hidden_size"], num_classes=MODEL.num_classes, T=c["T"],
                                          lstm_layers=MODEL.lstm_layers, attn_heads=c["attn_heads"], dropout=c["dropout"],
                                          categorical_features=cat).to(device)
            m.load_state_dict(torch.load(os.path.join(d, ck), map_location=device))
            pv = predict_proba(m, Xva).reshape(-1, C)
            pt = predict_proba(m, Xte).reshape(-1, C)
            res = analisis_ambang(pv, Yva.reshape(-1), pt, Yte.reshape(-1), pi)
            for cara, (par, _, te_) in res.items():
                agg.setdefault((nm, cara), []).append((c.get("seed"), par, te_))
    rows = []
    for (nm, cara), lst in agg.items():
        f1 = [x[2]["f1_macro"] for x in lst]
        rows.append((nm, cara, "; ".join(f"{x[0]}:{x[1]}" for x in lst if x[1]), mean_sd(f1),
                     mean_sd([x[2]["FAR"] for x in lst]), mean_sd([x[2]["recall_serangan"] for x in lst]),
                     mean_sd([x[2]["prec_serangan"] for x in lst]), mean_sd([x[2]["f1_serangan"] for x in lst])))
    md_table(["model", "cara", "parameter (seed:nilai)", "F1 macro", "FAR", "recall serangan", "presisi serangan", "F1 serangan"], rows)


def cmd_blok(a, ctx):
    """3.11.7: data uji dibagi K blok berurutan; metrik tiap blok (argmax dan koreksi prior)."""
    import torch
    from evaluate import predict_proba
    from sliding_window import build_windows
    from tft_model import TemporalFusionTransformer
    from sklearn.metrics import f1_score
    if not a.run_dir:
        sys.exit("Perintah 'blok' memerlukan --run-dir.")
    K = a.blok
    W, T = a.W, a.T
    WINDOW.W, WINDOW.T = W, T
    say(f"# 3.11 Evaluasi uji per blok berurutan (W={W}, T={T}, {K} blok); alpha dipilih di validasi")
    say()
    trf, vaf, tef, art = ctx.final
    tf, vf, ef, cols_if, _ = ctx.with_if
    cat = art.categorical_feature_indices or {}
    pi = np.bincount(trf[DATA.target_col].to_numpy().astype(int), minlength=C).astype(float)
    pi /= pi.sum()
    data = {"TFT_base": (build_windows(vaf, art.feature_cols), build_windows(tef, art.feature_cols), len(art.feature_cols)),
            "TFT_IF": (build_windows(vf, cols_if), build_windows(ef, cols_if), len(cols_if))}
    device = torch.device(TRAIN.device if torch.cuda.is_available() else "cpu")
    agg = {}
    info = {}
    for d in a.run_dir:
        with open(os.path.join(d, "results_summary.json")) as f:
            c = json.load(f)["config"]
        for nm, ck in (("TFT_base", "tft_base.pt"), ("TFT_IF", "tft_if.pt")):
            (Xva, Yva), (Xte, Yte), nf = data[nm]
            m = TemporalFusionTransformer(num_features=nf, hidden_size=c["hidden_size"], num_classes=MODEL.num_classes, T=c["T"],
                                          lstm_layers=MODEL.lstm_layers, attn_heads=c["attn_heads"], dropout=c["dropout"],
                                          categorical_features=cat).to(device)
            m.load_state_dict(torch.load(os.path.join(d, ck), map_location=device))
            pv = predict_proba(m, Xva).reshape(-1, C)
            pt = predict_proba(m, Xte).reshape(-1, C)
            yv, yt = Yva.reshape(-1), Yte.reshape(-1)
            f = lambda y, p: f1_score(y, p, labels=list(range(C)), average="macro", zero_division=0)
            al = max(np.linspace(0, 1, 11), key=lambda x: f(yv, _pred_prior(pv, pi, x)))
            edges = np.linspace(0, len(yt), K + 1).astype(int)
            for b in range(K):
                sl = slice(edges[b], edges[b + 1])
                y = yt[sl]
                ada = np.unique(y)
                info[b] = (len(y), float((y != NORMAL).mean()), ", ".join(NAMES[int(k)] for k in ada))
                for cara, pred in (("argmax", pt[sl].argmax(1)), (f"prior (alpha dari validasi)", _pred_prior(pt[sl], pi, al))):
                    r = _binary(y, pred)
                    r["f1_macro_ada"] = float(f1_score(y, pred, labels=list(ada), average="macro", zero_division=0))
                    agg.setdefault((nm, cara, b), []).append(r)
    say("Isi tiap blok (jumlah target, proporsi serangan, kelas yang ada):")
    say()
    md_table(["blok", "target", "% serangan", "kelas ada"], [(b + 1, ribu(n), f"{p * 100:.2f}", k) for b, (n, p, k) in info.items()])
    say("F1 macro dihitung hanya atas kelas yang ada pada blok itu. Nilai = rerata dan simpangan baku antar seed.")
    say()
    rows = []
    for (nm, cara, b), lst in sorted(agg.items()):
        rows.append((nm, cara, b + 1, mean_sd([x["f1_macro_ada"] for x in lst]), mean_sd([x["FAR"] for x in lst]),
                     mean_sd([x["recall_serangan"] for x in lst]), mean_sd([x["prec_serangan"] for x in lst])))
    md_table(["model", "cara", "blok", "F1 macro", "FAR", "recall serangan", "presisi serangan"], rows)


def _f1m_dari_cm(cm):
    tp = np.diag(cm).astype(float)
    pr = np.divide(tp, cm.sum(0), out=np.zeros(len(tp)), where=cm.sum(0) > 0)
    rc = np.divide(tp, cm.sum(1), out=np.zeros(len(tp)), where=cm.sum(1) > 0)
    f1 = np.divide(2 * pr * rc, pr + rc, out=np.zeros(len(tp)), where=(pr + rc) > 0)
    return float(f1.mean())


def cmd_uji(a, ctx):
    """3.11: uji selisih TFT-IF dan TFT-base. (a) uji-t berpasangan antar seed; (b) bootstrap blok pada data uji per seed."""
    import torch
    from evaluate import predict_proba
    from sliding_window import build_windows
    from tft_model import TemporalFusionTransformer
    from sklearn.metrics import f1_score
    from scipy import stats
    if not a.run_dir:
        sys.exit("Perintah 'uji' memerlukan --run-dir.")
    W, T = a.W, a.T
    WINDOW.W, WINDOW.T = W, T
    L, B = a.panjang_blok, a.ulang
    say(f"# 3.11 Uji selisih TFT-IF dan TFT-base (W={W}, T={T}); bootstrap blok L={L}, {B} ulangan")
    say()
    trf, vaf, tef, art = ctx.final
    tf, vf, ef, cols_if, _ = ctx.with_if
    cat = art.categorical_feature_indices or {}
    pi = np.bincount(trf[DATA.target_col].to_numpy().astype(int), minlength=C).astype(float)
    pi /= pi.sum()
    data = {"TFT_base": (build_windows(vaf, art.feature_cols), build_windows(tef, art.feature_cols), len(art.feature_cols)),
            "TFT_IF": (build_windows(vf, cols_if), build_windows(ef, cols_if), len(cols_if))}
    device = torch.device(TRAIN.device if torch.cuda.is_available() else "cpu")
    f = lambda y, p: f1_score(y, p, labels=list(range(C)), average="macro", zero_division=0)
    per_seed = {}
    for d in a.run_dir:
        with open(os.path.join(d, "results_summary.json")) as fh:
            c = json.load(fh)["config"]
        preds = {}
        for nm, ck in (("TFT_base", "tft_base.pt"), ("TFT_IF", "tft_if.pt")):
            (Xva, Yva), (Xte, Yte), nf = data[nm]
            m = TemporalFusionTransformer(num_features=nf, hidden_size=c["hidden_size"], num_classes=MODEL.num_classes, T=c["T"],
                                          lstm_layers=MODEL.lstm_layers, attn_heads=c["attn_heads"], dropout=c["dropout"],
                                          categorical_features=cat).to(device)
            m.load_state_dict(torch.load(os.path.join(d, ck), map_location=device))
            pv = predict_proba(m, Xva).reshape(-1, C)
            pt = predict_proba(m, Xte).reshape(-1, C)
            yv, yt = Yva.reshape(-1), Yte.reshape(-1)
            al = max(np.linspace(0, 1, 11), key=lambda x: f(yv, _pred_prior(pv, pi, x)))
            preds[nm] = {"argmax": pt.argmax(1), "prior": _pred_prior(pt, pi, al)}
        per_seed[c.get("seed")] = (yt, preds)
    rng = np.random.default_rng(0)
    hasil_seed, hasil_boot = {}, []
    for cara in ("argmax", "prior"):
        dif, boot_rows = [], []
        for sd, (yt, preds) in per_seed.items():
            n = len(yt)
            nb = n // L
            def cms(p):
                out = np.zeros((nb, C, C), dtype=np.int64)
                for b in range(nb):
                    sl = slice(b * L, (b + 1) * L)
                    out[b] = np.bincount(yt[sl] * C + p[sl], minlength=C * C).reshape(C, C)
                return out
            cb, ci = cms(preds["TFT_base"][cara]), cms(preds["TFT_IF"][cara])
            f1b, f1i = _f1m_dari_cm(cb.sum(0)), _f1m_dari_cm(ci.sum(0))
            dif.append((sd, f1b, f1i))
            ds = np.empty(B)
            for r in range(B):
                idx = rng.integers(0, nb, nb)
                ds[r] = _f1m_dari_cm(ci[idx].sum(0)) - _f1m_dari_cm(cb[idx].sum(0))
            lo, hi = np.percentile(ds, [2.5, 97.5])
            boot_rows.append((sd, f4(f1b), f4(f1i), f"{f1i - f1b:+.4f}", f"[{lo:+.4f}; {hi:+.4f}]", "tidak memuat 0" if lo > 0 or hi < 0 else "memuat 0"))
        say(f"## Keputusan: {cara}")
        say()
        say("Bootstrap blok per seed (selisih = F1 macro TFT-IF dikurangi TFT-base, interval 95%):")
        say()
        md_table(["seed", "F1 macro base", "F1 macro IF", "selisih", "interval 95%", "terhadap 0"], boot_rows)
        b_arr = np.array([x[1] for x in dif]); i_arr = np.array([x[2] for x in dif])
        if len(dif) > 1:
            t, p = stats.ttest_rel(i_arr, b_arr)
            say(f"Uji-t berpasangan atas {len(dif)} seed: selisih rerata {np.mean(i_arr - b_arr):+.4f}, t = {t:.3f}, p = {p:.4f}. Seed dengan selisih positif: {int((i_arr > b_arr).sum())} dari {len(dif)}.")
        say()


# --------------------------------------------------------------------------- main
PERLU_DATA = {"data", "praproses", "manual", "skorif", "window", "model", "pelatihan", "ambang", "blok", "uji"}
CMD = {"kerangka": cmd_kerangka, "lingkungan": cmd_lingkungan, "data": cmd_data, "praproses": cmd_praproses,
       "manual": cmd_manual, "skorif": cmd_skorif, "window": cmd_window, "model": cmd_model,
       "pelatihan": cmd_pelatihan, "skenario": cmd_skenario, "evaluasi": cmd_evaluasi, "ambang": cmd_ambang, "blok": cmd_blok, "uji": cmd_uji}
SEMUA = ["kerangka", "lingkungan", "data", "praproses", "skorif", "window", "pelatihan", "skenario", "evaluasi"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("perintah", choices=list(CMD) + ["semua"])
    ap.add_argument("--blok", type=int, default=3)
    ap.add_argument("--panjang-blok", type=int, default=2000)
    ap.add_argument("--ulang", type=int, default=2000)
    ap.add_argument("--raw-dir", default=DATA.raw_dir)
    ap.add_argument("--runs-dir", default=os.path.join(ROOT, "logs", "runs"))
    ap.add_argument("--W", type=int, default=25)
    ap.add_argument("--T", type=int, default=1)
    ap.add_argument("--run-dir", nargs="+", default=None)
    ap.add_argument("--fitur", default="dur,sbytes,sttl,ct_srv_dst")
    ap.add_argument("--bangun", action="store_true")
    a = ap.parse_args()
    WINDOW.W, WINDOW.T = a.W, a.T
    ctx = Ctx(a.raw_dir)
    daftar = SEMUA if a.perintah == "semua" else [a.perintah]
    for i, p in enumerate(daftar):
        if i:
            say()
            say("---")
            say()
        if p == "kerangka" and a.perintah == "kerangka":
            try:
                cmd_kerangka(a, ctx)
            except Exception as e:  # data mentah tidak ada: tetap tampilkan inventaris run
                say(f"(Bagian angka tahap dilewati: {e})")
                cmd_kerangka(a, None)
        elif p == "kerangka":
            cmd_kerangka(a, ctx)
        elif a.perintah == "semua":
            try:
                CMD[p](a, ctx)
            except Exception as e:  # satu bagian gagal tidak menghentikan yang lain
                say(f"(Bagian {p} gagal: {type(e).__name__}: {e})")
        else:
            CMD[p](a, ctx)
    simpan(a.perintah)


if __name__ == "__main__":
    main()
