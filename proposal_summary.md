# proposal_summary.md — Konteks Proyek TFT-IF (UNSW-NB15)

> **Fungsi dokumen ini**: anchor konteks utama untuk (1) pembuatan `summary.md` per paper
> di `model_papers/summaries/`, dan (2) penyusunan `spec.md` sebagai acuan kode.
> **`spec.md` v1.1 SUDAH TERSEDIA di root** — 14 keputusan terbuka (A1–A14) sudah
> terkunci (2026-09-27) + pembelaan skripsi + implementasi + smoke test lolos.
> Jadikan `spec.md` v1.1 referensi teknis utama; dokumen ini menjadi konteks
> proposal & peta bahan.
> **Sumber**: bacaan penuh `Proposal_Skripsi_Matematika_Fahri_Desian_v7_310526.pdf` (v7),
> `handoff.md`, `references_addendum.md`, `src/config.py`, `main.py`.
> **Dibuat**: 2026-09-26.

---

## 1. Identitas Proposal

- **Judul**: Sistem Deteksi Intrusi Jaringan Komputer Menggunakan *Temporal Fusion Transformer* dengan Integrasi Skor Anomali *Isolation Forest*
- **Penulis**: Fahri Desrian (2206048726) — Sarjana Matematika, FMIPA UI
- **Pembimbing**: Prof. Dr. Drs. Suryadi, M.T.
- **Versi dokumen**: v7 (disetujui 29 Mei 2026); rencana sidang Desember 2026

## 2. Ide Inti

1. NIDS diperlakukan sebagai **masalah deret waktu multivariat**: rekaman lalu lintas
   diurutkan kronologis memakai timestamp (`Stime`/`Ltime`), lalu dibentuk jendela geser
   `{X_{t−W+1},…,X_t} → Y_{(t+1,…,t+T)}` — bukan klasifikasi tabular per-rekaman.
2. **Isolation Forest (IF)** dilatih *unsupervised* **hanya pada subset kelas Normal
   dari data latih**, menghasilkan skor anomali kontinu `s ∈ (0,1)` per rekaman.
   Skor ini **bukan hasil klasifikasi**, melainkan **fitur tambahan ke-(F+1)**.
3. Dua model dilatih paralel sebagai kondisi komparasi:
   - **TFT-base**: fitur jaringan non-target saja (kondisi kontrol, tanpa skor anomali).
   - **TFT-IF**: fitur jaringan + skor anomali IF.
   - **Hipotesis**: TFT-IF > TFT-base, khususnya dalam mengenali pola serangan yang tidak normal.
4. TFT diadaptasi dari *multi-horizon forecasting* ke **klasifikasi temporal multikelas**:
   head akhir diganti `Linear + softmax` (4 kelas) per titik waktu horizon, loss
   **cross-entropy** (bukan quantile loss). *Known future inputs* **tidak dipakai**
   (tidak tersedia secara alami dalam deteksi intrusi).
5. Bukan prediksi waktu terjadinya serangan — melainkan estimasi **kecenderungan status
   lalu lintas pada horizon pendek** berdasarkan pola historis.
6. Interpretabilitas menjadi bagian eksplisit: *feature importance* dari **Variable
   Selection Network (VSN)**, *attention weight*, dan kontribusi skor anomali IF.

### Rumusan Masalah / Tujuan
1. Bagaimana merancang & mengimplementasikan integrasi skor anomali IF dalam model TFT untuk NIDS?
2. Bagaimana performa model TFT dengan integrasi skor anomali IF tersebut?

### Batasan Penelitian (Bagian 4 proposal)
1. Dataset UNSW-NB15 (±2,5 juta rekaman, 49 atribut, label kelas).
2. Klasifikasi 4 kelas: **Normal, Generic, Exploits, Fuzzers**.
3. Representasi deret waktu via sliding window; **W** (historis) dan **T** (target)
   ditentukan lewat eksperimen ablasi dalam rentang terbatas.
4. Ketidakseimbangan kelas ditangani utamanya dengan **class weighting** (tidak mengubah
   struktur temporal); oversampling/undersampling hanya eksperimen pembanding.
5. Evaluasi *offline*: akurasi, presisi, recall, F1 (macro & weighted), **FAR**;
   presisi/recall/F1 diprioritaskan pada data tidak seimbang.

## 3. Dataset UNSW-NB15

- Sumber: ACCS, UNSW (Moustafa & Slay 2015/2016). File: partisi `UNSW-NB15_1.csv` …
  `UNSW-NB15_4.csv` + dokumen fitur `UNSW-NB15_features.csv`.
- 49 atribut **termasuk** 2 kolom target: `attack_cat` (target multikelas) dan `Label`
  (label biner — tidak dipakai sebagai fitur).
- `Stime`/`Ltime` **hanya untuk pengurutan kronologis** (bukan fitur input).
- Kelompok fitur: basic, content, time, flow, generated.

### Distribusi kelas (Tabel 3 proposal; ★ = dipakai)

| Kelas | Rekaman | % |
|---|---|---|
| Normal ★ | 2.218.761 | 87,35% |
| Generic ★ | 215.481 | 8,48% |
| Exploits ★ | 44.525 | 1,75% |
| Fuzzers ★ | 24.246 | 0,95% |
| DoS | 16.353 | 0,64% |
| Reconnaissance | 13.987 | 0,55% |
| Analysis | 2.677 | 0,11% |
| Backdoors | 2.329 | 0,09% |
| Shellcode | 1.133 | 0,04% |
| Worms | 174 | 0,01% |

- IR keseluruhan (Normal:Worms) ≈ **12.751**; pada subset 4 kelas, IR Normal:Fuzzers ≈ **91,5**.
- UNSW-NB15 juga dilaporkan punya masalah **class overlap** (Zoghi & Serpen, 2021).
- Kuirk CSV asli yang sudah ditangani kode: kolom numerik berisi string liar (port hex
  seperti `0x000c`, spasi kosong), casing/whitespace tidak konsisten pada
  `attack_cat`/`proto`/`state`/`service`.

### ✔ Diskrepansi fitur proposal vs kode: TERSELESAIKAN (A1, 2026-09-27)

- **Kode (terkunci)**: `srcip, sport, dstip, dsport` di-drop → TFT-base = **41**,
  TFT-IF = **42**. Pembelaan lengkap di `spec.md` §0.1 (identity leak, tanpa skema
  encoding di proposal, pseudo-kategorikal hex, framing perilaku). Angka 45/46 di
  proposal direvisi di skripsi; varian 45/46 opsional sebagai cek sensitivitas.

## 4. Pipeline Eksperimen (Bagian 6 proposal) — pemetaan ke kode

Urutan wajib sekuensial (keluaran tiap tahap jadi masukan tahap berikutnya); bersifat
iteratif pada: strategi balancing, ablasi W/T/rasio, dan retrain fitur terpilih VSN.

| Tahap proposal | Isi ringkas | Modul kode | Status |
|---|---|---|---|
| 6.a studi literatur | — | — | (non-kode) |
| 6.b pengumpulan data | Muat 4 CSV + dokumen fitur | `data_loading.py` | ✔ (cek header CSV asli) |
| 6.c EDA & visualisasi | Distribusi kelas, imbalance, null/inf/dup, deskriptif, histogram, boxplot, heatmap, bar chart | `eda.py` | ✔ (A14; output `logs/eda/`) |
| 6.d praproses | Filter 4 kelas (+assertion A10); bersihkan null/inf/dup; urut kronologis; split 70:15:15 tanpa shuffle (A2); **imputasi median fit-on-train (A8)**; encoding + token "unknown" (A9); Min-Max kontinu fit-on-train | `preprocessing.py` | ✔ (v1.1) |
| 6.e imbalance | Baseline tanpa balancing dulu; utama **class weighting** hanya di data latih; oversampling/undersampling hanya pembanding | `train.py` (flag `use_class_weighting`) | ✔ class weighting; ✘ oversampling/undersampling |
| 6.f integrasi IF | IF *unsupervised* pada subset Normal-train (`n_estimators`, `max_samples`, `random_state`; `contamination` tak dipakai utk threshold); skor kontinu untuk semua split; fitur ke-(F+1) | `isolation_forest_module.py` | ✔ |
| 6.g sliding window | Sekuens W→T per split; jaga batas antar-split **dan** batas antar-file/segmen; ablasi W, T, rasio | `sliding_window.py` + `--window/--horizon` | ✔; ablasi terkunci A3 (grid W∈{10,25,50}×T∈{1,3}, selektor F1-macro val, anchor Psychogyios 2024) — **belum dieksekusi** |
| 6.h bangun & latih TFT | Konfigurasi `encoder_length, prediction_length, hidden_size`; TFT-base vs TFT-IF paralel; loss multikelas + Adam; LR scheduler; validasi berkala + early stopping | `tft_model.py`, `train.py`, `hyperparameter_tuning.py` (Optuna, objektif F1-macro validasi) | ✔ (v1.1: interpretable attention Eq.13–16 + embedding kategorikal) |
| 6.i evaluasi | Metrik: akurasi, presisi, recall, F1 (macro & weighted), FAR; confusion matrix; bandingkan base vs IF & dengan/tanpa balancing | `evaluate.py` | ✔ (v1.1: + per-kelas & per-horizon, metrik val utk ablasi) |
| 6.j interpretabilitas | Feature importance VSN; attention weight; kontribusi skor IF; visualisasi; **opsional** retrain dengan fitur terpilih VSN (pendukung) | `interpret.py` | ✔ VSN+attention; ✘ retrain fitur terpilih VSN |

Keputusan desain yang **sudah dikunci kode** (dari `handoff.md` + `config.py`):
1. `srcip, sport, dstip, dsport` di-drop (→ 41/42 fitur). *Perlu konfirmasi + konsistensi skripsi.*
2. Decoder TFT memakai **learned positional query tokens** (bukan known-future-inputs).
3. **Tidak ada static covariates** (`static_enrichment_grn` berjalan tanpa context vector).
4. TFT **diimplementasikan dari nol** dengan PyTorch (bukan `pytorch-forecasting`):
   VSN, GRN (GLU + ELU + residual + LayerNorm), LSTM encoder-decoder,
   *masked interpretable multi-head attention* (causal mask), head klasifikasi Linear+softmax.
5. Hyperparameter hasil tuning Optuna **pada TFT-IF** dipakai juga untuk TFT-base
   (arsitektur identik; yang membedakan hanya jumlah fitur).
6. Training pakai **AMP (mixed precision)**; checkpoint disimpan ke CPU.
7. Seed global 42; split default 70:15:15.

## 5. Metrik Evaluasi & Definisi FAR (Bagian 5.8)

- `Accuracy = (TP+TN)/(TP+TN+FP+FN)`; `Precision = TP/(TP+FP)`; `Recall = TP/(TP+FN)`;
  `F1 = 2PR/(P+R)`; `F1_macro = (1/K)Σ F1_k`; `F1_weighted = Σ (n_k/N) F1_k`.
- **FAR (definisi khusus, agregasi Normal-vs-Attack)**: `FAR = FP/(FP+TN)` dengan
  TN = Normal→Normal dan FP = Normal→(Generic | Exploits | Fuzzers).
  **Kesalahan antar kelas serangan TIDAK dihitung sebagai FAR** (tetap terlihat di
  confusion matrix 4×4).
- Multikelas: perhitungan per kelas secara *one-vs-rest*; F1 macro (sensitif kelas
  minoritas) dan weighted (dipengaruhi mayoritas) dilaporkan bersamaan.

## 6. Matematika Kunci (untuk spesifikasi implementasi)

1. Sliding window: `{(X_{t−W+1},…,X_t)} → Y_{(t+1,…,t+T)}`, `X_t ∈ ℝ^F` (geser 1 langkah/iterasi → stride 1).
2. Skor anomali IF: `s(x,n) = 2^{−E[h(x)]/c(n)}`; `c(n) = 2H(n−1) − 2(n−1)/n`;
   `H(i) ≈ ln(i) + 0,5772` (Euler–Mascheroni). Interpretasi: →1 anomali, →0 normal, ≈0,5 ambigu.
3. Self-attention: `Attention(Q,K,V) = softmax(QKᵀ/√d_k)V`; diperluas ke multi-head.
4. GRN: `GRN(a,c) = LayerNorm(a + GLU(η₁))`.
5. Imbalance ratio: `IR = max(n_k)/min(n_k)`.

## 7. Parameter & Default Kode Saat Ini (`src/config.py`)

| Config | Nilai default | Catatan |
|---|---|---|
| Split | 0.70 / 0.15 / 0.15 | kronologis, tanpa shuffle |
| Imputasi | median (null numerik) | + coerce kolom numerik string liar |
| IF | n_estimators=100, max_samples="auto", contamination="auto", random_state=42 | skor kontinu, tanpa threshold |
| Window | W=10, T=3, stride=1 | **placeholder — wajib ablasi** |
| Model | hidden_size=64, lstm_layers=1, attn_heads=4, dropout=0.1, num_classes=4 | dioptimalkan Optuna (20 trial, F1-macro val) |
| Train | batch=1024, epochs=30, lr=1e-3, wd=1e-5, ES patience=5, scheduler patience=2/factor 0.5, AMP, seed 42, num_workers=6 | target GPU RTX 3060 12GB |

Status verifikasi: pipeline **lulus smoke test end-to-end dengan data sintetis**
pada versi v1.1 (2026-09-27, setelah refactor keputusan A1–A14; termasuk
`eda.py`) — **belum pernah dijalankan pada data UNSW-NB15 asli** dengan
konfigurasi final; `data/raw/` masih berisi dummy (regenerate saat run asli).

## 8a. Keputusan Terkunci (ringkas — detail & pembelaan di `spec.md` §0)

A1: 41/42 fitur · A2: 70:15:15 · A3: ablasi W∈{10,25,50}×T∈{1,3} (TFT-IF,
selektor F1-macro val, anchor Psychogyios 2024: F1 jenuh di W=50) ·
A4: class weighting `balanced` · A5: early stopping val_loss (Prechelt 1998 —
tambah bibliografi) · A6: hyperparameter identik kedua model · A7: CSV
headerless 49 kolom · A8: imputasi median fit-on-train · A9: token "unknown" ·
A10: assertion attack_cat kosong∧label=1 · A11: skor IF literal s(x,n)
(`-score_samples`, tanpa min-max) · A12: interpretable attention Eq. 13–16 ·
A13: entity embeddings kategorikal (tanpa scaler utk kategorikal — hindari
bug round-off) · A14: per-kelas, per-horizon, distribusi split, eda.py.

## 8. Keputusan yang Masih Menggantung (lihat `spec.md` §12)

Semua keputusan desain telah terkunci (A1–A14, `spec.md` §0). Tersisa hanya
**eksekusi** dan item opsional:

1. **Eksekusi ablasi W/T** (A3) — perintah siap: `python main.py --window W --horizon T --epochs 30`; 6 kombinasi ≈ 3 jam (W=50 butuh ±24–32 GB RAM).
2. **Sensitivitas 80:10:10** — opsional.
3. **Oversampling/undersampling pembanding** (6.e.iii) — class weighting terkunci
   sebagai strategi utama (A4); comparator opsional bila waktu.
4. **Retrain fitur terpilih VSN** (6.j.ii) & **persentil importance** — opsional.
5. **Tambah Prechelt (1998) ke bibliografi** (dasar A5).
6. **Checkpoint lama tidak valid** — arsitektur berubah (A12/A13); retrain wajib.

## 9. Inventaris Paper di `model_papers/` & Rencana Summary

| File | Paper | Peran dalam implementasi |
|---|---|---|
| `NIPS-2017-attention-is-all-you-need-Paper.pdf` | Vaswani et al. 2017 | dasar scaled dot-product self-attention (sudah di pustaka proposal) |
| `dauphin17a.pdf` | Dauphin et al. 2017 | GLU di dalam GRN/VSN |
| `1511.07289v5.pdf` | Clevert et al. 2016 | aktivasi ELU di GRN |
| `1607.06450v1.pdf` | Ba et al. 2016 | Layer Normalization |
| `1412.6980v9.pdf` | Kingma & Ba 2015 | optimizer Adam |
| `1912.01703v1.pdf` | Paszke et al. 2019 | framework PyTorch |
| `pedregosa11a.pdf` | Pedregosa et al. 2011 | scikit-learn: MinMaxScaler, LabelEncoder, IsolationForest, metrik |
| `3292500.3330701.pdf` | Akiba et al. 2019 | Optuna (tuning) |
| `1710.03740v3.pdf` | Micikevicius et al. 2018 | mixed precision training (AMP) |
| `prechelt1998.pdf` | Prechelt 1998 | early stopping (keputusan A5): kelas kriteria GL/PQ/UP; patience≈kelas UP — tradeoff terbaik & robust utk split kronologis |

- **Main paper (sudah diringkas)**: `1912.09363v3.pdf` → `lim2021_tft_summary.md`
  (Lim et al. 2021, TFT — arsitektur inti); `liu2008.pdf` →
  `liu2008_isolation_forest_summary.md` (Liu et al. 2008, IF — algoritme skor anomali,
  termasuk justifikasi §5.4 untuk pelatihan normal-only).
- **Belum ada di folder** (opsional): Moustafa & Slay 2015/2016 (dataset),
  Psychogyios et al. 2024 (sliding window IDS), Liu et al. 2012 (versi jurnal IF).
- **Konvensi file summary** (folder khusus): `model_papers/summaries/<firstauthor><tahun>_<topik>_summary.md`.
  Isi hanya bagian yang benar-benar dipakai implementasi (formula, komponen,
  hyperparameter, perilaku yang menempel ke kode).
- **Progress summary: SELESAI 12/12** — 9 referensi implementasi + 3 pendukung
  (`lim2021_tft`, `liu2008_isolation_forest`, `prechelt1998_early_stopping`).
  Bahan lengkap untuk `spec.md`.

## 10. Catatan Struktur Kode (koreksi terhadap `handoff.md`)

- Modul berada **langsung di `src/`** (bukan `src/nids_tft_if/` seperti digambarkan tree
  di handoff.md), dan `main.py` meng-import secara flat (`from config import ...`)
  setelah menyisipkan `src/` ke `sys.path`. `pyproject.toml` tetap tersedia untuk
  `pip install -e .`, tapi nama package perlu disesuaikan bila dipakai.
- Output pipeline: `checkpoints/tft_base.pt`, `checkpoints/tft_if.pt`,
  `logs/results_summary.json` (metrik, perbandingan, ranking feature importance,
  peringkat pentingnya `if_score`).
