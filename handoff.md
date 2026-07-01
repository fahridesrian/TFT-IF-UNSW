# Handoff — Pipeline TFT + Isolation Forest untuk UNSW-NB15

Status: **selesai dibangun dan diuji end-to-end dengan data sintetis** (bentuk
mirip UNSW-NB15, termasuk null/inf/duplikat yang sengaja disisipkan).
**Belum pernah dijalankan pada data UNSW-NB15 asli** — bagian "Yang perlu
Anda lakukan" di bawah wajib dibaca sebelum eksperimen sungguhan.

## Changelog

- **[Fitur]** `train.py`: progress bar `tqdm` ditambahkan untuk loop epoch
  maupun loop batch (train & val), menampilkan loss berjalan secara live.
  Otomatis diberi label per model (`[TFT-base]` / `[TFT-IF]`) di `main.py`.
  Progress bar batch otomatis dimatikan saat hyperparameter tuning
  (`hyperparameter_tuning.py`) agar tidak spam puluhan bar sekaligus lintas
  trial Optuna — hanya ringkasan epoch yang tampil (kalau `verbose=True`).
  Nonaktifkan seluruhnya dengan `python main.py --quiet`, atau panggil
  `train_model(..., show_progress=False)` langsung.
- **[Fix]** `preprocessing.py`: kolom yang seharusnya numerik tapi berisi
  string liar (spasi kosong `' '`, port dalam format hex seperti `0x000c`)
  — kuirk yang memang dikenal ada di CSV UNSW-NB15 asli — sekarang dipaksa
  dikonversi ke numerik (`coerce_numeric_columns`), nilai yang gagal
  dikonversi menjadi NaN lalu diimputasi median seperti kolom numerik
  lainnya. Sebelumnya kolom seperti ini lolos tanpa dibersihkan karena
  pengecekan dtype tidak menangkap semua varian dtype string pandas.
  Ditambahkan juga `normalize_string_columns` untuk trim whitespace &
  menyamakan casing pada `attack_cat`/`proto`/`state`/`service` (mis.
  `' fuzzers '` → `Fuzzers`). Kasus ini sekarang direproduksi permanen di
  `tests/make_dummy_data.py` sebagai regression test.
- **[Fix]** `train.py`: mengganti `copy.deepcopy(model.state_dict())` (dua
  tempat) dengan snapshot ke CPU (`_cpu_state_dict`). `deepcopy` langsung di
  tensor CUDA rapuh pada beberapa kombinasi driver/CUDA Windows dan memicu
  `CUDA error: unknown error` yang sebenarnya berasal dari operasi async
  sebelumnya. Bonus: menghindari duplikasi memori GPU untuk checkpoint
  terbaik.
- **[Fix]** `train.py`: `GradScaler` (AMP) sebelumnya dibuat ulang setiap
  epoch — ini membuang adaptasi loss-scale yang seharusnya berlangsung
  sepanjang training. Sekarang diinisialisasi sekali di luar loop epoch.
- **[Fitur]** `config.py` / `train.py`: `num_workers` DataLoader kini bisa
  diatur lewat `TrainConfig.num_workers` (default 4), tidak lagi hardcoded.

## Struktur proyek

```
nids_tft_if/
├── pyproject.toml              # opsional: pip install -e . agar `import nids_tft_if` bisa dari mana saja
├── requirements.txt
├── main.py                     # orkestrator: jalankan pipeline penuh dari sini (python main.py)
├── src/
│   └── nids_tft_if/
│       ├── __init__.py
│       ├── config.py                  # SEMUA parameter (ubah di sini, bukan di kode inti)
│       ├── data_loading.py            # load & gabung 4 CSV UNSW-NB15
│       ├── preprocessing.py           # filter kelas, cleaning, split kronologis, encode+scale
│       ├── isolation_forest_module.py # IF dilatih di subset Normal-train, skor -> fitur
│       ├── sliding_window.py          # bentuk (X, Y) jaga batas antar file CSV
│       ├── tft_model.py               # TFT dari nol: VSN, GRN, LSTM enc-dec, causal attention
│       ├── train.py                   # manual training loop, class weighting, early stopping
│       ├── evaluate.py                # akurasi, precision/recall/F1 (macro & weighted), FAR
│       ├── interpret.py               # ekstraksi VSN weight & attention weight
│       └── hyperparameter_tuning.py   # Optuna, objektif = F1-macro validasi
├── tests/make_dummy_data.py    # generator data sintetis untuk uji pipeline (BUKAN data riil)
├── data/raw/                   # KOSONG — taruh 4 CSV UNSW-NB15 asli di sini
├── checkpoints/                 # tempat model tersimpan (tft_base.pt, tft_if.pt)
└── logs/results_summary.json   # ringkasan hasil setelah main.py dijalankan
```

Import di dalam package memakai bentuk absolut `from nids_tft_if.config import ...`
(bukan `from config import ...`). `main.py` menyisipkan `src/` ke `sys.path`
secara otomatis di baris paling atas, jadi **`python main.py` langsung jalan
tanpa instalasi apa pun**. Kalau Anda ingin `import nids_tft_if` dari skrip
lain di luar folder ini (mis. notebook eksplorasi terpisah), jalankan
`pip install -e .` sekali di root proyek — ini memakai `pyproject.toml` yang
sudah disediakan.

## Cara menjalankan

```bash
pip install -r requirements.txt

# 1. Taruh 4 file asli di data/raw/:
#    UNSW-NB15_1.csv, UNSW-NB15_2.csv, UNSW-NB15_3.csv, UNSW-NB15_4.csv

# 2. Jalankan pipeline penuh (tanpa tuning, pakai default src/nids_tft_if/config.py)
python main.py

# 3. Dengan hyperparameter tuning (Optuna, 20 trial default)
python main.py --tune --n-trials 20

# Opsi lain:
python main.py --epochs 50 --raw-dir /path/lain --quiet
```

Output: `checkpoints/tft_base.pt`, `checkpoints/tft_if.pt`,
`logs/results_summary.json` (berisi metrik lengkap, perbandingan TFT-base
vs TFT-IF, ranking feature importance, dan peringkat pentingnya `if_score`).

## Apa yang sudah diverifikasi (smoke test dengan data sintetis)

- `tft_model.py`: forward + backward pass sukses, semua parameter menerima
  gradien, causal mask terbukti bekerja (posisi awal tidak "mengintip" masa
  depan), attention weight berjumlah 1 per baris **hanya dalam mode
  `model.eval()`** (dalam mode `train()` dropout ikut diterapkan ke attention
  weight — ini perilaku default PyTorch, sudah ditangani dengan benar di
  `evaluate.py` dan `interpret.py` yang selalu memanggil `.eval()` dulu).
- Pipeline data penuh (`data_loading` → `preprocessing` → `isolation_forest_module`
  → `sliding_window`): bentuk output `X: (N, W, F)`, `Y: (N, T)` sudah sesuai
  ekspektasi, null/inf/duplikat sintetis berhasil dibersihkan.
- `main.py` end-to-end: jalur normal maupun jalur `--tune` (Optuna, 2 trial)
  sukses tanpa error, `results_summary.json` konsisten.
- Semua ini diuji dengan `epochs=2–3` dan dataset sintetis ~12.000 baris —
  **angka metrik pada uji ini TIDAK BERARTI APA-APA** (data acak, bukan pola
  serangan sungguhan). Ini murni bukti bahwa pipa tersambung, bukan bukti
  performa model.

## Yang PERLU Anda lakukan sebelum eksperimen sungguhan

1. **Cocokkan `src/nids_tft_if/config.py` → `DataConfig.columns`** dengan header CSV UNSW-NB15
   Anda yang sebenarnya. Saya asumsikan 49 kolom + `source_file` dengan nama
   huruf kecil (`srcip`, `sport`, ..., `attack_cat`, `label`). Jika file asli
   Anda tanpa header atau nama kolomnya berbeda kapitalisasi/ejaan
   (mis. `Stime` vs `stime`), sesuaikan `DATA.columns` atau tambahkan mapping
   di `data_loading.py`.
2. **Cek ulang `id_cols_to_drop`** (`srcip, sport, dstip, dsport`) — ini saya
   drop by default karena kardinalitas tinggi (risiko identity leak, bukan
   pola serangan). Proposal Anda tidak eksplisit menyebutkan ini didrop atau
   tidak — putuskan dan dokumentasikan konsisten di Bab 3/4 skripsi.
3. **Jalankan dulu dengan subset kecil** (mis. potong 50–100 ribu baris per
   file) sebelum full run 2,5 juta baris, terutama sebelum `--tune`, karena
   Optuna akan mengulang training penuh N kali (`n_trials`).
4. **Sesuaikan `TRAIN.batch_size` dan `MODEL.hidden_size`** dengan kapasitas
   RTX 3060 12GB Anda kalau nanti OOM — turunkan `batch_size` lebih dulu,
   baru `hidden_size`.
5. **`W` dan `T`** di `src/nids_tft_if/config.py` (default 10 dan 3) masih placeholder —
   proposal Anda menyebutkan ini ditentukan lewat eksperimen ablasi
   (Bagian 6.g.v). Jalankan beberapa kombinasi dan bandingkan lewat
   `results_summary.json`.
6. **Class weighting** sudah aktif secara default (`use_class_weighting=True`
   di `train_model`). Kalau nanti ingin membandingkan dengan tanpa balancing
   sebagai baseline (proposal Bagian 6.e.i), panggil `train_model(...,
   use_class_weighting=False)` secara terpisah.
7. Bagian oversampling/undersampling sebagai pembanding (Bagian 6.e.iii) —
   **belum diimplementasikan** di kode ini. Kalau dibutuhkan, beri tahu saya,
   akan saya tambahkan sebagai modul terpisah agar tidak mengganggu
   `class weighting` yang sudah jadi strategi utama.
8. Hapus atau abaikan folder `tests/` saat submit kode final skripsi (isinya
   generator data dummy, bukan bagian dari eksperimen).

## Rujukan tambahan untuk implementasi kode

File `references_addendum.md` (dibuat terpisah, lihat pesan sebelumnya) berisi
8 rujukan tambahan (Optuna, PyTorch, scikit-learn, GLU, ELU, LayerNorm, Adam)
beserta pemetaan eksplisit ke komponen kode mana yang memakainya — siap
ditempel ke daftar pustaka proposal Anda dan disebut di Bab Metodologi.

## Keputusan desain yang belum eksplisit di proposal (perlu Anda konfirmasi)

- Decoder TFT memakai **learned positional query tokens** (bukan
  known-future-inputs, karena memang tidak tersedia secara alami dalam
  konteks ini — sudah konsisten dengan penjelasan Bagian 5.4.2 proposal Anda).
- Tidak ada static covariates — proposal juga tidak menyebutkan fitur statik
  apa pun, jadi `static_enrichment_grn` dijalankan tanpa context vector `c`.
- FAR dihitung persis sesuai formula agregasi Normal-vs-Attack di Bagian 5.8.4
  (Generic/Exploits/Fuzzers digabung sebagai kelas serangan untuk FP/TN).
