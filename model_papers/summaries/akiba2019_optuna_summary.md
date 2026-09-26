# Summary — Akiba et al. (2019), *Optuna: A Next-generation Hyperparameter Optimization Framework*

**File sumber**: `model_papers/3292500.3330701.pdf` (KDD '19)
**Peran dalam proyek**: framework **hyperparameter tuning** di
`src/hyperparameter_tuning.py` (`optuna.create_study`, `trial.suggest_*`),
dipanggil opsional via `python main.py --tune` (proposal Bagian 6.h.vi).

## 1. Konsep paper yang dipakai

- **Define-by-run API** (§2): fungsi objektif menerima objek `trial` dan ruang
  pencarian dibangun dinamis lewat `trial.suggest_*` — persis pola penulisan
  `hyperparameter_tuning.py`.
- **Study & trial** (§2): satu `study` = kumpulan evaluasi objektif (`trial`);
  `create_study(direction="maximize")` karena objektif proyek **memaksimalkan
  F1-macro data validasi** (`TUNING.metric = "f1_macro"`, `direction = "maximize"`,
  `n_trials = 20`).
- **Sampler default TPE** (Tree-structured Parzen estimator, §3.1): proyek memakai
  sampler default Optuna (TPE, independent sampling) tanpa kustomisasi.
- **In-memory storage** (§4): backend default, single process — tanpa distributed
  computing / database eksternal.

## 2. Pemetaan ke kode & keputusan proyek

| Paper | Di proyek |
|---|---|
| `study.optimize(objective, n_trials)` | `run_tuning(..., n_trials)`; tiap trial = training penuh satu konfigurasi pada data window TFT-IF |
| Objektif mengembalikan skor validasi | F1-macro validasi; hyperparameter terbaik (`hidden_size`, `dropout`, `attn_heads`, `learning_rate`) dipakai untuk **kedua** model (TFT-base & TFT-IF) di `main.py` |
| — | **Pruning (ASHA, §3.2) TIDAK dipakai**: tiap trial dijalankan sampai selesai (early stopping biasa di dalam `train.py` tetap aktif); konsekuensinya `--tune` mahal karena mengulang training penuh N kali — catatan `handoff.md` |
| — | Progress bar dimatikan selama tuning agar log Optuna tidak penuh |

## 3. Rasional yang bisa dikutip di skripsi

- Pemilihan konfigurasi hyperparameter dilakukan secara **otomatis dan sistematis**
  dengan framework optimasi black-box berbasis TPE (bukan grid search manual),
  dengan kriteria pemilihan F1-macro pada data validasi — metrik yang prioritas
  pada data tidak seimbang; data uji tetap terpisah dari proses tuning.

## 4. Bagian paper yang TIDAK dipakai (tidak perlu dibuka ulang)

- §3.2: detail algoritme ASHA/pruning (tidak diaktifkan di proyek).
- §4: arsitektur distributed/scalable, SQLite storage, dashboard.
- §5–6: benchmark 56 test-case, eksperimen pruning AlexNet/SVHN, aplikasi
  RocksDB/FFmpeg/HPL.

## Kutipan (APA — sudah terdaftar di `references_addendum.md`)

Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). Optuna:
A next-generation hyperparameter optimization framework. Proceedings of the 25th
ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, 2623–2631.
https://doi.org/10.1145/3292500.3330701
