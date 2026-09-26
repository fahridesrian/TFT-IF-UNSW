# spec.md — Spesifikasi Sistem TFT-IF untuk Deteksi Intrusi (UNSW-NB15)

> **Status**: spesifikasi *as-built* — mendokumentasikan sistem persis seperti
> diimplementasikan di `src/` (terverifikasi dengan membaca seluruh modul pada
> 2026-09-27), dipetakan ke proposal (Bagian 5–6) dan paper sumber.
> **Sumber**: `proposal_summary.md`, `model_papers/summaries/*` (11 dokumen),
> `handoff.md`, dan pembacaan langsung `src/*.py`.
> **Fungsi**: acuan tunggal untuk (a) melanjutkan/memodifikasi kode, (b) menulis
> Bab Metodologi & Hasil skripsi secara konsisten dengan kode, (c) audit
> sebelum eksperimen data asli.

---

## 1. Ringkasan Sistem & Alur End-to-End

NIDS berbasis deret waktu: rekaman lalu lintas UNSW-NB15 diurutkan kronologis,
dibentuk jendela geser `W → T`, dan diklasifikasikan menjadi 4 kelas
(Normal, Generic, Exploits, Fuzzers) oleh TFT dari nol. Isolation Forest dilatih
**hanya pada kelas Normal data latih** dan skor anomali kontinunya menjadi **fitur
tambahan ke-(F+1)**. Dua model dilatih dan dibandingkan:

- **TFT-base**: 41 fitur dasar (tanpa skor anomali) — kondisi kontrol.
- **TFT-IF**: 42 fitur (41 + `if_score`) — hipotesis: performa lebih baik,
  terutama mengenali pola serangan tidak normal.

```
data_loading → preprocessing (filter→clean→sort→split→encode+scale)
             → isolation_forest_module (fit Normal-train → skor → fitur +1)
             → sliding_window (per file, per split)  [dua versi: tanpa & dengan if_score]
             → [opsional: Optuna tuning pada TFT-IF]
             → train TFT-base & TFT-IF (paralel, hyperparameter identik)
             → evaluate pada test (flatten N·T) → interpret (VSN + attention)
             → logs/results_summary.json + checkpoints/*.pt
```

## 2. Data & Praproses (`data_loading.py`, `preprocessing.py`)

### 2.1 Sumber & kolom
- 4 file: `data/raw/UNSW-NB15_{1..4}.csv`. Loader mencoba baca dengan header;
  bila nama kolom tak cocok `config.DATA.columns`, fallback `header=None` dengan
  49 nama kolom urutan config (`srcip … label`) + kolom `source_file` (nama file asal).
- Target: `attack_cat` (multikelas). `label` (biner) **tidak dipakai**.
  `stime`/`ltime` **hanya untuk pengurutan**.

### 2.2 Urutan tahap (sesuai `run_preprocessing`)
1. **Filter kelas** (`filter_classes`): trim whitespace; `attack_cat` kosong/nan →
   `"Normal"`; pemetaan case-insensitive ke 4 kelas kanonik; buang baris di luar 4 kelas.
2. **Cleaning** (`clean_data`): paksa numerik semua kolom non-kategorikal/target
   (`pd.to_numeric`, string liar seperti hex port `0x000c`/spasi → NaN);
   ±inf → NaN; **imputasi median per kolom**; kategorikal kosong → `"unknown"`;
   `drop_duplicates()` penuh.
3. **Sort kronologis**: `sort_values([source_file, stime])` — kronologis *dalam* tiap
   file; urutan antar file = urutan file 1→4.
4. **Split kronologis per file** (`chronological_split`): dalam tiap `source_file`,
   70% awal → train, 15% berikut → val, 15% akhir → test; **tanpa shuffle**.
   Konsekuensi: keempat file terwakili di ketiga split; tidak ada sekuens lintas waktu.
5. **Encoding & scaling** (`encode_and_scale`): drop `srcip, sport, dstip, dsport`
   (4) + `stime, ltime` (2) + `label` (1); `source_file` **dipertahankan** di dataframe
   (dikeluarkan dari `feature_cols`) untuk kebutuhan batas sliding window.
   - `LabelEncoder` per kategorikal (`proto, state, service`) — **fit hanya train**;
     kategori unseen di val/test → dipetakan ke `classes_[0]` (fallback terdokumentasi).
   - Target: `LabelEncoder` atas urutan `classes_used` → **Normal=0, Generic=1,
     Exploits=2, Fuzzers=3**.
   - `MinMaxScaler` fit **hanya train**, transform ke ketiga split.
- **`feature_cols` = 41 fitur** (49 − 2 target − 4 id − 2 waktu); TFT-IF: +`if_score` = 42.

### 2.3 ⚠️ Catatan anti-leakage (deviasi halus dari proposal 6.d.ii)
Imputasi median dan `drop_duplicates` dilakukan **sebelum split**, pada seluruh
dataset hasil filter — jadi statistik imputasi "melihat" val/test (walau median
sangat robust dan dampaknya kecil). Proposal menulis parameter praproses hanya
dari train. **Dua opsi**: (a) dokumentasikan sebagai keputusan sadar di skripsi
(dedup pre-split justru mengurangi risiko duplikat lintas split), atau (b) pindahkan
imputasi median ke tahap fit-on-train agar konsisten literal dengan proposal.

## 3. Isolation Forest (`isolation_forest_module.py`)

- **Fit**: `IsolationForest(n_estimators=100, max_samples="auto"[=256], contamination="auto",
  random_state=42, n_jobs=-1)` pada **subset `attack_cat == Normal` dari train**,
  atas 41 fitur ter-scale. Justifikasi normal-only: Liu et al. 2008 §5.4
  (lihat `liu2008_isolation_forest_summary.md` §2).
- **Skor** (`score_and_attach`): `raw = iso.decision_function(X)` (besar = normal);
  `anomaly = −raw` (besar = anomali); dinormalisasi min-max ke [0,1] memakai
  **min/max dari train** (dipakai ulang untuk val/test), hasil di-clip [0, 1];
  ditambahkan sebagai kolom `if_score`.
- ⚠️ **Presisi definisi untuk skripsi**: `decision_function` sklearn **bukan** persis
  `s(x,n) = 2^{−E[h(x)]/c(n)}` (Eq. 2 Liu et al./proposal), melainkan transformasi
  affin monoton dari `E[h(x)]` (dengan offset `contamination`), lalu dinormalisasi.
  **Urutan (ranking) anomali identik** dengan `s(x,n)`; yang berbeda hanya skala.
  Opsi: (a) tulis di skripsi "skor monoton terhadap derajat anomali, dinormalisasi
  min-max dengan statistik data latih", atau (b) ganti ke `iso.score_samples` dan
  hitung `s(x,n)` eksplisit bila ingin setia ke Eq. 2.

## 4. Sliding Window (`sliding_window.py`)

- `{X[start : start+W]} → Y[start+W : start+W+T]`, stride 1.
- Dibentuk **terpisah per split dan per `source_file`** — tidak ada sekuens yang
  melintasi batas file CSV maupun batas split (proposal 6.g.iii–iv).
- Segmen dengan `n < W + T` tidak menghasilkan window (di-skip).
- Output: `X: (N, W, F)`, `Y: (N, T)` (label kelas per titik target).
- Dibangun dua kali dari dataframe yang sama: tanpa `if_score` (TFT-base) dan dengan
  `if_score` (TFT-IF) — sehingga pasangan window keduanya identik.

## 5. Arsitektur Model (`tft_model.py`) — TFT klasifikasi, mengikuti Lim et al. 2021

Input `(B, W, F)` → output **logits** `(B, T, C=4)` (softmax implisit di loss).
Pipeline komponen dan bentuk tensor:

| # | Komponen | Spesifikasi | Fidelity ke paper |
|---|---|---|---|
| 1 | **GLU** | `Linear(F_in → 2·F_out)` → chunk → `a ⊙ σ(b)` — ekuivalen dua proyeksi W₄/W₅ Eq. 5 | ✓ Eq. 5 |
| 2 | **GRN** | `η₂ = W₁a (+ W_c c)`, `η₁ = dropout(W₂ η₂)`, `GLU(η₁)`, `LayerNorm(skip(a) + ·)`; `skip` = Linear bila dimensi beda; tanpa context → `c=None` | ✓ Eq. 2–4; **dropout pada η₁ tepat sesuai §4.1** (terverifikasi) |
| 3 | **VSN** | per fitur `Linear(1→h)` (semua fitur, termasuk kategorikal hasil LabelEncoder) → concat `f·h` → `GRN(→f)` → softmax → bobot `v`; per fitur `GRN(h→h)`; gabungan tertimbang; bobot `v` disimpan untuk interpretasi | ✓ struktur Eq. 6–8; ⚠️ deviasi representasi (lihat §11-D1) |
| 4 | **LSTM encoder** | `nn.LSTM(h→h, 1 layer)` atas output VSN, `(B,W,h)` | ✓ §4.5.1 (seq2seq locality) |
| 5 | **Decoder** | `T` **learned positional query tokens** `(T,h)` (init N(0, 0.02²)) → `nn.LSTM(h→h)` diinisialisasi dengan state akhir encoder `(h_n, c_n)` | Deviasi sadar: paper memasukkan known-future inputs; tidak tersedia di konteks ini. **Didukung paper §6.6** (opsi positional encoding tanpa local processing) |
| 6 | **Gated skip post-LSTM** (Eq. 17) | concat `[enc_out; dec_out]` → GLU → `LayerNorm([vsn_out; dec_in] + GLU(...))` → `(B, W+T, h)` | ✓ Eq. 17 (padanan) |
| 7 | **Static enrichment** | `GRN(h→h)` tanpa context (dipanggil `c=None`) | ⚠️ Eq. 18 tanpa `c_e` — konsisten karena tidak ada static covariates |
| 8 | **Self-attention** | `nn.MultiheadAttention(h, heads=4, dropout, batch_first)` atas `W+T` posisi, **causal mask** `triu(−inf, diagonal=1)` (posisi target hanya melihat ke belakang); `average_attn_weights=True`; bobot attention disimpan | ⚠️ paper memakai varian *interpretable* (value weights di-share antar head + rata-rata attention, Eq. 14–16); kode memakai MHA standar — rata-rata attention diekstrak sebagai aproksimasi (lihat §11-D2) |
| 9 | **Gated skip post-attention** (Eq. 20) | GLU → `LayerNorm(enriched + ·)` | ✓ Eq. 20 |
| 10 | **Position-wise FF** (Eq. 21–22) | `GRN(h→h)` → GLU → `LayerNorm(gated_attn + ·)` | ⚠️ Eq. 22 paper me-skip kembali ke output seq2seq (φ̃); kode me-skip ke output post-attention (`gated_attn`) — jalur residual lebih pendek satu blok (lihat §11-D3) |
| 11 | **Classification head** | `Linear(h→4)` pada **posisi target saja** `final[:, W:, :]` → logits `(B,T,4)` | Adaptasi inti: menggantikan quantile output (Eq. 23) + quantile loss (Eq. 24–25) |

Parameter `d_V = d_attn = h / attn_heads` (konstrain `h % heads == 0`, dijaga tuner).

## 6. Training (`train.py`)

- **Loss**: `CrossEntropyLoss(weight=class_weights)` di atas **flatten penuh**:
  logits `(B·T, C)` vs target `(B·T,)` — setiap titik waktu target berkontribusi sama.
- **Class weighting** (default aktif): `sklearn compute_class_weight("balanced")`
  dihitung pada target train ter-flatten (`n_samples / (n_classes · n_c)`); kelas tak
  hadap diberi bobot 1.0. Bisa dimatikan (`use_class_weighting=False`) untuk baseline
  tanpa balancing (proposal 6.e.i).
- **Optimizer**: Adam, `lr=1e-3`, `weight_decay=1e-5` (default Kingma & Ba: β₁=0.9,
  β₂=0.999, ε=1e-8).
- **Scheduler**: `ReduceLROnPlateau(mode="min", factor=0.5, patience=2)` pada
  `val_loss` per epoch.
- **Early stopping**: monitor `val_loss`, `min_delta=1e-5`, `patience=5`; snapshot
  state terbaik ke CPU, di-load di akhir. `epochs=30` maksimum.
- **AMP**: `torch.amp.autocast("cuda")` + `GradScaler` — aktif hanya di CUDA;
  scaler dibuat **sekali di luar loop epoch**. Prediksi/eval memakai float32 penuh.
- **Loader**: batch 1024; shuffle **hanya train**; `num_workers=6`, `pin_memory=True`;
  `cudnn.benchmark=True`; seed global 42.
- Kedua model (TFT-base & TFT-IF) dilatih dengan **hyperparameter identik**
  (hasil tuning TFT-IF dipakai bersama) — perbandingan adil; satu-satunya beda
  adalah jumlah fitur input.

## 7. Hyperparameter Tuning (`hyperparameter_tuning.py`, opsional `--tune`)

- Optuna TPE (sampler default), `direction="maximize"`, `n_trials=20`.
- **Objektif**: F1-macro pada **data validasi** (bukan akurasi) — sesuai prioritas
  metrik pada data tidak seimbang; data uji di luar proses tuning.
- **Ruang pencarian**: `hidden_size ∈ {32, 64, 128}`; `dropout ∈ [0.0, 0.4]`;
  `attn_heads ∈ {2,4,8}` dibatasi ke pembagi `hidden_size` (param `attn_heads_{h}`);
  `learning_rate ∈ [1e-4, 5e-3]` (log).
- Selama tuning: epochs dipangkas ke `min(30, 15)`; progress bar dimatikan.
- Tidak memakai pruning ASHA (tiap trial = training penuh) — mahal, jalankan sadar.

## 8. Evaluasi (`evaluate.py`) — pada **data uji**

- Prediksi: `argmax` logits → `(N, T)`; seluruh metrik dihitung pada **pasangan
  ter-flatten `(N·T)`** — agregasi merata atas seluruh horizon T (jawaban untuk
  pertanyaan agregasi multi-horizon; opsi per-horizon bisa jadi pengembangan).
- Metrik: `accuracy`; `precision/recall/F1` **macro** & **weighted** (`zero_division=0`);
  **FAR**; `confusion_matrix` 4×4 (urutan kelas: Normal, Generic, Exploits, Fuzzers).
- **FAR** (definisi proposal 5.8.4, agregasi Normal-vs-Attack, persis):
  `FAR = FP/(FP+TN)` dengan `TN` = Normal→Normal dan `FP` = Normal→(Generic|Exploits|
  Fuzzers). Kesalahan antar kelas serangan **tidak** masuk FAR.

## 9. Interpretabilitas (`interpret.py`) — proposal 6.j

- **Feature importance**: bobot VSN `v` dirata-rata atas `(N, W)` → ranking fitur
  (bila TFT-IF, `if_score` ikut diranking → `summarize_if_contribution` mengambil
  rank & skornya). ⚠️ Paper §7.1 melaporkan persentil 10/50/90; kode memakai **mean
  saja** — bisa diperkaya nanti bila ingin identik dengan paper.
- **Attention**: matriks attention dirata-rata atas batch → `(W+T, W+T)` untuk analisis
  pola temporal (proposal 6.j.iii/v). **Wajib `model.eval()`** — di mode train, dropout
  membuat baris attention tidak berjumlah 1 (terverifikasi smoke test).
- Sampel: 256 window pertama dari test TFT-IF (`main.py`).
- Retrain dengan fitur terpilih VSN (6.j.ii) — **belum diimplementasikan** (sifat pendukung).

## 10. Konfigurasi Default (`src/config.py`)

| Grup | Parameter | Nilai |
|---|---|---|
| DATA | csv_files, columns, id_cols_to_drop | 4 file; 49 kolom; `srcip, sport, dstip, dsport` |
| DATA | categorical / time / target / label | `proto, state, service` / `stime, ltime` / `attack_cat` / `label` |
| PREPROC | split 0.70/0.15/0.15; fillna | per-file kronologis; median |
| IF | n_estimators, max_samples, contamination, random_state | 100; "auto"(256); "auto"; 42 |
| WINDOW | W, T, stride | **10, 3 (placeholder — wajib ablasi)**; 1 |
| MODEL | hidden_size, lstm_layers, attn_heads, dropout, num_classes | 64; 1; 4; 0.1; 4 |
| TRAIN | batch_size, epochs, lr, weight_decay | 1024; 30; 1e-3; 1e-5 |
| TRAIN | patience, scheduler(patience, factor), seed, num_workers | 5; (2, 0.5); 42; 6 |
| TUNING | n_trials, metric, direction | 20; f1_macro; maximize |

## 11. Deviasi Sadar dari Proposal/Paper (konsolidasi — kutip di skripsi)

| ID | Deviasi | Dasar |
|---|---|---|
| D1 | Kategorikal via `LabelEncoder` → `Linear(1→h)` di VSN; paper memakai *entity embeddings* per variabel kategorikal | Sederhana & konsisten; catat keterbatasan (ordinalitas semu). Opsi perbaikan: `nn.Embedding` untuk proto/state/service |
| D2 | `nn.MultiheadAttention` standar + rata-rata bobot attention; paper Eq. 14–16 memakai value weights di-share antar head + averaging attention | Aproksimasi interpretabilitas; bila ingin setia ke Eq. 16 (dan analisis attention lebih valid), implementasikan varian interpretable — perubahan lokal di satu modul |
| D3 | Skip residual FF block kembali ke output post-attention; paper Eq. 22 kembali ke output seq2seq (melompati 2 blok) | Minor; sebut di metodologi bila dianggap perlu |
| D4 | Tidak ada static covariates & known futures; decoder = learned positional queries berkondisi state encoder | Tidak tersedia alami; didukung Lim 2021 §6.6 |
| D5 | Klasifikasi (CE + class weighting) menggantikan quantile output/loss | Inti adaptasi skripsi |
| D6 | Skor IF = −`decision_function` dinormalisasi min-max (statistik train, clip [0,1]); bukan literal `s(x,n)` Eq. 2 | Ranking identik; lihat §3 opsi |
| D7 | Imputasi median & dedup **sebelum** split | Lihat §2.3 opsi |
| D8 | `attack_cat` kosong → "Normal" | Asumsi: baris tanpa label kategori = trafik normal; **verifikasi vs CSV asli** |
| D9 | Tuning Optuna/TPE, epochs dipangkas 15 saat tuning; paper memakai random search | Peningkatan metodologis |

## 12. Keputusan Terbuka (wajib ditutup sebelum/di awal eksperimen)

1. **41/42 vs 45/46 fitur** — proposal menulis 45/46 (tanpa drop IP/port); kode men-drop
   `srcip, sport, dstip, dsport` → 41/42. Putuskan (rekomendasi: pertahankan drop,
   revisi angka & justifikasi *identity leak* di skripsi).
2. **Ablasi W, T, rasio split** (proposal 6.g.v) — W=10/T=3 masih placeholder; kandidat
   rasio 70:15:15 (kode) vs 80:10:10 (proposal).
3. **Oversampling/undersampling pembanding** (6.e.iii) — belum diimplementasikan;
   putuskan perlu/tidak.
4. **EDA (6.c)** — belum ada artefak; rencanakan notebook terpisah (histogram, boxplot,
   heatmap, bar chart distribusi kelas).
5. **Retrain fitur terpilih VSN** (6.j.ii) — opsional, pendukung.
6. **D7 & D6 & D8** — putuskan dokumentasi vs perubahan kode (lihat §2.3, §3, §11).

## 13. Checklist Sebelum Run Data Asli

1. Taruh 4 CSV di `data/raw/`; jalankan dulu **subset kecil** (50–100 ribu baris/file).
2. Verifikasi header/nama kolom CSV vs `DATA.columns` (kapitalisasi `Stime` dsb.).
3. Verifikasi asumsi D8 (baris `attack_cat` kosong memang `label=0`).
4. Jalankan `python main.py` penuh → cek `logs/results_summary.json` (metrik,
   perbandingan base vs IF, ranking `if_score`).
5. Ablasi W/T (± rasio) via ubah `config.py`, bandingkan lewat results_summary.
6. Jika OOM di RTX 3060 12GB: turunkan `batch_size` dulu, baru `hidden_size`.
7. Simpan seed & config final yang dipakai untuk setiap tabel hasil skripsi.
