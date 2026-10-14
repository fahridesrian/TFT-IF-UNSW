# spec.md — Spesifikasi Sistem TFT-IF untuk Deteksi Intrusi (UNSW-NB15)

> **Status**: **v1.2 (2026-10-14)** — v1.1 (keputusan terkunci 2026-09-27, A1–A14,
> lolos smoke test) + revisi Okt 2026: split global (A2), grid (W,T) pada kedua
> model (A3/A6), empat kelas (A15), persistensi (A16), batch 4096 (A17).
> **Revisi A2/A3/A16 belum diterapkan di kode** dan ada **bug pemetaan label**
> (lihat §2.2 butir 6 dan §14); hasil training 13 Okt belum dapat dipakai di skripsi.
> Dokumen ini mendokumentasikan sistem persis seperti di `src/`, dipetakan ke
> proposal (Bagian 5–6) dan paper sumber, **termasuk pembelaan (justifikasi
> skripsi) untuk setiap keputusan**.
> **Sumber**: `proposal_summary.md`, `model_papers/summaries/*` (11 dokumen),
> `handoff.md`, jawaban keputusan A1–A14, dan pembacaan langsung `src/*.py`.
> **Fungsi**: acuan tunggal untuk (a) eksperimen data asli, (b) penulisan Bab
> Metodologi & Hasil secara konsisten dengan kode, (c) pembelaan saat sidang.

---

## 0. Keputusan Terkunci (2026-09-27) + Pembelaan Skripsi

| ID | Keputusan | Ringkasan |
|---|---|---|
| A1 | **41/42 fitur** (drop `srcip, sport, dstip, dsport`) | lihat §0.1 |
| A2 | **Rasio 70:15:15** kronologis **global** (revisi 6 Okt 2026, menggantikan per file) | 80:10:10 opsional sebagai sensitivitas; lihat §0.2 |
| A3 | **Grid W∈{10,25,50} × T∈{1,3}** pada **kedua model** (revisi Okt 2026), terbaik per model menurut F1-macro validasi + perbandingan pada (W,T) sama | anchor: Psychogyios et al. 2024; lihat §0.3 |
| A4 | **Class weighting `balanced`** n/(k·n_k) | strategi imbalance utama |
| A5 | **Early stopping pada val_loss** | F1-macro tetap selektor hyperparameter |
| A6 | **Hyperparameter identik** untuk TFT-base & TFT-IF; **(W,T) dipilih terpisah per model** (revisi Okt 2026) | eksperimen terkontrol; lihat §0.6 |
| A7 | CSV **tanpa header, 49 kolom** | fallback loader sudah benar |
| A8 | **Imputasi median fit-on-train** (setelah split) | konsisten literal proposal 6.d.vii |
| A9 | Kategori unseen → token **"unknown" eksplisit** | bukan fallback ke kelas pertama |
| A10 | **Assertion** `attack_cat` kosong ∧ label=1 → error | asumsi jadi kontrak eksplisit |
| A11 | Skor IF = **literal s(x,n) Eq. 2 Liu 2008**, tanpa min-max | `s = -iso.score_samples` |
| A12 | **Interpretable multi-head attention Eq. 13–16** Lim 2021 | bukan MHA standar |
| A13 | **Entity embeddings** untuk fitur kategorikal di VSN | mengikuti paper §4.2 |
| A14 | Artefak pelaporan: **per-kelas, per-horizon, distribusi split, EDA** | semua tersedia |
| A15 | **Empat kelas = empat kategori dengan rekaman terbanyak** (Normal, Generic, Exploits, Fuzzers); enam lainnya dibuang | lihat §0.9 |
| A16 | **Acuan persistensi** (label rekaman terakhir) sebagai satu baris referensi | lihat §0.10 |
| A17 | **Batch 4096** (naik dari 1024, 13 Okt 2026) | efisiensi: <1 menit/epoch vs 4–5 menit; lihat §6 |

### 0.1 A1 — 41/42 fitur (drop IP & port)
**Keputusan**: dari 49 kolom: −2 target (`attack_cat`, `label`) − 2 waktu
(`stime`, `ltime`) − 4 identitas (`srcip`, `sport`, `dstip`, `dsport`) = **41
fitur**; +`if_score` = **42** untuk TFT-IF. Angka 45/46 di proposal direvisi.

**Pembelaan**:
1. Proposal §5.2 hanya mendefinisikan skema encoding untuk `proto`, `service`,
   `state` — tidak ada skema untuk IP/port; men-drop adalah interpretasi paling
   konsisten terhadap proposal sendiri.
2. `srcip`/`dstip` adalah **pengenal (identifier)**, bukan perilaku: hampir
   unik per host, sehingga model cenderung menghafal identitas. Pada split
   kronologis, host di test berbeda dari train — fitur pengenal justru
   **merusak generalisasi** (identity leak).
3. `sport`/`dsport` pseudo-kategorikal kardinalitas tinggi (terkonfirmasi ada
   nilai hex `0x000c` di data mentah) tanpa urutan alami: sebagai numerik
   memberi sinyal palsu; sebagai kategorikal butuh embedding ratusan ribu item.
4. Framing deret waktu ini menargetkan **pola perilaku lalu lintas** (durasi,
   byte, TTL, flag, hitungan koneksi) yang lengkap pada 41 fitur.
5. Framing kalimat skripsi: *"dari 45 atribut non-target, 4 atribut identitas
   (IP/port) dikecualikan sehingga terdapat 41 fitur (42 dengan skor anomali)."*
6. Cadangan: bila penguji menuntut, jalankan sekali varian 45/46 sebagai
   **cek sensitivitas** (bukan eksperimen utama).

### 0.2 A2 — Rasio split 70:15:15
**Revisi (6 Okt 2026)**: pemotongan dilakukan **sekali secara global** pada
urutan kronologis seluruh data (urut `[stime, source_file]`, stabil), bukan per
berkas. Alasan: pembagian per berkas menempatkan bagian akhir tiap berkas di
data uji sehingga uji bercampur dari beberapa periode dan tidak mencerminkan
"masa depan" sesungguhnya; pembagian global menjaga satu garis waktu dan
mencegah kebocoran antar periode. Konsekuensi: data uji berasal dari periode
paling akhir (berkas ke-4), sehingga distribusi kelas antarsplit berbeda;
laporkan `split_class_distribution`. **Status kode (14 Okt 2026): belum
diterapkan** — `chronological_split` dan `sort_chronologically` masih per
berkas; hasil training 13 Okt memakai pembagian per berkas.

**Pembelaan**: angka yang tertulis di proposal; dengan ±2,54 juta baris, 15%
test ≈ 381 ribu window — jauh di atas ukuran minimum untuk estimasi metrik
stabil. Perbandingan rasio bukan pertanyaan penelitian inti (inti: efek skor
IF), jadi cukup satu rasio; 80:10:10 opsional bila sempat (sensitivitas).

### 0.3 A3 — Grid W dan T (proposal 6.g.v)
**Keputusan (revisi Okt 2026)**: grid **W ∈ {10, 25, 50} × T ∈ {1, 3}**
dijalankan pada **kedua model** (hyperparameter default, epochs penuh).
Selektor = **F1-macro validasi** (`results_val_TFT_*.f1_macro`). Setiap model
mengambil (W,T) terbaiknya sendiri; selain itu dilaporkan perbandingan pada
(W,T) yang sama agar selisih tidak tercampur pengaruh pemilihan. Data uji tidak
dipakai memilih. Hasil disimpan **per kombinasi** (nama berkas memuat model, W,
T). Default sampai grid selesai: W=10, T=3. **Status kode: belum diterapkan**
(hanya override CLI `--window/--horizon`; run 13 Okt memakai W=10, T=3 tanpa
grid). Catatan: T berbeda mengubah tugas prediksi; bandingkan antarmodel hanya
pada T yang sama bila membandingkan metrik secara langsung.

**Anchor literatur**: **Psychogyios et al. 2024** (Future Internet 16(3):73) —
TFT pada UNSW-NB15 yang sama menemukan F1 memuncak di **W=50** dan tidak
bertambah pada W=100/200; horizon lebih panjang membuat tugas lebih mudah
(karena label "kehadiran serangan dalam T ke depan" lebih jarang semuanya
negatif). Grid di atas direplikasi dari tren itu (titik jenuh W=50, titik
kecil W=10/25 untuk melihat kemiringan), dengan T kecil karena skripsi
memakai **label per-timestep** (lebih ketat dari label kehadiran).

**Biaya & RAM**: ±30 menit/run (RTX 3060, perkiraan lama pada batch 1024;
dengan batch 4096 lebih singkat — ukur ulang) → 6 kombinasi × 2 model.
W=50: array train ±15 GB + val/test ±6 GB → butuh ±24–32 GB RAM; bila tidak
cukup, batasi grid ke W∈{10,25} dan nyatakan di skripsi. Perintah:
`python main.py --window 50 --horizon 1 --epochs 30` (dst.).

### 0.4 A4 — Class weighting `balanced`
`w_c = n_samples / (n_classes · n_c)` pada target train ter-flatten.
**Pembelaan**: IR Normal:Fuzzers ≈ 91,5 → tanpa pembobot, loss didominasi
kelas Normal dan kelas minoritas hampir tak dipelajari. Weighting menyetarakan
kontribusi tiap kelas **tanpa menyentuh struktur temporal data** (oversampling/
SMOTE memasukkan sampel sintetis yang bisa merusak urutan waktu — sifat yang
justru inti pemodelan TFT). Didukung literatur imbalance NIDS yang sudah ada
di bibliografi proposal (Altalhan et al. 2025; Bakirarar & Elhan 2023).

### 0.5 A5 — Early stopping pada val_loss (patience=5)
**Pembelaan**: val_loss sinyal kontinu dan halus (perubahan kecil pada
probabilitas terdeteksi), sedangkan F1-macro diskrit dan ber-plateau (hanya
berubah bila prediksi individual berpindah kelas) → stopping berbasis loss
lebih stabil dan lebih awal. Dua tahap terpisah: **kapan berhenti** (val_loss)
vs **konfigurasi mana terbaik** (F1-macro validasi saat tuning) — praktik
standar; `ReduceLROnPlateau` juga bekerja pada loss sehingga satu sinyal
konsisten.

**Referensi**: Prechelt (1998), *Early Stopping — But When?* (LNCS 1524;
ringkasan: `model_papers/summaries/prechelt1998_early_stopping_summary.md`).
Dua dukungan spesifik dari paper:
1. Kriteria kita (berhenti setelah 5 epoch tanpa perbaikan > `min_delta`, dengan
   retensi bobot terbaik `E_opt`) termasuk **kelas UP** — naiknya error validasi
   pada s strip berturut-turut. Prechelt menemukan UP3–UP6 memberi tradeoff
   terbaik (test error vs waktu training) dan paling robust; `patience=5`
   tepat di rentang itu.
2. Kriteria berbasis magnitudo (kelas GL) **tidak stabil pada data yang
   dipartisi kronologis** (masalah `building` Proben1) — pipeline kita split
   kronologis, sehingga kelas UP/patience memang lebih tepat.
Tradeoff kuantitatif paper: kriteria lambat hanya memperbaiki test error ±4%
dengan biaya waktu ±4× — patience moderat (5) adalah kompromi rasional.

### 0.6 A6 — Hyperparameter identik untuk kedua model
Hyperparameter model (hidden_size, dropout, heads, lr, batch) **identik** pada
kedua model; **(W,T) dipilih terpisah per model** lewat grid (A3). Bila
`--tune` dipakai, tuning dilakukan pada TFT-IF dan hasilnya dipakai bersama;
run 13 Okt tidak memakai `--tune`. **Pembelaan**: eksperimen terkontrol —
satu-satunya perbedaan antar model (selain pilihan (W,T)) adalah kehadiran fitur `if_score`, sehingga
selisih performa dapat **diatribusikan bersih** ke skor IF (bukan tercampur
perbedaan hyperparameter = confounder). Arsitektur identik dengan satu fitur
tambahan tidak mengubah ruang hyperparameter yang optimal secara sistematis;
ini praktik ablation standar.

### 0.7 A7 — Format CSV
Terbentuk dari sampel baris asli: **tanpa header, 49 kolom, urutan persis
`config.DATA.columns`**; ada baris `attack_cat` kosong dengan `label=0`.
Loader `data_loading.py` (fallback `header=None, names=DATA.columns`) sudah
benar; tidak ada perubahan.

### 0.8 A8 — Imputasi median fit-on-train
Median per kolom numerik **dihitung dari train saja** (`impute_missing`,
dipanggil setelah split) dan diterapkan ke train/val/test.
**Pembelaan**: konsisten literal dengan proposal 6.d.vii (semua parameter
praproses di-fit dari train) → tidak ada leakage statistik; median robust
terhadap outlier, cocok untuk trafik jaringan yang heavy-tailed.
`drop_duplicates` **tetap sebelum split** — disengaja: duplikat identik yang
melintasi batas split justru menyebabkan overlap sampel antar split; dedup
global mencegahnya (dokumentasikan di skripsi sebagai proteksi leakage).

### 0.9 A9 — Token "unknown" eksplisit
`LabelEncoder` di-fit pada kategori train **+ token "unknown"**; nilai
val/test di luar kategori train dipetakan ke "unknown" (indeks embedding
sendiri). **Pembelaan**: fallback lama (kelas pertama) menciptakan asumsi
semu bahwa unseen ≡ kategori arbitrer; token khusus membuat model belajar
perilaku "kategori tak dikenal" secara eksplisit — penting pada split
kronologis di mana kategori baru memang bisa muncul (distribusi bergeser).

### 0.10 A10 — Assertion konsistensi `attack_cat`
`filter_classes` **berhenti dengan error** bila ada baris `attack_cat` kosong
tetapi `label=1`. **Pembelaan**: asumsi "kosong = Normal" adalah pola dataset
 UNSW-NB15 (baris tanpa kategori = trafik normal, label=0 — terkonfirmasi di
sampel asli); assertion mengubah asumsi diam-diam menjadi kontrak eksplisit:
bila data riil melanggar, lebih berhenti daripada salah label.

### 0.11 A11 — Skor IF literal s(x,n)
`if_score = -iso.score_samples(X) = 2^{−E[h(x)]/c(n)}` — **persis Persamaan 2
proposal / Eq. 2 Liu et al. 2008**, tanpa normalisasi min-max.
**Pembelaan**: (i) `score_samples` sklearn mengimplementasikan Algorithm 3
Liu et al. (termasuk koreksi c(Size) pada external node), sehingga negasi-nya
**adalah** s(x,n) — bukan transformasi monoton seperti `decision_function`;
(ii) s∈(0,1) punya interpretasi baku: →0 normal, ≈0.5 ambigu, →1 anomali
(Liu et al. 2008 Fig. 2); (iii) rumus yang dipresentasikan di skripsi identik
dengan angka yang mengalir ke model — tanpa langkah normalisasi tambahan yang
harus dibela tanpa referensi; (iv) normalisasi min-max lama membuat skor
bergantung pada min/max sampel train (tidak lagi murni fungsi s).

### 0.12 A12 — Interpretable multi-head attention (Eq. 13–16)
Kelas kustom `InterpretableMultiHeadAttention` menggantikan
`nn.MultiheadAttention`:
- per head: `W_q^h, W_k^h : h → d_attn = h/H` (proyeksi Q,K);
- **W_V di-share antar head** (`h → d_attn`), Eq. 14;
- mask kausal aditif (0/−inf) sebelum softmax; dropout pada probabilitas
  attention per-head;
- `Ã(Q,K) = (1/H) Σ_h softmax(QW_q^h (KW_k^h)^T/√d_attn)`, Eq. 15;
- output = `Ã · V W_V · W_H` (Eq. 13–16; tepat karena rata-rata linear).

**Pembelaan**: proposal Gambar 1 eksplisit menulis *"Masked Interpretable
Multi-Head Attention"* → kesetiaan ganda ke proposal & paper; analisis
attention (6.j.iii) kini **valid secara definisi** — matriks yang diekstrak
adalah matriks yang benar-benar dipakai model, bukan rata-rata aproksimasi
dari MHA standar. Catatan: **checkpoint lama tidak valid** → retrain wajib.

### 0.13 A13 — Entity embeddings untuk kategorikal (VSN)
Setiap fitur kategorikal (`proto`, `state`, `service`, termasuk token
"unknown") dipetakan `nn.Embedding(kardinalitas, h)`; fitur kontinu tetap
`Linear(1→h)`. **Pembelaan**: paper §4.2 memang memakai entity-style
embedding untuk input kategorikal → kesetiaan ke paper sekaligus **menghapus
ordinalitas semu** (LabelEncoder membedakan kategori hanya berdasar urutan
alfabet); embedding dipelajari end-to-end; kardinalitas kecil (±5/5/6) →
tambahan parameter nyaris nol. Kategorikal **tidak di-scale** — kode integer
mentah dipakai langsung sebagai indeks embedding (memperbaiki risiko bug
round-off float32 bila ikut di-MinMaxScale). Implikasi: `if_score` (fitur
kontinu) tetap unscaled karena s(x,n) sudah berada di (0,1).

### 0.14 A14 — Artefak pelaporan (semua tersedia, teruji)
1. **Per kelas**: precision/recall/F1/support one-vs-rest tiap kelas
   (`results_*.per_class`) — bahan tabel pembahasan per kategori.
2. **Per horizon**: accuracy, F1-macro, FAR untuk tiap t+1..t+T
   (`results_*.per_horizon`) — bahan analisis degradasi horizon.
3. **Distribusi kelas per split** (print + `split_class_distribution` di JSON)
   — transparansi imbalance sejak awal.
4. **`eda.py`** (proposal 6.c): bar chart distribusi kelas + IR, laporan
   null/inf/duplikat, deskriptif keseluruhan & per kelas, histogram+density,
   boxplot per kelas, heatmap korelasi → `logs/eda/`.
   Jalankan: `python eda.py` (data asli; `--sample` mengatur ukuran sampel plot).

### 0.9 A15 — Mengapa empat kelas
Dataset sangat tidak seimbang. Untuk mengurangi ketidakseimbangan dipilih
**empat kategori dengan jumlah rekaman terbanyak**: Normal 2.218.764, Generic
215.481, Exploits 44.525, Fuzzers 24.246 (CSV asli); enam kategori lain (DoS,
Reconnaissance, Analysis, Backdoors, Shellcode, Worms) dibuang (37.031 rekaman,
sebelum deduplikasi). Imbalance ratio ≈ 91,5 (Normal:Fuzzers). Tidak memakai
ambang persentase tertentu. Angka final menunggu keluaran `main.py` setelah
deduplikasi (jumlah Generic turun sangat besar karena duplikat).

### 0.10 A16 — Acuan persistensi
Prediksi "kelas rekaman berikutnya = kelas rekaman terakhir" memakai **label
sebenarnya** yang tidak dimiliki model (model hanya melihat fitur lalu
lintas), sehingga hanya **satu baris acuan** pada tabel hasil, bukan pesaing.
Hitung pada jendela yang sama dengan data uji (setelah deduplikasi). Tambahkan
juga acuan "selalu Normal". **Status kode: belum ada di `evaluate.py`.**
Hitungan awal (pra-deduplikasi): pembagian global acc 92,80% / F1 macro 0,5805;
per berkas acc 94,64% / F1 macro 0,6170 — harus dihitung ulang.

## 1. Ringkasan Sistem & Alur End-to-End

NIDS berbasis deret waktu: rekaman lalu lintas UNSW-NB15 diurutkan kronologis,
dibentuk jendela geser `W → T`, dan diklasifikasikan menjadi 4 kelas
(Normal, Generic, Exploits, Fuzzers) oleh TFT dari nol. Isolation Forest dilatih
**hanya pada kelas Normal data latih** dan skor anomali literal `s(x,n)`
menjadi **fitur tambahan ke-(F+1)**. Dua model dilatih dan dibandingkan:

- **TFT-base**: 41 fitur dasar (tanpa skor anomali) — kondisi kontrol.
- **TFT-IF**: 42 fitur (41 + `if_score`) — hipotesis: performa lebih baik,
  terutama mengenali pola serangan tidak normal.

```
data_loading (headerless 49 kolom, fallback names=config)
  → eda.py (6.c, terpisah, → logs/eda/)
  → preprocessing: filter(+assertion)→clean→sort→split 70:15:15 (global, A2*)
      → impute (median, fit train) → encode (+unknown) → scale kontinu
  → isolation_forest_module: fit Normal-train → s(x,n) literal → fitur +1
  → sliding_window (per segmen, per split)  [dua versi: tanpa & dengan if_score]
  → [opsional: Optuna tuning pada TFT-IF] / grid (W,T) pada kedua model (A3*)
  → train TFT-base & TFT-IF (hyperparameter identik)
  → evaluate test + val (flatten N·T, per kelas, per horizon) + acuan persistensi (A16*)
  → interpret (VSN + attention Ã) → logs/results_summary.json + checkpoints
```
(* = diputuskan, belum diterapkan di kode per 14 Okt 2026.)

## 2. Data & Praproses (`data_loading.py`, `preprocessing.py`)

### 2.1 Sumber & kolom
- 4 file: `data/raw/UNSW-NB15_{1..4}.csv`, **tanpa header** (A7 — terkonfirmasi
  dari baris asli). Loader (commit 1f16f37) selalu membaca `header=None`,
  melewati baris rusak (`on_bad_lines="skip"`), membuang baris terpotong
  (label NaN), mendeteksi header secara eksplisit bila ada, memberi 49 nama
  kolom urutan `config.DATA.columns` + kolom `source_file`, dan mencetak
  jumlah baris yang dilewati per berkas. Dummy test (`tests/make_dummy_data.py`)
  menulis header, sehingga deteksi header tetap perlu.
- Target: `attack_cat` (multikelas). `label` (biner) tidak masuk fitur.
  `stime`/`ltime` **hanya untuk pengurutan**.

### 2.2 Urutan tahap (sesuai `run_preprocessing`)
1. **Filter kelas** (`filter_classes`): trim whitespace; **assertion A10**
   (attack_cat kosong ∧ label=1 → `ValueError`); `attack_cat` kosong/nan →
   `"Normal"`; pemetaan case-insensitive ke 4 kelas kanonik; buang baris di
   luar 4 kelas.
2. **Cleaning** (`clean_data`): paksa numerik kolom non-kategorikal/target
   (`pd.to_numeric`; hex port/spasi → NaN); ±inf → NaN; kategorikal kosong →
   `"unknown"`; `drop_duplicates()` penuh (**sengaja pre-split** — lihat §0.8).
   NaN numerik **tidak diisi di sini**.
3. **Sort kronologis**: target (A2): `sort_values([stime, source_file])` stabil,
   satu urutan global. Kode saat ini: `[source_file, stime]`.
4. **Split kronologis** (`chronological_split`): target (A2): 70/15/15 awal-
   tengah-akhir pada urutan **global**, **tanpa shuffle**. Kode saat ini masih
   per `source_file` (belum diubah).
5. **Imputasi** (`impute_missing`): median **fit-on-train** (A8).
6. **Encoding & scaling** (`encode_and_scale`): drop 4 id + 2 waktu + `label`;
   `source_file` dipertahankan di dataframe (di luar `feature_cols`).
   - `LabelEncoder` per kategorikal **fit train + "unknown"** (A9); kode
     integer mentah dipertahankan (tanpa scaler) — indeks embedding (A13);
     `categorical_cardinalities` & `categorical_feature_indices` diekspos di
     `PreprocArtifacts`.
   - Target: **pemetaan yang dimaksud** Normal=0, Generic=1, Exploits=2,
     Fuzzers=3 (urutan `classes_used`). **BUG di kode (14 Okt 2026)**:
     `LabelEncoder.fit(classes_used)` mengurutkan alfabet → Exploits=0,
     Fuzzers=1, Generic=2, Normal=3, sehingga nama kelas dan FAR pada
     `results_summary.json` bergeser. Perbaikan: set `classes_` eksplisit
     atau kamus nama→indeks, plus assert urutan.
   - `MinMaxScaler` **fit hanya train, hanya fitur kontinu**; `if_score`
     (ditambah setelah tahap ini) tidak di-scale — sudah di (0,1) (A11).
- **`feature_cols` = 41** (49−2 target−4 id−2 waktu); TFT-IF: +`if_score` = 42.

### 2.3 Status anti-leakage (selesai — A8/A10)
Imputasi kini fit-on-train; encoder & scaler fit-on-train; IF fit Normal-train
saja. `drop_duplicates` pre-split dipertahankan dengan argumen proteksi
overlap antar split (§0.8). Tidak ada lagi deviasi leakage yang terbuka.

## 3. Isolation Forest (`isolation_forest_module.py`)

- **Fit**: `IsolationForest(n_estimators=100, max_samples="auto"[=256],
  contamination="auto", random_state=42, n_jobs=-1)` pada **subset Normal
  dari train**, atas 41 fitur. Justifikasi normal-only: Liu et al. 2008 §5.4.
- **Skor** (`score_and_attach`): `if_score = -iso.score_samples(X)` = **literal
  s(x,n) = 2^{−E[h(x)]/c(n)}** (Eq. 2; Algorithm 3 sklearn, A11). s∈(0,1):
  →0 normal, ≈0.5 ambigu, →1 anomali. Tanpa normalisasi/clip.
- Rumus untuk skripsi: s(x,n), c(n)=2H(n−1)−2(n−1)/n, H(i)≈ln(i)+0.5772156649
  (Euler–Mascheroni) — identik dengan angka yang mengalir ke model.

## 4. Sliding Window (`sliding_window.py`) + rencana ablasi (A3)

- `{X[start : start+W]} → Y[start+W : start+W+T]`, stride 1; terpisah per
  split dan per segmen (saat ini per `source_file`; pada split global, tentukan
  perlakuan jeda waktu antar berkas, mis. jeda 624 jam di berkas ke-2);
  segmen `n < W+T` di-skip. Implementasi memakai `sliding_window_view`
  (commit 1f16f37).
- Output: `X: (N, W, F)` float32, `Y: (N, T)` int64; dibangun dua kali (tanpa/dengan
  `if_score`) — pasangan window identik.
- Override CLI: `python main.py --window W --horizon T` (mutasi singleton
  `WINDOW` — tidak perlu edit config).
- **Grid (A3, revisi)**: W∈{10,25,50}×T∈{1,3} pada **kedua model**; selektor =
  `results_val_TFT_*.f1_macro`; terbaik per model + perbandingan pada (W,T)
  sama; hasil per kombinasi; anchor & catatan RAM di §0.3.

## 5. Arsitektur Model (`tft_model.py`) — mengikuti Lim et al. 2021

Input `(B, W, F)` → output **logits** `(B, T, C=4)` (softmax implisit di loss).

| # | Komponen | Spesifikasi | Fidelity ke paper |
|---|---|---|---|
| 1 | **GLU** | `Linear(F_in→2·F_out)` → chunk → `a ⊙ σ(b)` | ✓ Eq. 5 |
| 2 | **GRN** | `η₂=W₁a (+W_c c)`, `η₁=dropout(W₂ η₂)`, `GLU(η₁)`, `LayerNorm(skip(a)+·)` | ✓ Eq. 2–4; dropout pada η₁ sesuai §4.1 |
| 3 | **VSN** | kategorikal → `nn.Embedding(kardinalitas, h)` **termasuk token "unknown"** (A13); kontinu → `Linear(1→h)`; concat `f·h` → `GRN(→f)` → softmax → bobot `v`; per fitur `GRN(h→h)`; gabungan tertimbang; `v` disimpan | ✓ Eq. 6–8 + entity embeddings §4.2 |
| 4 | **LSTM encoder** | `nn.LSTM(h→h, 1 layer)` | ✓ §4.5.1 |
| 5 | **Decoder** | `T` learned positional query tokens `(T,h)` → `nn.LSTM` init dari `(h_n, c_n)` encoder | Deviasi sadar D4; didukung Lim §6.6 |
| 6 | **Gated skip post-LSTM** | GLU → `LayerNorm([vsn_out; dec_in] + GLU([enc_out; dec_out]))` | ✓ Eq. 17 (padanan) |
| 7 | **Static enrichment** | `GRN(h→h)` tanpa context | ✓ (tanpa static covariates) |
| 8 | **Interpretable self-attention** | kustom Eq. 13–16 (A12): per-head `W_q/W_k`→`d_attn=h/H`; **W_V shared**; mask kausal aditif; `Ã`=rata-rata antar head (yang diekstrak untuk interpretasi); out=`Ã·V W_V·W_H` | ✓ Eq. 13–16 persis |
| 9 | **Gated skip post-attention** | GLU → `LayerNorm(enriched + ·)` | ✓ Eq. 20 |
| 10 | **Position-wise FF** | `GRN(h→h)` → GLU → `LayerNorm(gated_attn + ·)` | ⚠️ D3: skip ke post-attention (bukan φ̃ Eq. 22) |
| 11 | **Classification head** | `Linear(h→4)` pada `final[:, W:, :]` → logits `(B,T,4)` | Adaptasi inti D5 (menggantikan quantile) |

Konstrain `h % heads == 0` (dijaga tuner; berlaku untuk
`InterpretableMultiHeadAttention`).

## 6. Training (`train.py`)

- **Loss**: `CrossEntropyLoss(weight=class_weights)` di atas flatten penuh
  `(B·T, C)` vs `(B·T,)` — setiap titik horizon berkontribusi sama.
- **Class weighting** `balanced` (A4): `compute_class_weight` pada train
  ter-flatten; kelas tak hadap bobot 1.0; bisa dimatikan
  (`use_class_weighting=False`).
- **Optimizer**: Adam `lr=1e-3`, `wd=1e-5` (β default Kingma & Ba).
- **Scheduler**: `ReduceLROnPlateau(min, factor=0.5, patience=2)` pada val_loss.
- **Early stopping**: val_loss, `min_delta=1e-5`, patience=5 (A5); snapshot
  state terbaik ke CPU; maksimum 30 epoch.
- **AMP**: autocast + `GradScaler` sekali di luar loop epoch (CUDA saja).
- **Loader**: batch **4096** (A17; semula 1024); shuffle hanya train; workers 6;
  pin_memory; tensor `share_memory_()` agar worker tidak menyalin data (Windows
  spawn); seed 42 global. Langkah pembaruan per epoch ±4× lebih sedikit pada
  `lr` yang sama; patience/jadwal LR tidak diubah.
- Hyperparameter identik kedua model (A6); `categorical_features` diteruskan
  dari `PreprocArtifacts` melalui `train_model` → konstruktor TFT.

## 7. Hyperparameter Tuning (`hyperparameter_tuning.py`, opsional `--tune`)

- Optuna TPE, `maximize` **F1-macro validasi**, 20 trial; epochs dipangkas
  `min(30, 15)` saat tuning; tanpa pruning (trial = training penuh, sadar biaya).
- Ruang: `hidden_size ∈ {32,64,128}`; `dropout ∈ [0.0,0.4]`;
  `attn_heads ∈ {2,4,8}` pembagi `hidden_size`; `lr ∈ [1e-4,5e-3]` log.
- `categorical_features` diteruskan ke tiap trial.

## 8. Evaluasi (`evaluate.py`) — data uji (+ validasi untuk ablasi)

- Prediksi argmax → `(N,T)`; metrik utama pada **flatten (N·T)**.
- Metrik: accuracy; precision/recall/F1 macro & weighted (`zero_division=0`);
  **FAR**; confusion matrix 4×4 (urutan harus sama dengan pemetaan label di §2.2
  butir 6: Normal, Generic, Exploits, Fuzzers). `normal_idx` dicari dari
  `class_names`, jadi FAR benar hanya bila pemetaan label benar. F1 weighted
  masih dihitung tetapi tidak dilaporkan di skripsi.
- **FAR** (proposal 5.8.4): `FP/(FP+TN)` agregasi Normal-vs-Attack
  (FP = Normal→serangan apa pun).
- **Per kelas** (A14): `per_class[kelas] = {precision, recall, f1, support}`
  one-vs-rest.
- **Per horizon** (A14): `per_horizon["t+1".."t+T"] = {accuracy, f1_macro,
  false_alarm_rate}` — degradasi performa per langkah target.
- **Metrik validasi** juga dihitung (`results_val_TFT_*`) khusus selektor
  grid A3 (data uji tidak boleh dipakai memilih konfigurasi).

## 9. Interpretabilitas (`interpret.py`) — proposal 6.j

- **Feature importance**: bobot VSN `v` dirata-rata `(N,W)` → ranking;
  `summarize_if_contribution` mengambil rank `if_score`. (Paper §7.1 memakai
  persentil 10/50/90; kode mean — bisa diperkaya bila ingin identik.)
- **Attention**: matriks `Ã` (rata-rata antar head, Eq. 15) dirata-rata atas
  batch → `(W+T, W+T)`; kini **secara definisi** matriks yang dipakai model
  (A12). Wajib `model.eval()` (dropout pada probabilitas attention).
- Sampel: 256 window pertama test TFT-IF.
- Retrain fitur terpilih VSN (6.j.ii) — opsional, belum diimplementasikan.

## 10. Konfigurasi Default (`src/config.py`)

| Grup | Parameter | Nilai |
|---|---|---|
| DATA | csv_files, columns | 4 file; 49 kolom (headerless — A7) |
| DATA | id_cols_to_drop | `srcip, sport, dstip, dsport` (A1) |
| DATA | categorical / time / target / label | `proto, state, service` / `stime, ltime` / `attack_cat` / `label` |
| PREPROC | rasio split | 0.70/0.15/0.15 kronologis global (A2; kode masih per file) |
| IF | n_estimators, max_samples, contamination, seed | 100; "auto"(256); "auto"; 42 |
| WINDOW | W, T, stride | 10; 3; 1 (**default awal — final dari grid A3**) |
| MODEL | hidden_size, lstm_layers, attn_heads, dropout, classes | 64; 1; 4; 0.1; 4 |
| TRAIN | batch, epochs, lr, wd | **4096** (A17); 30; 1e-3; 1e-5 |
| TRAIN | patience, scheduler, seed, workers | 5; (2, 0.5); 42; 6 |
| TUNING | n_trials, metric, direction | 20; f1_macro; maximize |

## 11. Deviasi Sadar dari Proposal/Paper (kutip di skripsi)

| ID | Deviasi | Status |
|---|---|---|
| D1 | ~~Kategorikal via LabelEncoder→Linear~~ | **SELESAI (A13)**: entity embeddings |
| D2 | ~~MHA standar + rata-rata bobot~~ | **SELESAI (A12)**: Eq. 13–16 kustom |
| D3 | Skip residual FF kembali ke output post-attention (bukan φ̃ Eq. 22 yang melompati 2 blok) | Terdokumentasi; sebut di metodologi |
| D4 | Tanpa static covariates & known futures; decoder = learned positional queries berkondisi state encoder | Terdokumentasi; didukung Lim §6.6 |
| D5 | Klasifikasi (CE + class weighting) menggantikan quantile output/loss | Inti adaptasi skripsi |
| D6 | ~~Skor IF = −decision_function min-max~~ | **SELESAI (A11)**: literal s(x,n) |
| D7 | ~~Imputasi median pre-split~~ | **SELESAI (A8)**: fit-on-train; dedup tetap pre-split dengan argumen anti-overlap |
| D8 | ~~attack_cat kosong → Normal tanpa cek~~ | **SELESAI (A10)**: assertion |
| D9 | Optuna TPE (paper TFT memakai random search); epochs 15 saat tuning | Peningkatan metodologis; kutip Akiba 2019 |

## 12. Keputusan Tersisa (tidak memblokir run pertama)

1. **Eksekusi grid W/T** (A3) pada kedua model — 6 kombinasi × 2 model; kode
   perlu diperluas (lihat §14). Jalankan di PC RTX 3060 (bukan laptop RTX 3050).
2. **Sensitivitas 80:10:10** — opsional (A2).
3. **Oversampling/undersampling pembanding** (proposal 6.e.iii) — class
   weighting terkunci sebagai strategi utama (A4); comparator SMOTE/dkk.
   opsional bila waktu (atau didiskusikan di bab terkait tanpa eksperimen).
4. **Retrain fitur terpilih VSN** (6.j.ii) — opsional, pendukung.
5. **Persentil VSN importance** (paper §7.1: 10/50/90) — kode mean; opsional.

Referensi A5 sudah lengkap: Prechelt (1998) ada di `model_papers/` + summary
ke-12 (`prechelt1998_early_stopping_summary.md`).

## 13. Checklist Eksperimen Data Asli

1. Taruh 4 CSV asli di `data/raw/`.
2. **EDA dulu**: `python eda.py` → cek `logs/eda/summary.md` (**periksa jumlah
   baris: ringkasan di repo per 13 Okt berasal dari data tiruan, 12.120 baris**),
   `class_distribution.png`, `data_quality_report.csv` (distribusi & null/inf
   sesuai ekspektasi; assertion A10 juga memvalidasi konsistensi label).
3. **Grid W/T** (A3): `python main.py --window W --horizon T --epochs 30`
   untuk 6 kombinasi → bandingkan `results_val_TFT_base.f1_macro` dan
   `results_val_TFT_IF.f1_macro` per kombinasi (simpan hasil per kombinasi
   agar tidak tertimpa). (W=50 butuh ±24–32 GB RAM — lihat §0.3.)
4. Run final pada (W,T) terbaik tiap model, ≥3 seed (42, 43, 44): laporkan
   rata-rata ± simpangan baku (+ `--tune` bila ingin tuning lebih dulu).
5. Cek `logs/results_summary.json`: comparison base vs IF, `per_class`,
   `per_horizon`, rank `if_score`, distribusi split.
6. Jika OOM GPU (RTX 3060 12 GB): turunkan `batch_size` dulu, baru `hidden_size`.
7. Simpan seed & konfigurasi final yang dipakai untuk setiap tabel skripsi.
8. **Checkpoint lama (pra-v1.1) tidak valid** — arsitektur berubah (A12/A13);
   semua hasil harus dari training sesudah tanggal ini.

## 14. Temuan Hasil Training 13 Okt 2026 (belum dipakai di skripsi)

Run: pembagian per berkas, W=10, T=3, batch 4096, tanpa grid. Sumber:
`logs/results_summary.json`.
- **Bug label** (§2.2 butir 6): nama kelas dan FAR di JSON salah. Hitung ulang
  dari `confusion_matrix` (baris indeks 3 = Normal): FAR benar 0,3135 (base)
  dan 0,3504 (IF); accuracy 0,677 / 0,641; F1 macro 0,2816 / 0,2613.
- Kedua model di bawah tebakan "selalu Normal" (accuracy ≈95,9% pada data uji)
  dan jauh di bawah acuan persistensi; presisi kelas serangan 4–9%.
- TFT-IF tidak lebih baik dari TFT-base (satu seed; berhenti di epoch 12 vs 18;
  peringkat `if_score` 28 dari 42). Penyebab belum diuji: variasi seed, AUC
  `if_score` per kelas, pergeseran distribusi skor latih→uji, pengaruh
  pembobotan `balanced` (bobot serangan ±100× Normal).
- Pekerjaan kode tersisa: perbaiki pemetaan label; split global + sort stabil;
  grid (W,T) pada kedua model dengan hasil per kombinasi; acuan persistensi dan
  "selalu Normal" di `evaluate.py`; perbaiki assertion A10 pada pandas 3
  (`astype(str)` mempertahankan NaN → gunakan `isna() | isin([...])`); catat
  bahwa fitur kategorikal masuk IF sebagai kode integer mentah.
