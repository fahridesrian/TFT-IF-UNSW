# prechelt1998_early_stopping_summary.md

## Identitas Paper
- **Judul**: Early Stopping — But When?
- **Penulis**: Lutz Prechelt (Fakultät für Informatik, Universität Karlsruhe)
- **Publikasi**: dalam G.B. Orr & K.-R. Müller (Eds.), *Neural Networks: Tricks of
  the Trade*, LNCS 1524, Springer-Verlag, hlm. 55–69, 1998.
- **Peran di proyek**: justifikasi keputusan **A5** (`spec.md` §0.5) — early
  stopping berbasis `val_loss` dengan `patience=5` di `train.py`; referensi
  skripsi untuk desain kriteria berhenti.

## 1. Teknik dasar early stopping (§2.1.1–2.1.2)
1. Split data latih menjadi train + **validation set**; validasi **tidak pernah**
   dipakai untuk update bobot — murni estimasi generalization error.
2. Latih hanya pada train; evaluasi error validasi berkala (tiap "strip" k epoch).
3. Berhenti bila error validasi lebih tinggi dari pemeriksaan sebelumnya.
4. **Hasil training = bobot pada langkah dengan error validasi terendah** —
   persis pola `train.py`: snapshot `best_state` (CPU) di-load di akhir.

Kurva validasi riil **bergerigi** (Fig. 2.2: hingga 16 local minima sebelum
overfitting parah) → berhenti pada kenaikan pertama terlalu dini; kriteria
butuh **toleransi** (patience/strip), bukan reaksi pada satu kenaikan.

## 2. Kelas kriteria berhenti (§2.2.1) — definisi formal
- `E_opt(t) = min_{t'≤t} E_va(t')` — error validasi terbaik sejauh ini.
- **Generalization loss**: `GL(t) = 100 · (E_va(t)/E_opt(t) − 1)`.
  Kelas **GL_α**: berhenti pada epoch pertama dengan GL(t) > α (berbasis magnitudo).
- **Training progress** (strip k, default k=5):
  `P_k(t) = 1000 · ( (1/k)Σ E_tr(t') / min E_tr(t') − 1 )` atas strip;
  Kelas **PQ_α**: berhenti bila `GL(t)/P_k(t) > α` (gabungan loss & progress).
- Kelas **UP_s**: berhenti bila E_va **naik pada s strip berturut-turut**
  (UP1: `E_va(t) > E_va(t−k)`). Berbasis **sign/perubahan lokal**, bukan magnitudo.
- Jaminan terminasi: progress < 0.1 atau maksimum 3000 epoch
  (analog proyek: `epochs=30`).

## 3. Aturan pemilihan (§2.2.2) & temuan empiris (§2.3, Tabel 2.1)
- Eksperimen: **1296 run** (12 masalah Proben1 × 24 topologi × 3 partisi × 3
  run), 14 kriteria dievaluasi simultan per run.
- **Tradeoff kuantitatif (§2.3.4.6)**: kriteria lambat memperbaiki test error
  ±4% (B: 1.024 → 0.988) dengan biaya waktu ±4× (S: 0.766 → 3.095). Selisih
  error antar kriteria umumnya kecil (±1–2%) — kriteria cepat tetap "good"
  (menemukan error validasi terbaik run-nya) pada ±60% run, terlambat ±80%.
- **Tradeoff terbaik untuk satu run: UP3, UP4, UP6** (badness vs slowness;
  semuanya kelas UP) — sekaligus yang lebih robust.
- **Robustness — sangat relevan untuk proyek ini (§2.3.4.4)**: semua kriteria
  GL **tidak stabil pada masalah `building` yang partisinya KRONOLOGIS
  (non-random)**; varian lambat kriteria non-GL (UP/PQ) justru robust pada kasus
  itu. Pipeline kita split **kronologis** → kelas **UP/patience** adalah pilihan
  yang didukung temuan ini, bukan threshold GL.

## 4. Pemetaan ke implementasi (`src/train.py`)
| Konsep Prechelt | Implementasi proyek |
|---|---|
| E_va per strip (default k=5 epoch) | `val_loss` per epoch (strip k=1) |
| UP_s: E_va naik s strip berturut-turut | `patience=5` epoch tanpa perbaikan > `min_delta=1e-5` |
| Hasil = bobot pada E_opt (minimum E_va) | `best_state` (CPU snapshot) di-load di akhir |
| Batas keras epoch (3000 di paper) | `epochs=30` |
| Validasi tak pernah dipakai update bobot (§2.3.2) | ✓ val hanya untuk stopping/scheduler/monitoring |

`patience=5` berada di rentang sweet spot **UP3–UP6** (Tabel 2.1), dan kelas UP
robust untuk data kronologis — dua alasan yang saling menguatkan untuk A5.
Catatan Prechelt (§2.3.2): aplikasi riil kadang menggabungkan kembali val ke
train lalu retrain — tidak dilakukan di proyek ini (val tetap terpisah; bisa
disebut sebagai keterbatasan/opsi).

## 5. Poin lain yang bisa dikutip
- "Kriteria cepat memperbaiki prediktabilitas **waktu**; kriteria lambat
  prediktabilitas **kualitas**" (§2.3.4.5).
- Analisis teoretis early stopping (Wang et al.; Amari et al.) terbatas pada
  skenario sempit → investigasi empiris diperlukan; landasan pendekatan
  patience yang praktis.

## Sitasi
Prechelt, L. (1998). Early Stopping — But When? Dalam: Orr, G.B., Müller, K.-R.
(Eds.), *Neural Networks: Tricks of the Trade*, LNCS 1524, Springer, hlm. 55–69.
