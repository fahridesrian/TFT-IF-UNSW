# Summary — Clevert, Unterthiner & Hochreiter (2016), *Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)*

**File sumber**: `model_papers/1511.07289v5.pdf` (arXiv:1511.07289v5, ICLR 2016)
**Peran dalam proyek**: fungsi aktivasi **ELU di dalam Gated Residual Network (GRN)**
pada `src/tft_model.py` (`nn.ELU()`), mengikuti spesifikasi komponen GRN dari
Lim et al. 2021 dan penjelasan proposal Bagian 5.4.2 (ELU + GLU + residual + LayerNorm).

> Prinsip: hanya isi yang benar-benar dipakai implementasi. Bagian teoretis/eksperimen
> dicatat singkat di §4 agar tidak perlu dibuka ulang.

## 1. Yang dipakai: definisi ELU (Persamaan 15 paper)

Dengan hyperparameter `α > 0`:

- `f(x) = x`            jika `x > 0` (identitas)
- `f(x) = α·(exp(x) − 1)` jika `x ≤ 0` (eksponensial, saturasi ke `−α`)

Turunan: `f′(x) = 1` untuk `x > 0`, dan `f′(x) = f(x) + α` untuk `x ≤ 0`.

**Nilai yang dipakai proyek: α = 1.0** — sama dengan setting eksperimen paper
dan sama dengan default `nn.ELU()` PyTorch. `α` tidak termasuk ruang pencarian
Optuna (tidak perlu; paper tidak menyarankan tuning α).

## 2. Pemetaan ke kode

| Paper | Di proyek |
|---|---|
| ELU, α = 1.0 | `nn.ELU()` (default `alpha=1.0`) di dalam `GatedResidualNetwork` (`tft_model.py`) |
| — | Tidak ada hyperparameter tambahan dari paper ini yang masuk `config.py` / Optuna |

Posisi ELU dalam GRN: hasil transformasi non-linear (bersama konteks opsional)
sebelum digerbangi GLU, lalu residual + LayerNorm — sesuai Persamaan 5 proposal:
`GRN(a, c) = LayerNorm(a + GLU(η₁))`.

## 3. Rasional yang bisa dikutip di skripsi (mengapa ELU, bukan ReLU/tanh)

1. **Sisi positif identitas** → gradien tidak kontraktif, masalah vanishing gradient
   teratasi seperti ReLU (berbeda dengan tanh/sigmoid yang kontraktif hampir di mana-mana).
2. **Nilai negatif** → memusatkan rata-rata aktivasi mendekati nol, mengurangi efek
   *bias shift*, sehingga pembelajaran lebih cepat dan lebih stabil.
3. **Saturasi negatif yang jelas** (menuju `−α`) → unit yang "mati" membawa sedikit
   variasi/informasi → representasi lebih *noise-robust* dan ber-kompleksitas rendah;
   deaktivasi bersifat non-informatif sehingga hanya unit aktif yang membawa sinyal.
4. Eksperimen paper (CIFAR-100: 28,75% vs 31,56% ReLU, p < 0,001) mendukung 1–3 —
   cukup dikutip sebagai justifikasi pemilihan aktivasi, tanpa perlu direproduksi.

## 4. Bagian paper yang TIDAK dipakai (tidak perlu dibuka ulang)

- §1–2: teori *bias shift* via unit natural gradient / Fisher information matrix,
  Theorem 1–2, dan seluruh appendix (Lemma 1–2, bukti blok-matriks).
- §4: eksperimen MNIST/CIFAR-10/CIFAR-100/ImageNet, autoencoder, perbandingan
  dengan ReLU/LReLU/SReLU dan batch normalization.
- Catatan arsitektur CNN paper (jumlah layer, dropout schedule, lr schedule) —
  konteks vision, tidak relevan untuk TFT 1D.

## Kutipan (APA — sudah terdaftar di `references_addendum.md`)

Clevert, D.-A., Unterthiner, T., & Hochreiter, S. (2016). Fast and accurate deep
network learning by exponential linear units (ELUs). International Conference on
Learning Representations (ICLR). https://doi.org/10.48550/arXiv.1511.07289
