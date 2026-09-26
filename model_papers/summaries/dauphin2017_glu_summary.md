# Summary — Dauphin, Fan, Auli & Grangier (2017), *Language Modeling with Gated Convolutional Networks*

**File sumber**: `model_papers/dauphin17a.pdf` (ICML 2017, PMLR 70)
**Peran dalam proyek**: sumber mekanisme **Gated Linear Unit (GLU)** yang dipakai
di dalam setiap `GatedResidualNetwork` (`src/tft_model.py`, class `GLU`) — sesuai
spesifikasi GRN pada Lim et al. 2021 dan Persamaan 5 proposal:
`GRN(a, c) = LayerNorm(a + GLU(η₁))`.

> Catatan penting: yang diadopsi **hanya mekanisme gerbangnya (GLU)**, bukan
> arsitektur convolutional network paper ini. Paper berlatar language modeling;
> GLU-nya yang diambil TFT dan diturunkan ke proyek.

## 1. Yang dipakai: definisi GLU (§2, Eq. 1) dan sifat gradiennya (§3, Eq. 3)

- **Definisi**: `GLU(X) = (X∗W + b) ⊗ σ(X∗V + c)` — jalur linear yang digerbangi
  sigmoid (`⊗` = element-wise product). Implementasinya butuh **dua proyeksi**:
  satu untuk konten linear, satu untuk gerbang sigmoid.
- **Sifat gradien** (Eq. 3): `∇[X ⊗ σ(X)] = ∇X ⊗ σ(X) + X ⊗ σ′(X)∇X` — ada jalur
  `∇X ⊗ σ(X)` **tanpa downscaling** untuk unit gerbang yang aktif → berperan seperti
  *multiplicative skip connection* yang melancarkan aliran gradien antar layer.
- Perbandingan pada paper: GLU **konvergen lebih cepat dan ke hasil lebih baik**
  dibanding GTU (`tanh ⊗ σ`, Eq. 2 — gradiennya menyusut karena faktor tanh′ dan σ′),
  Tanh, maupun ReLU (§5.2, Gambar 3; gap ≈ 5 poin perplexity vs ReLU).

## 2. Pemetaan ke kode

| Paper | Di proyek |
|---|---|
| `GLU(X) = (X∗W + b) ⊗ σ(X∗V + c)` | Class `GLU` di `tft_model.py` — gerbang sigmoid yang mengontrol informasi mana yang diteruskan, di dalam setiap GRN |
| Peran gerbang: "model memilih fitur/informasi mana yang relevan diteruskan" (§3) | Konsisten dengan fungsi GRN dalam proposal: "mengontrol aliran informasi, mempertahankan yang penting, menekan yang kurang relevan" (Bagian 5.4.2) |
| — | Tidak ada hyperparameter khusus dari GLU yang masuk config/Optuna (bentuknya fix) |

## 3. Rasional yang bisa dikutip di skripsi (mengapa gerbang, dan mengapa GLU)

1. Gerbang memberi kontrol eksplisit atas aliran informasi antar layer — analog dengan
   gerbang input/forget pada LSTM, tetapi cukup **output gate saja** (paper menemukan
   jaringan konvolusional tidak butuh forget gate).
2. GLU mempertahankan kemampuan non-linear **sekaligus** menyediakan jalur linear
   bagi gradien → mengurangi masalah vanishing gradient pada arsitektur dalam.
3. Bukti empiris paper: gerbang memberi keunggulan besar dibanding tanpa gerbang
   (linear/bilinear, §5.3), dan GLU unggul atas GTU/Tanh/ReLU (§5.2).

## 4. Bagian paper yang TIDAK dipakai (tidak perlu dibuka ulang)

- Arsitektur gated **convolutional** network secara utuh (stacked temporal convolutions,
  kernel shifting/causal padding, bottleneck blocks) — proyek memakai LSTM + attention, bukan CNN.
- Adaptive softmax (§2), Nesterov momentum, **gradient clipping** (Eq. 4) dan weight
  normalization (§4.2, §5.5) — tidak diadopsi proyek (proyek: Adam + AMP).
- Eksperimen language modeling (GBW, WikiText-103, Gigaword, Penn Treebank) dan
  temuan context size (§5.4) — konteks NLP.

## Kutipan (APA — sudah terdaftar di `references_addendum.md`)

Dauphin, Y. N., Fan, A., Auli, M., & Grangier, D. (2017). Language modeling with
gated convolutional networks. Proceedings of the 34th International Conference on
Machine Learning (ICML), 933–941.
