# Summary — Ba, Kiros & Hinton (2016), *Layer Normalization*

**File sumber**: `model_papers/1607.06450v1.pdf` (arXiv:1607.06450v1)
**Peran dalam proyek**: `nn.LayerNorm` di **setiap gating block** TFT (`src/tft_model.py`):
di dalam `GatedResidualNetwork` (sesuai Persamaan 5 proposal:
`GRN(a, c) = LayerNorm(a + GLU(η₁))`), post-LSTM, post-attention, dan pre-output head.

> Prinsip: hanya isi yang benar-benar dipakai implementasi. Teori/eksperimen dicatat
> singkat di §5 agar tidak perlu dibuka ulang.

## 1. Yang dipakai: definisi LayerNorm (Eq. 3, 15–16 paper)

Untuk vektor input suatu layer `z ∈ ℝ^D` pada **satu sampel** (satu training case):

- `μ = (1/D) Σ z_i`, `σ = √[(1/D) Σ (z_i − μ)²]` — statistik dihitung **lintas fitur
  dalam satu layer**, bukan lintas sampel dalam batch.
- `LN(z; α, β) = ((z − μ)/σ) ⊙ α + β` — dengan **gain α dan bias β adaptif** yang
  dipelajari, diterapkan setelah normalisasi (sebelum non-linearitas).

## 2. Pemetaan ke kode

| Paper | Di proyek |
|---|---|
| `LN(z; α, β)` dengan gain/bias adaptif | `nn.LayerNorm(hidden_size)` — `elementwise_affine=True` (gain & bias dipelajari), `eps=1e-5` |
| Inisialisasi default paper: gain = 1, bias = 0 | Default PyTorch `nn.LayerNorm` (`weight=1`, `bias=0`) — identik, tidak diubah |
| Tidak ada hyperparameter tambahan yang disarankan paper | `eps` dan affine dibiarkan default; tidak masuk ruang Optuna |

## 3. Alasan pemilihan LayerNorm (bukan BatchNorm) — relevan untuk konteks proyek

1. **Bebas dari ukuran batch**: statistik dari satu sampel sendiri, tanpa running
   average → komputasi **identik saat training dan inference**, tidak ada perbedaan
   behavior `model.train()` vs `model.eval()` akibat statistik batch.
2. **Cocok untuk arsitektur rekuren**: LayerNorm dirancang agar bisa diterapkan pada
   RNN — statistik dihitung per time-step dengan satu set gain/bias yang dibagi semua
   time-step (persis pola pemakaian di encoder-decoder LSTM TFT).
3. **Stabilisasi hidden state dynamics**: normalisasi membuat layer invarian terhadap
   rescaling seluruh summed input → mengurangi risiko exploding/vanishing pada aliran
   LSTM → attention (§3.1 paper).
4. **Peringatan paper yang diikuti proyek** (§6.6): LN **tidak diterapkan pada layer
   logit/softmax akhir** — di kode, LayerNorm hanya di blok internal; head klasifikasi
   `Linear → softmax` tidak dinormalisasi.

## 4. Catatan posisi penerapan di proyek

Proyek memakai pola "LN sebagai bagian gating block" (post-LSTM, post-attention,
pre-output) — **bukan** pola "layer-normalized LSTM" dari supplementary paper
(Eq. 20–22, LN di dalam gerbang sel LSTM). Kode memakai `nn.LSTM` standar; LN
diterapkan pada keluaran blok, sesuai spesifikasi TFT Lim et al. 2021.

## 5. Bagian paper yang TIDAK dipakai (tidak perlu dibuka ulang)

- §2: latar batch normalization (cukup tahu bedanya dengan LN).
- §5: analisis invariance (Tabel 1, bukti re-scaling/re-centering), geometri
  parameter space / Riemannian metric / Fisher information — teoretis.
- §6: semua eksperimen (order-embedding MSCOCO, attentive reader, skip-thoughts,
  DRAW, handwriting, MNIST, CNN).
- Supplementary: persamaan LN di dalam sel LSTM/GRU (tidak diadopsi, lihat §4).
- Relasi ke weight normalization (§4–5) — konteks saja.

## Kutipan (APA — sudah terdaftar di `references_addendum.md`)

Ba, J. L., Kiros, J. R., & Hinton, G. E. (2016). Layer normalization. arXiv
preprint arXiv:1607.06450.
