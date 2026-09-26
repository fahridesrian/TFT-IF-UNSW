# Summary — Lim, Arık, Loeff & Pfister (2021), *Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting*  【MAIN PAPER 1】

**File sumber**: `model_papers/1912.09363v3.pdf` (arXiv:1912.09363v3; versi jurnal:
International Journal of Forecasting 37 (2021) 1748–1764)
**Peran dalam proyek**: **paper inti penelitian** — spesifikasi arsitektur TFT yang
diimplementasikan dari nol di `src/tft_model.py`, dikutip di hampir semua bagian
proposal (Bab 1, 5.4, 6). Komponen lain (ELU, LayerNorm, GLU, attention) adalah
sub-referensi dari paper ini.

## 0. Pemetaan notasi paper ↔ proposal ↔ kode

| Paper | Proposal / proyek |
|---|---|
| `k` (look-back window), input `y_{t−k:t}, z_{t−k:t}` | `W` = panjang jendela historis (`WINDOW.W`) |
| `τ_max` (horizon), output `τ ∈ {1..τ_max}` | `T` = panjang jendela target (`WINDOW.T`) |
| `d_model` (hidden state size) | `MODEL.hidden_size` (default 64) |
| `m_H` (jumlah head) | `MODEL.attn_heads` (default 4) |
| Input static `s_i`, observed `z_{i,t}`, known `x_{i,t}` | Hanya padanan `z` (observasi historis + skor anomali IF); `s` & `x` tidak ada (lihat §5) |

## 1. Komponen yang diimplementasikan di proyek

### a. Gated Residual Network (§4.1, Eq. 2–5) — blok dasar semua proses non-linear
- `GRN_ω(a, c) = LayerNorm(a + GLU_ω(η₁))`; `η₁ = W₁η₂ + b₁`;
  `η₂ = ELU(W₂a + W₃c + b₂)`.
- Tanpa context vector → `c = 0`. **Dropout diterapkan ke `η₁`** — yaitu *sebelum*
  gating & LayerNorm (kalimat terakhir §4.1).
- GLU (Eq. 5): `σ(W₄γ + b₄) ⊙ (W₅γ + b₅)` → memungkinkan GRN "dilewati" seluruhnya
  (output gerbang ≈ 0) → kedalaman adaptif. Bobot `ω` di-share (variabel sama memakai
  GRN yang sama di semua time-step, §4.2).

### b. Variable Selection Network (§4.2, Eq. 6–8) — sumber feature importance
- Tiap variabel ditransformasi ke `d_model`: **entity embedding untuk kategorikal,
transformasi linear untuk kontinu** (kalimat awal §4.2).
- Bobot seleksi: `v_χt = Softmax(GRN(Ξ_t, c_s))`; proses per-variabel:
`ξ̃⁽ʲ⁾ₜ = GRN(ξ⁽ʲ⁾ₜ)`; gabungan tertimbang `ξ̃ₜ = Σ v⁽ʲ⁾ ξ̃⁽ʲ⁾ₜ`.
- **Interpretasi (§7.1)**: variabel penting = agregasi bobot seleksi `v⁽ʲ⁾` (persentil
10/50/90) di seluruh test set — ini yang direplikasi `src/interpret.py`
(proposal Bagian 6.j.i, termasuk peringkat `if_score`).

### c. LSTM encoder–decoder sebagai local processing (§4.5.1, Eq. 17)
- Seq2seq layer menangani konteks lokal; **berfungsi sebagai pengganti positional
encoding** (kalimat eksplisit §4.5.1); gated skip: `φ̃ = LayerNorm(ξ̃ + GLU(φ))`.

### d. Interpretable multi-head attention (§4.4, Eq. 9–16; dipakai §4.5.3, Eq. 19–20)
- Scaled dot-product `A(Q,K) = Softmax(QKᵀ/√d_attn)`; **varian interpretable**:
value weights **di-share antar head** dan attention di-**averaging**:
`Ã(Q,K) = (1/H) Σ_h A(QW^Q_h, KW^K_h)` (Eq. 15–16) → satu matriks attention yang
dapat diinterpretasikan langsung.
- `d_V = d_attn = d_model / m_H`.
- **Decoder masking**: tiap dimensi temporal hanya boleh menghadap ke posisi sebelumnya
(Eq. 19 + §4.5.3; preseden Vaswani §3.2.3) → causal mask di kode.
- Gated residual setelah attention: `δ = LayerNorm(θ + GLU(β))` (Eq. 20).

### e. Position-wise feed-forward (§4.5.4, Eq. 21–22)
- `ψ = GRN(δ)` + gated residual yang **melompati seluruh transformer block**:
`ψ̃ = LayerNorm(φ̃ + GLU(ψ))` (Eq. 22) — jalur langsung ke layer seq2seq.

### f. Keputusan arsitektur paper yang diikuti proyek
- **Satu layer interpretable multi-head attention saja** (§6.2: "To preserve
explainability, we adopt only a single interpretable multi-head attention layer").
- LSTM encoder–decoder (bukan CNN) untuk local processing.

## 2. Adaptasi inti skripsi: klasifikasi, bukan quantile forecasting

Paper menghasilkan **quantile output** (Eq. 23: `ŷ = W_q ψ̃ + b_q`) dilatih dengan
**quantile loss** (Eq. 24–25, Q = {0.1, 0.5, 0.9}) dan dievaluasi dengan q-Risk (Eq. 26).
**Proyek mengganti ini seluruhnya**: head klasifikasi `Linear + softmax` (4 kelas) per
titik horizon target dengan **cross-entropy loss** (+ class weighting) — persis seperti
dijelaskan proposal Bagian 5.4.2 (baris 505–513). Encoder (VSN → LSTM → attention → FF)
tetap mengikuti §4.

## 3. Deviasi proyek dari paper (dengan dasar justifikasi)

| # | Paper | Proyek | Dasar/justifikasi |
|---|---|---|---|
| 1 | Static covariate encoders, 4 context vectors `c_s, c_e, c_c, c_h` (§4.3, §4.5.2 static enrichment Eq. 18) | **Tidak ada static covariates**; GRN statik berjalan tanpa context (`c=0`) | UNSW-NB15 tidak punya metadata statik per-sekuens yang wajar; `handoff.md` |
| 2 | Known future inputs masuk decoder seq2seq (§4.5.1) | **Tidak ada known futures**; decoder memakai **learned positional query tokens** | Known futures tidak tersedia secara alami dalam deteksi intrusi (proposal baris 514–516). **Didukung paper sendiri** (§6.6 akhir): seq2seq layer dapat diperlakukan sebagai hyperparameter, termasuk opsi "simple positional encoding without any local processing" |
| 3 | Quantile output + quantile loss (Eq. 23–26) | Head klasifikasi + cross-entropy | Inti adaptasi skripsi (§2 di atas) |
| 4 | Hyperparameter tuning via **random search** (§6.2) | **Optuna/TPE** (F1-macro validasi) | Peningkatan metodologis; tetap sekuens train/val/test terpisah (§6.2 & Appendix A juga kronologis) |
| 5 | Ruang pencarian: state size {10,20,40,80,160,240,320}, dropout {0.1..0.9}, minibatch {64,128,256}, lr {0.0001, 0.001, 0.01}, **max grad norm {0.01, 1, 100}**, heads {1,4} | `hidden_size=64` (di luar grid paper), dropout 0.1, batch 1024, lr 1e-3, heads 4, **tanpa gradient clipping** | Nilai proyek pilihan sendiri; sebagian (dropout 0.1, lr 1e-3, heads 4) berada dalam rentang paper. Gradient clipping paper **tidak** diadopsi — konsisten dengan tidak adanya clipping di config |

## 4. Titik verifikasi untuk spec.md (bedakan paper vs kode aktual)

1. **Varian attention**: paper memakai *interpretable* MHA (value di-share antar head +
rata-rata attention, Eq. 14–16). `handoff.md` menyebut kode memakai
`nn.MultiheadAttention` standar + causal mask. **Perlu diverifikasi** `tft_model.py`:
jika ingin setia ke Eq. 16, V harus di-share & attention di-rata-rata — ini juga
menentukan validitas analisis attention weight (Bagian 6.j.iii proposal).
2. **Representasi variabel kategorikal**: paper memakai *entity embeddings* (§4.2);
proyek memakai `LabelEncoder` untuk `proto/state/service` — perlu dicek bagaimana
integer hasil encoding ditransformasi ke `d_model` di dalam VSN.
3. **Posisi dropout**: paper menaruh dropout pada `η₁` (sebelum GLU/LayerNorm, §4.1) —
cek kesesuaian penempatan dropout di GRN implementasi.

## 5. Bagian paper yang TIDAK dipakai (tidak perlu dibuka ulang)

- §2 related work, §5 quantile loss & q-Risk (Eq. 24–26), §6.1/6.4/6.5 eksperimen
  (Electricity/Traffic/Retail/Volatility, Tabel 2), §6.6 ablation (cukup dikutip
  bahwa semua komponen berkontribusi; gating paling berdampak pada dataset kecil/noisy).
- §7.3 regime identification (Eq. 27–30, Bhattacharyya distance) — tidak direplikasi;
  interpretabilitas proyek hanya §7.1 (VSN importance) + §7.2 (pola attention).
- Appendix A/B (detail dataset forecasting paper).

## Kutipan (APA — sudah ada di daftar pustaka proposal)

Lim, B., Arık, S. Ö., Loeff, N., & Pfister, T. (2021). Temporal Fusion Transformers
for interpretable multi-horizon time series forecasting. International Journal of
Forecasting, 37, 1748–1764. http://arxiv.org/abs/1912.09363
