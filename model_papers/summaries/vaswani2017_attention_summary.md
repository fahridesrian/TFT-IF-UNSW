# Summary — Vaswani et al. (2017), *Attention Is All You Need*

**File sumber**: `model_papers/NIPS-2017-attention-is-all-you-need-Paper.pdf` (NIPS 2017)
**Peran dalam proyek**: dasar mekanisme **scaled dot-product attention & multi-head
attention** pada komponen attention TFT (`src/tft_model.py`, memakai
`nn.MultiheadAttention` dengan causal mask) — sudah tercantum di daftar pustaka
proposal; Persamaan 4 proposal (`Attention(Q,K,V) = softmax(QKᵀ/√d_k)·V`) adalah
persis Persamaan 1 paper ini.

> Catatan: yang diadopsi **mekanisme attention-nya saja**, bukan arsitektur Transformer
> utuh. Model proyek adalah TFT (Lim et al. 2021) yang memadukan attention dengan
> LSTM encoder-decoder, bukan Transformer murni tanpa rekurensi.

## 1. Yang dipakai dari paper

### a. Scaled dot-product attention (§3.2.1, Eq. 1)
`Attention(Q, K, V) = softmax(QKᵀ/√d_k)·V`
- **Faktor skala 1/√d_k**: untuk `d_k` besar, dot-product tumbuh besar (varians `d_k`)
  dan mendorong softmax ke daerah gradien sangat kecil — skala mengatasi ini.
- Dipilih dot-product (bukan additive) karena lebih cepat & hemat memori lewat
  operasi matriks teroptimasi — di kode: `nn.MultiheadAttention`.

### b. Multi-head attention (§3.2.2, Eq. 2)
`MultiHead(Q,K,V) = Concat(head₁,…,head_h)·W^O`, dengan
`head_i = Attention(Q·W_i^Q, K·W_i^K, V·W_i^V)` — `h` proyeksi paralel ke subruang
representasi berbeda; dengan `d_k = d_model/h`, biaya komputasi total ≈ single-head
berdimensi penuh. Di proyek: `attn_heads = 4` (default config, dapat dioptimasi Optuna).

### c. Masking pada decoder self-attention (§3.2.3)
Posisi dilarang menghadap ke posisi **berikutnya**: nilai ilegal di input softmax
diset **−∞** sebelum softmax. Di proyek: **causal mask** pada interpretable
multi-head attention TFT — sudah terverifikasi di smoke test (`handoff.md`: posisi
awal tidak "mengintip" masa depan; attention weight berjumlah 1 per baris pada
`model.eval()`).

### d. Pola residual + LayerNorm (§3.1)
`LayerNorm(x + Sublayer(x))` — preseden pola gating/residual yang diadopsi ulang
oleh GRN TFT (lima proyek memakai pola serupa, lihat summary Ba et al. 2016).

## 2. Bagian paper yang TIDAK dipakai (tidak perlu dibuka ulang)

- **Arsitektur Transformer utuh**: encoder-decoder stack murni attention (N=6 layer),
  feed-forward position-wise (Eq. 2), shared embedding–softmax weight (§3.4).
- **Positional encoding sinusoidal** (§3.5): TFT memakai *learned positional query
  tokens* pada decoder — keputusan desain proyek yang konsisten dengan keterangan
  `handoff.md` (paper sendiri mencatat learned embeddings hasilnya setara, Tabel 3 row E).
- **Jadwal learning rate warmup** (§5.3, Eq. 3): proyek memakai ReduceLROnPlateau
  (patience 2, factor 0.5), bukan formula warmup/inverse-sqrt paper. Juga β₂ Adam paper
  = 0.98, sedangkan proyek memakai default Kingma & Ba = 0.999 (lihat summary
  kingma2015_adam).
- Label smoothing (§5.4), beam search/checkpoint averaging (§6.1), ablation Tabel 3,
  dan seluruh eksperimen machine translation WMT 2014.

## 3. Relevansi tambahan untuk skripsi

- §4 ("Why Self-Attention") membandingkan kompleksitas per layer: self-attention
  `O(n²·d)` dengan **maximum path length O(1)** vs recurrent `O(n)` — argumen bawaan
  mengapa attention menangkap **dependensi temporal jangka panjang** lebih baik,
  yang dikutip proposal (Bagian 5.4.1) sebagai alasan pemilihan model Transformer-based.
- §4 paragraf terakhir: distribusi attention **bisa diinspeksi → model lebih
  interpretable** — preseden untuk analisis attention weight pada Bagian 6.j proposal.

## Kutipan (APA — sudah ada di daftar pustaka proposal)

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N.,
Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need. Advances in Neural
Information Processing Systems 30 / NIPS 2017, 5998–6008.
