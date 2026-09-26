# Summary — Kingma & Ba (2015), *Adam: A Method for Stochastic Optimization*

**File sumber**: `model_papers/1412.6980v9.pdf` (arXiv:1412.6980v9, ICLR 2015)
**Peran dalam proyek**: optimizer pelatihan TFT-base & TFT-IF (proposal Bagian 6.h.iv),
diimplementasikan sebagai `torch.optim.Adam` di `src/train.py`.

> Prinsip: hanya isi yang benar-benar dipakai implementasi. Bagian paper yang tidak
> relevan dicatat singkat di §5 agar tidak perlu dibuka ulang.

## 1. Yang dipakai: Algorithm 1 (update rule Adam)

Untuk gradien minibatch `g_t`, Adam memelihara dua rata-rata bergerak eksponensial
(semua operasi element-wise):

- `m_t = β1·m_{t−1} + (1−β1)·g_t` — estimasi momen ke-1 (mean gradien)
- `v_t = β2·v_{t−1} + (1−β2)·g_t²` — estimasi momen ke-2 mentah (uncentered variance)
- Koreksi bias inisialisasi-nol: `m̂_t = m_t/(1−β1^t)`, `v̂_t = v_t/(1−β2^t)`
- Update parameter: `θ_t = θ_{t−1} − α · m̂_t/(√v̂_t + ε)`

Inilah yang dieksekusi `torch.optim.Adam` pada setiap `optimizer.step()` di training
loop. Tanpa koreksi bias, langkah awal menjadi terlalu besar — terutama saat `β2`
mendekati 1 (kasus gradien sparse; lihat §3 paper tentang efeknya di RMSProp).

## 2. Hyperparameter default paper → nilai proyek

| Simbol paper | Makna | Default paper | Nilai di proyek |
|---|---|---|---|
| α | stepsize / learning rate | 0.001 | `TRAIN.learning_rate = 1e-3` (= default paper; bisa dioptimasi Optuna) |
| β1 | decay momen ke-1 | 0.9 | default `torch.optim.Adam` (tidak diubah) |
| β2 | decay momen ke-2 | 0.999 | default `torch.optim.Adam` (tidak diubah) |
| ε | pengaman pembagi | 10⁻⁸ | default `torch.optim.Adam` (tidak diubah) |

## 3. Alasan pemilihan Adam yang relevan untuk konteks penelitian

- Dirancang untuk **objective stokastik yang noisy/sparse dan non-stationary** — cocok
  dengan cross-entropy ber-class-weighting pada minibatch besar (1024).
- Magnitudo langkah efektif `|Δt|` praktis **terbatas oleh α** (efek signal-to-noise
  ratio) → skala learning rate mudah ditebak sejak awal.
- **Invarian terhadap rescaling gradien** — relevan karena fitur hasil Min-Max dan hasil
  encoding kategorikal menghasilkan skala gradien yang berbeda-beda antar parameter.
- Kebutuhan memori kecil (dua vektor sebesar parameter) — aman untuk RTX 3060 12GB
  dengan `hidden_size` 64.

## 4. Keputusan proyek yang menempel ke paper ini

1. `learning_rate = 1e-3` memang default yang direkomendasikan paper (bukan angka acak).
2. β1, β2, ε dibiarkan pada default `torch.optim.Adam`, identik dengan rekomendasi paper.
3. `TRAIN.weight_decay = 1e-5` (regularisasi L2) **bukan** spesifikasi paper — keputusan
   proyek sendiri; paper hanya memakai L2 weight decay pada eksperimen multi-layer NN
   (§6.2) tanpa menetapkan nilai.
4. LR scheduler (ReduceLROnPlateau, patience 2, factor 0.5) adalah *annealing* eksplisit
   yang terpisah dari Adam; paper hanya menyebut Adam memiliki *automatic annealing*
   alami melalui menurunnya SNR mendekati optimum (§2.1).

## 5. Bagian paper yang TIDAK dipakai (tidak perlu dibuka ulang)

- §4 & §10: analisis konvergensi, regret bound, dan bukti (teoretis, setting convex).
- §5: relasi Adam ↔ RMSProp/AdaGrad/AdaDelta (konteks historis).
- §6: seluruh eksperimen (MNIST/IMDB/CIFAR-10, ablation bias-correction pada VAE).
- §7.1 AdaMax (varian berbasis norma L∞) dan §7.2 temporal averaging — tidak
  diimplementasikan dalam proyek ini.

## Kutipan (APA — sudah terdaftar di `references_addendum.md`)

Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization.
International Conference on Learning Representations (ICLR). arXiv:1412.6980.
