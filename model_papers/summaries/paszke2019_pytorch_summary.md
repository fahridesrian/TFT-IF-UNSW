# Summary — Paszke et al. (2019), *PyTorch: An Imperative Style, High-Performance Deep Learning Library*

**File sumber**: `model_papers/1912.01703v1.pdf` (arXiv:1912.01703v1, NeurIPS 2019)
**Peran dalam proyek**: framework seluruh implementasi — TFT custom (`tft_model.py`),
training loop + AMP (`train.py`), DataLoader, autograd. Dikutip sebagai justifikasi
"diimplementasikan dari nol dengan PyTorch, bukan library TFT siap pakai".

> Paper ini jangkar sitasi, bukan referensi teknis rinci — summary sengaja pendek.

## 1. Konsep paper yang menempel ke keputusan implementasi proyek

| Konsep paper | Pemakaian di proyek |
|---|---|
| **Imperative / define-by-run**: model hanyalah program Python biasa (§4.1) | Memungkinkan arsitektur TFT non-standar (VSN, GRN, *masked interpretable attention*, head klasifikasi menggantikan quantile output) ditulis bebas tanpa terkunci pola forecasting library — ini dasar keputusan "custom dari nol" |
| Pola `nn.Module` (layer = class dengan `__init__`/`forward`, Listing 1) | Struktur semua komponen di `tft_model.py` |
| **Autograd reverse-mode** (§4.3) | `loss.backward()` pada training loop; gradien mengalir ke seluruh komponen custom |
| `DataLoader` (batching, shuffling, paralelisasi, pinned CUDA memory) (§4.2) | Batching dataset window `(N, W, F)` dengan `batch_size=1024`, `num_workers=6` (`TrainConfig`) |
| Eksekusi asinkron GPU + caching allocator (§5.2–5.3) | Konteks performa di RTX 3060; tidak ada intervensi manual yang dilakukan proyek |

## 2. Rasional yang bisa dikutip di skripsi

- Framework dipilih karena memberi **kontrol penuh atas arsitektur** (semua komponen
  adalah program Python biasa) sekaligus performa setara library tercepat (benchmark
  §6.3, dalam 17% dari framework tercepat) — mendukung reproduksibilitas spesifikasi
  TFT Lim et al. 2021 secara eksak, termasuk adaptasi klasifikasi yang tidak tersedia
  di library forecasting siap pakai.

## 3. Bagian paper yang TIDAK dipakai (tidak perlu dibuka ulang)

- §3 prinsip desain, §5.1 core C++/TorchScript, §5.4 multiprocessing, §5.5 reference
  counting, §6 benchmark & adoption, dan seluruh daftar pustaka internal paper.

## Kutipan (APA — sudah terdaftar di `references_addendum.md`)

Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T.,
Lin, Z., Gimelshein, N., Antiga, L., Desmaison, A., Köpf, A., Yang, E., DeVito, Z.,
Raison, M., Tejani, A., Chilamkurthy, S., Steiner, B., Fang, L., Bai, J., &
Chintala, S. (2019). PyTorch: An imperative style, high-performance deep learning
library. Advances in Neural Information Processing Systems, 32, 8024–8035.
