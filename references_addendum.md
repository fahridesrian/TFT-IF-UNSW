# Rujukan Tambahan untuk Implementasi Kode

Rujukan ini melengkapi daftar pustaka proposal Anda (yang sudah mencakup
Lim et al. 2021 untuk TFT, Vaswani et al. 2017 untuk self-attention, dan
Liu et al. 2008/2012 untuk Isolation Forest). Setiap entri diberi keterangan
komponen kode mana yang menggunakannya.

## Format APA (siap tempel ke daftar pustaka, urut abjad menyatu dengan yang sudah ada)

```
Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). Optuna: A
    next-generation hyperparameter optimization framework. Proceedings of
    the 25th ACM SIGKDD International Conference on Knowledge Discovery &
    Data Mining, 2623-2631. https://doi.org/10.1145/3292500.3330701

Ba, J. L., Kiros, J. R., & Hinton, G. E. (2016). Layer normalization. arXiv
    preprint arXiv:1607.06450.

Clevert, D.-A., Unterthiner, T., & Hochreiter, S. (2016). Fast and accurate
    deep network learning by exponential linear units (ELUs). International
    Conference on Learning Representations (ICLR).
    https://doi.org/10.48550/arXiv.1511.07289

Dauphin, Y. N., Fan, A., Auli, M., & Grangier, D. (2017). Language modeling
    with gated convolutional networks. Proceedings of the 34th International
    Conference on Machine Learning (ICML), 933-941.

Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization.
    International Conference on Learning Representations (ICLR).

Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G.,
    Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., Desmaison, A., Kopf,
    A., Yang, E., DeVito, Z., Raison, M., Tejani, A., Chilamkurthy, S.,
    Steiner, B., Fang, L., Bai, J., & Chintala, S. (2019). PyTorch: An
    imperative style, high-performance deep learning library. Advances in
    Neural Information Processing Systems, 32, 8024-8035.

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B.,
    Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V.,
    Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., &
    Duchesnay, E. (2011). Scikit-learn: Machine learning in Python. Journal
    of Machine Learning Research, 12, 2825-2830.
```

## Peta rujukan -> komponen kode

| Rujukan | Dipakai di | Komponen |
|---|---|---|
| Dauphin et al. (2017) | `tft_model.py` | `GLU` class (Gated Linear Unit) di dalam setiap Gated Residual Network |
| Clevert et al. (2016) | `tft_model.py` | Aktivasi `nn.ELU()` di dalam `GatedResidualNetwork` |
| Ba et al. (2016) | `tft_model.py` | `nn.LayerNorm` di setiap gating block (post-LSTM, post-attention, pre-output) |
| Vaswani et al. (2017)* | `tft_model.py` | `nn.MultiheadAttention` (scaled dot-product attention, dasar interpretable multi-head attention TFT) |
| Kingma & Ba (2015) | `train.py` | `torch.optim.Adam` sebagai optimizer pelatihan |
| Akiba et al. (2019) | `hyperparameter_tuning.py` | Seluruh proses tuning (`optuna.create_study`, `trial.suggest_*`) |
| Pedregosa et al. (2011) | `preprocessing.py`, `isolation_forest_module.py`, `evaluate.py` | `LabelEncoder`, `MinMaxScaler`, `IsolationForest`, seluruh metrik evaluasi |
| Paszke et al. (2019) | seluruh modul model/training | Framework PyTorch yang dipakai untuk implementasi TFT kustom |

\* Sudah ada di daftar pustaka proposal Anda — disertakan di tabel ini hanya
untuk kelengkapan peta rujukan -> kode.

## Catatan untuk Bab Metodologi

Disarankan menyebutkan eksplisit bahwa:

1. Arsitektur TFT **diimplementasikan dari nol** (bukan memakai library
   `pytorch-forecasting`) menggunakan PyTorch (Paszke et al., 2019),
   mengikuti spesifikasi komponen pada Lim et al. (2021): Gated Residual
   Network memakai GLU (Dauphin et al., 2017), ELU (Clevert et al., 2016),
   dan Layer Normalization (Ba et al., 2016); attention memakai
   implementasi scaled dot-product multi-head attention standar
   (Vaswani et al., 2017) dengan causal mask tambahan agar bersifat
   "masked interpretable multi-head attention" sesuai Gambar 1 proposal.
2. Karena tugas adalah klasifikasi (bukan quantile forecasting), *output
   head* TFT asli diganti menjadi lapisan `Linear` + softmax dengan
   cross-entropy loss.
3. Hyperparameter (hidden_size, dropout, attn_heads, learning rate) dicari
   menggunakan Optuna (Akiba et al., 2019) dengan objektif F1-macro pada
   data validasi.
