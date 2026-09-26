# Summary — Micikevicius, Narang et al. (2018), *Mixed Precision Training*

**File sumber**: `model_papers/1710.03740v3.pdf` (arXiv:1710.03740v3, ICLR 2018)
**Peran dalam proyek**: dasar **AMP (Automatic Mixed Precision)** pada training loop
`src/train.py` — `torch.cuda.amp.autocast` + `GradScaler` untuk melatih TFT-base &
TFT-IF di RTX 3060 12GB.

> Catatan penulisan: halaman judul PDF mencantumkan Narang dkk. (Baidu) lebih dulu,
> tetapi kedua penulis pertama berkontribusi setara dan daftar pustaka proyek
> (`references_addendum.md`) memakai urutan Micikevicius dulu — summary ini mengikuti
> urutan daftar pustaka proyek.

## 1. Tiga teknik inti paper (§3) — semua diadopsi implisit lewat PyTorch AMP

1. **Master copy bobot tetap FP32** (§3.1): forward/backward memakai FP16, tetapi
   update optimizer dihitung dan disimpan dalam FP32.
   *Alasan*: update (gradien × learning rate) yang lebih kecil dari 2⁻²⁴ menjadi nol
   di FP16; juga rasio bobot : update > 2048 membuat update hilang saat penjumlahan.
2. **Loss scaling** (§3.2): kalikan loss sebelum backward agar semua gradien tergeser
   ke rentang representable FP16 (eksponen efektif [−14, 15]); gradien **di-unscale
   setelah backward, sebelum step optimizer** (dan sebelum operasi gradien lain).
   *Alasan*: gradien praktis didominasi magnitudo kecil yang underflow → nol.
3. **Aritmetika FP16 dengan akumulasi FP32** (§3.3): dot-product mengakumulasi hasil
   parsial ke FP32; reduksi besar (statistik normalisasi, softmax) dihitung FP32.

## 2. Pemetaan ke kode

| Paper | Di proyek |
|---|---|
| FP32 master weights + loss scaling + skip update saat overflow | `torch.cuda.amp.GradScaler` — mendeteksi inf/NaN, **melewati step** saat overflow, dan menyesuaikan faktor skala secara dinamis (persis opsi yang diusulkan paper §3.2/§5) |
| FP16 arithmetic, FP32 accumulation | `torch.autocast` di sekitar forward/backward; model tetap FP32 |
| Faktor skala konstan 8–32K pada paper | PyTorch memakai *dynamic* scaling (init 65536, growth ×2) — superset dari rekomendasi paper |
| — | **GradScaler dibuat sekali di luar loop epoch** (fix di `handoff.md`) agar adaptasi loss-scale tidak hilang tiap epoch |

Dampak yang relevan: konsumsi memori training ~setengahnya (aktivasi FP16) dan
throughput lebih tinggi via Tensor Cores — memberi ruang untuk `batch_size=1024`
di GPU 12GB. Klaim kunci paper: **tanpa mengubah hyperparameter apa pun** dibanding
FP32 (lr, weight_decay, dll. tetap seperti tabel Adam summary).

## 3. Rasional yang bisa dikutip di skripsi

- AMP dipakai murni sebagai **optimasi komputasi**, bukan bagian metodologi deteksi:
  paper menunjukkan akurasi setara FP32 di berbagai arsitektur termasuk **RNN/LSTM**
  (§4.3–4.5) tanpa tuning hyperparameter — sehingga perbandingan TFT-base vs TFT-IF
  tidak terdistorsi oleh presisi numerik.
- Komponen sensitif presisi (LayerNorm/softmax) dihitung FP32 oleh autocast,
  konsisten dengan §3.3 paper.

## 4. Bagian paper yang TIDAK dipakai (tidak perlu dibuka ulang)

- §2: kajian quantization (biner/ternary/fixed-point) — konteks saja.
- §4.1–4.6: seluruh eksperimen (ILSVRC CNN, Faster R-CNN/SSD, DeepSpeech 2,
  machine translation, bigLSTM, DCGAN) — cukup dikutip hasil umumnya.
- DeepBench benchmark (§5) dan diskusi framework.

## Kutipan (APA — sudah terdaftar di `references_addendum.md`)

Micikevicius, P., Narang, S., Alben, J., Diamos, G., Elsen, E., Garcia, D.,
Ginsburg, B., Houston, M., Kuchaiev, O., Venkatesh, G., & Wu, H. (2018). Mixed
precision training. International Conference on Learning Representations (ICLR).
https://arxiv.org/abs/1710.03740
