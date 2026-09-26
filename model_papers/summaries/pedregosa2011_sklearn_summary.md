# Summary — Pedregosa et al. (2011), *Scikit-learn: Machine Learning in Python*

**File sumber**: `model_papers/pedregosa11a.pdf` (JMLR 12, 2011)
**Peran dalam proyek**: library yang menyediakan **empat komponen inti pipeline**:
`LabelEncoder` & `MinMaxScaler` (`src/preprocessing.py`), `IsolationForest`
(`src/isolation_forest_module.py`), dan seluruh metrik evaluasi (`src/evaluate.py`).

> Catatan presisi sitasi (penting untuk skripsi): paper 2011 ini mendokumentasikan
> **desain library**, bukan tiap estimator. `IsolationForest` baru masuk scikit-learn
> di v0.18 (2016) — algoritmenya sendiri tetap merujuk **Liu et al. 2008/2012**
> (sudah ada di daftar pustaka proposal). Jadi di skripsi: sitasi Pedregosa et al.
> untuk library/implementasi, Liu et al. untuk algoritmenya.

## 1. Konsep paper yang menempel ke proyek

- **Antarmuka estimator/transformer** (§4): objek dengan `fit` → `transform`/`predict`.
  Inilah pola yang membuat aturan anti-*data leakage* proyek bersih secara implementasi:
  `LabelEncoder` dan `MinMaxScaler` **di-fit hanya pada data latih**, lalu `transform`
  diterapkan ke data validasi & uji (proposal Bagian 6.d.v–vii); `IsolationForest`
  **di-fit hanya pada subset Normal data latih**, lalu skor dihitung untuk semua split
  (Bagian 6.f).
- **Konsistensi API & solid implementation** (§2): alasan praktis memakai implementasi
  bawaan ketimbang menulis sendiri untuk komponen non-inti (praproses, IF, metrik),
  sementara bagian inti penelitian (TFT) tetap custom.

## 2. Pemetaan ke kode

| Komponen scikit-learn | Di proyek | Nilai/aturan |
|---|---|---|
| `LabelEncoder` | encoding `proto`, `state`, `service` (fit di train) | proposal Bagian 6.d.v |
| `MinMaxScaler` | normalisasi fitur numerik (fit di train) | proposal Bagian 6.d.vi |
| `IsolationForest` | skor anomali kontinu, fit pada Normal-train | `n_estimators=100`, `max_samples="auto"` (= sub-sampel 256/pohon, sesuai rekomendasi Liu et al. 2008), `contamination="auto"` (tidak dipakai untuk threshold — skor kontinu yang jadi fitur), `random_state=42` |
| Metrik (`accuracy_score`, `precision_recall_fscore_support`, `confusion_matrix`) | `evaluate.py` | accuracy, precision/recall/F1 macro & weighted, FAR agregasi Normal-vs-Attack, confusion matrix 4×4 |

## 3. Bagian paper yang TIDAK dipakai (tidak perlu dibuka ulang)

- §4–5: `GridSearchCV`/`Pipeline` (proyek memakai Optuna, bukan grid search sklearn),
  benchmark SVM/LARS/ElasticNet/kNN/PCA/k-means (Tabel 1).
- §2–3: lisensi BSD, ekosistem NumPy/SciPy/Cython — konteks umum saja.
- Catatan: paper ini sama sekali tidak membahas IsolationForest atau metrik detail —
  bagian §4.1–2 di atas cukup sebagai landasan sitasi.

## Kutipan (APA — sudah terdaftar di `references_addendum.md`)

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O.,
Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A.,
Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, É. (2011). Scikit-learn:
Machine learning in Python. Journal of Machine Learning Research, 12, 2825–2830.
