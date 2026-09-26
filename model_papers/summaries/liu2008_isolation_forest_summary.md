# Summary — Liu, Ting & Zhou (2008), *Isolation Forest*  【MAIN PAPER 2】

**File sumber**: `model_papers/liu2008.pdf` (IEEE ICDM 2008, DOI 10.1109/ICDM.2008.17)
**Peran dalam proyek**: **paper inti kedua** — algoritme penghasil skor anomali yang
diimplementasikan via `sklearn.ensemble.IsolationForest` di
`src/isolation_forest_module.py`; dikutip di proposal Bagian 5.3 (Persamaan 2 & 3
proposal = Persamaan 2 & 1 paper ini) dan Bagian 6.f.

> Catatan: proposal juga mengutip versi jurnal Liu et al. 2012 (TKDD,
> "Isolation-based anomaly detection") yang memformalkan analisis paper ini —
> PDF yang diringkas di sini versi konferensi 2008. sklearn mengimplementasikan
> keduanya; tingkat analisis di bawah sudah cukup untuk kebutuhan implementasi.

## 1. Yang dipakai: algoritme & skor anomali

### a. Konsep isolasi (§1–2)
Anomali bersifat "**few and different**" → lebih cepat terisolasi oleh partisi acak →
path length pendek → anomali. Berbeda dari metode profiling normal / distance-based /
density-based: tanpa perhitungan jarak, kompleksitas linear, cocok untuk data besar —
alasan praktis pemilihan IF pada 2,5 juta rekaman UNSW-NB15.

### b. Skor anomali (Eq. 1–2 paper = Eq. 3 & 2 proposal)
- `c(n) = 2H(n−1) − 2(n−1)/n`, dengan `H(i) ≈ ln(i) + 0.5772156649` (Euler–Mascheroni)
  — normalisasi path length, setara rata-rata unsuccessful search BST.
- `s(x, n) = 2^{−E[h(x)]/c(n)}`:
  `s → 1` anomali; `s ≈ 0.5` tidak dapat dibedakan; `s → 0` normal.
- Path length `h(x)` = jumlah edge root→external node; **penyesuaian `c(Size)`
  ditambahkan bila traversal berhenti di external node berisi >1 instan** (Algorithm 3,
  `PathLength` → `return e + c(T.size)`) — mengoreksi subtree yang tak terbangun karena
  height limit.
- Proyek memakai skor **kontinu** `s` sebagai fitur ke-(F+1) — **tanpa thresholding**.
  (Contoh ambang `s ≥ 0.6` di §2/Fig. 3 paper bersifat ilustratif untuk visualisasi
  kontur, bukan bagian algoritme — tidak diadopsi proyek.)

### c. Dua parameter algoritme (§4.1) → config proyek

| Paper | Rekomendasi & temuan | Di proyek (`IFConfig`) |
|---|---|---|
| `t` = jumlah pohon | Path length konvergen **sebelum t = 100**; default `t = 100` | `n_estimators = 100` ✓ default paper |
| `ψ` = ukuran sub-sampel | **ψ = 256 (2⁸)** secara empiris cukup dan near-optimal; kinerja tak sensitif pada rentang lebar ψ; sub-sampling menekan efek *swamping* & *masking* (§3) | `max_samples = "auto"` (= 256 per pohon di sklearn) ✓ default paper |
| Height limit | `l = ceiling(log₂ ψ)` otomatis | otomatis di sklearn |
| — | — | `contamination = "auto"`: **tidak ada di paper 2008** — konsep sklearn untuk thresholding; proyek tidak memakai threshold sehingga parameter ini efeknya nol |
| Kompleksitas | Training `O(tψ log ψ)`, evaluasi `O(n·t·log ψ)` | justifikasi skalabilitas ke jutaan rekaman |

## 2. Temuan paper yang paling krusial untuk desain skripsi

**§5.4 "Training using normal instances only"** — eksperimen pelatihan **tanpa satu pun
anomali** (Http & ForestCover): penurunan AUC kecil (0.9997→0.9919; 0.8817→0.8802), dan
**dapat dipulihkan dengan memperbesar ψ** (256 → 8192/512).
→ Ini **justifikasi empiris langsung** untuk keputusan inti proyek/proposal: melatih IF
*unsupervised* **hanya pada subset kelas Normal data latih** (proposal Bagian 6.f.i dan
penjelasan baris 398: skor merepresentasikan penyimpangan terhadap pola lalu lintas
normal). Bila ingin menanggapi pertanyaan penguji soal validitas normal-only training,
bagian inilah rujukannya.

## 3. Rasional tambahan yang bisa dikutip di skripsi

- Isolasi (bukan profiling) menghindari dua kelemahan metode profiling: detektor
  dioptimalkan untuk profil normal, bukan untuk mendeteksi anomali (§1).
- Sub-sampling kecil memberi AUC tinggi + waktu rendah (§5.2, Fig. 6) — mendukung
  efisiensi pada dataset besar.
- IF bekerja baik pada data berdimensi tinggi dengan banyak atribut tidak relevan
  (§5.3; opsi tambahan attribute selector seperti Kurtosis **tidak** dipakai proyek).

## 4. Bagian paper yang TIDAK dipakai (tidak perlu dibuka ulang)

- §5.1 perbandingan ORCA/LOF/RF (Tabel 2–3) dan seluruh eksperimen benchmark —
  cukup dikutip kesimpulan umumnya (unggul dalam AUC & waktu pada data besar).
- §5.3 eksperimen Kurtosis attribute selector (tidak direplikasi).
- §6 diskusi online anomaly detection system.

## Kutipan (APA — sudah ada di daftar pustaka proposal)

Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). Isolation forest. Proceedings - IEEE
International Conference on Data Mining, ICDM, 413–422.
https://doi.org/10.1109/ICDM.2008.17
