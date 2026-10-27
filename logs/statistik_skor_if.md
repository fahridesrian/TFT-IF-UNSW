# Statistik skor anomali Isolation Forest

Banyak fitur masukan IF: 41. Pohon: 100. max_samples efektif (psi): 256. random_state: 42.
Rekaman Normal data latih (dipakai melatih IF): 1387816. Batas atas rekaman yang pernah masuk satu pohon: 25600 (1.84%).

| Bagian | Kelas | Jumlah | Rata-rata | Median | Persentil 5 | Persentil 95 |
|---|---|---|---|---|---|---|
| Latih | Normal | 1387816 | 0.3970 | 0.3776 | 0.3490 | 0.5214 |
| Latih | Generic | 11360 | 0.6061 | 0.6071 | 0.5375 | 0.6786 |
| Latih | Exploits | 13798 | 0.5850 | 0.5874 | 0.5279 | 0.6430 |
| Latih | Fuzzers | 11201 | 0.5901 | 0.5913 | 0.5279 | 0.6502 |
| Validasi | Normal | 285668 | 0.4037 | 0.3779 | 0.3488 | 0.5845 |
| Validasi | Generic | 7105 | 0.6330 | 0.6399 | 0.5601 | 0.6804 |
| Validasi | Exploits | 6845 | 0.5957 | 0.5976 | 0.5363 | 0.6538 |
| Validasi | Fuzzers | 5562 | 0.5947 | 0.5962 | 0.5273 | 0.6541 |
| Uji | Normal | 286280 | 0.4033 | 0.3790 | 0.3491 | 0.5846 |
| Uji | Generic | 6913 | 0.6222 | 0.6308 | 0.5449 | 0.6767 |
| Uji | Exploits | 6956 | 0.5929 | 0.5965 | 0.5337 | 0.6500 |
| Uji | Fuzzers | 5032 | 0.5894 | 0.5917 | 0.5265 | 0.6559 |

Kemampuan skor itu sendiri membedakan Normal (0) dan serangan (1):

| Bagian | ROC-AUC | PR-AUC | Proporsi serangan |
|---|---|---|---|
| Latih | 0.9794 | 0.4376 | 0.0255 |
| Validasi | 0.9683 | 0.5919 | 0.0639 |
| Uji | 0.9645 | 0.5200 | 0.0619 |
