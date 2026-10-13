# Hasil cek data UNSW-NB15

## 1. Rentang waktu per berkas (UTC, semua kelas)
| Berkas | Rekaman | stime min | stime max | sudah terurut |
|---|---|---|---|---|
| UNSW-NB15_1.csv | 700,001 | 2015-01-22 11:49:37 | 2015-01-22 19:44:02 | tidak |
| UNSW-NB15_2.csv | 700,001 | 2015-01-22 19:43:15 | 2015-02-18 03:45:29 | tidak |
| UNSW-NB15_3.csv | 700,001 | 2015-02-18 03:44:59 | 2015-02-18 09:00:09 | tidak |
| UNSW-NB15_4.csv | 440,044 | 2015-02-18 08:59:44 | 2015-02-18 12:21:08 | tidak |

- UNSW-NB15_1.csv dan UNSW-NB15_2.csv: TUMPANG TINDIH (selisih -0.0 jam antara akhir yang pertama dan awal yang kedua)
- UNSW-NB15_2.csv dan UNSW-NB15_3.csv: TUMPANG TINDIH (selisih -0.0 jam antara akhir yang pertama dan awal yang kedua)
- UNSW-NB15_3.csv dan UNSW-NB15_4.csv: TUMPANG TINDIH (selisih -0.0 jam antara akhir yang pertama dan awal yang kedua)

## 2. Jumlah rekaman empat kelas per berkas
| Berkas | Normal | Generic | Exploits | Fuzzers | kelas lain (dibuang) |
|---|---|---|---|---|---|
| UNSW-NB15_1.csv | 677,786 | 7,522 | 5,409 | 5,051 | 4,233 |
| UNSW-NB15_2.csv | 647,252 | 27,883 | 11,103 | 4,668 | 9,095 |
| UNSW-NB15_3.csv | 542,576 | 118,198 | 16,574 | 9,137 | 13,516 |
| UNSW-NB15_4.csv | 351,150 | 61,878 | 11,439 | 5,390 | 10,187 |

## 3. Pembagian 70:15:15 per berkas (meniru kode)
| Berkas | Split | Rekaman | Normal | Generic | Exploits | Fuzzers | stime min | stime max |
|---|---|---|---|---|---|---|---|---|
| UNSW-NB15_1.csv | train | 487,037 | 469,055 | 7,522 | 5,409 | 5,051 | 2015-01-22 11:49:37 | 2015-01-22 16:56:00 |
| UNSW-NB15_1.csv | val | 104,365 | 104,365 | 0 | 0 | 0 | 2015-01-22 16:56:00 | 2015-01-22 18:13:52 |
| UNSW-NB15_1.csv | test | 104,366 | 104,366 | 0 | 0 | 0 | 2015-01-22 18:13:53 | 2015-01-22 19:44:02 |
| UNSW-NB15_2.csv | train | 483,634 | 470,824 | 7,535 | 3,910 | 1,365 | 2015-01-22 19:43:15 | 2015-02-18 01:51:34 |
| UNSW-NB15_2.csv | val | 103,635 | 89,394 | 10,214 | 2,290 | 1,737 | 2015-02-18 01:51:34 | 2015-02-18 02:51:41 |
| UNSW-NB15_2.csv | test | 103,637 | 87,034 | 10,134 | 4,903 | 1,566 | 2015-02-18 02:51:41 | 2015-02-18 03:45:29 |
| UNSW-NB15_3.csv | train | 480,539 | 377,756 | 83,251 | 13,155 | 6,377 | 2015-02-18 03:45:07 | 2015-02-18 07:31:49 |
| UNSW-NB15_3.csv | val | 102,972 | 77,536 | 23,248 | 1,328 | 860 | 2015-02-18 07:31:49 | 2015-02-18 08:05:30 |
| UNSW-NB15_3.csv | test | 102,974 | 87,284 | 11,699 | 2,091 | 1,900 | 2015-02-18 08:05:30 | 2015-02-18 09:00:09 |
| UNSW-NB15_4.csv | train | 300,899 | 251,095 | 36,653 | 9,299 | 3,852 | 2015-02-18 08:59:44 | 2015-02-18 11:30:36 |
| UNSW-NB15_4.csv | val | 64,478 | 48,902 | 13,817 | 994 | 765 | 2015-02-18 11:30:36 | 2015-02-18 11:55:04 |
| UNSW-NB15_4.csv | test | 64,480 | 51,153 | 11,408 | 1,146 | 773 | 2015-02-18 11:55:04 | 2015-02-18 12:21:08 |

Total per split:
- train: 1,752,109 rekaman; Normal=1,568,730, Generic=134,961, Exploits=31,773, Fuzzers=16,645
- val: 375,450 rekaman; Normal=320,197, Generic=47,279, Exploits=4,612, Fuzzers=3,362
- test: 375,457 rekaman; Normal=329,837, Generic=33,241, Exploits=8,140, Fuzzers=4,239
- Kelas dengan kurang dari 1.000 rekaman pada suatu split: tidak ada
- Data latih paling akhir: 2015-02-18 11:30:36; data uji paling awal: 2015-01-22 18:13:53. Ada data uji yang LEBIH AWAL daripada sebagian data latih (urutan global tidak terjaga).

## 4. Persistensi label pada urutan stime (empat kelas, per berkas)
| Berkas | Pasangan berurutan sekelas | Panjang run rata-rata | Akurasi kelas mayoritas | Akurasi 'sama dengan rekaman terakhir' |
|---|---|---|---|---|
| UNSW-NB15_1.csv | 97.65% | 42.62 | 97.42% | 97.65% |
| UNSW-NB15_2.csv | 96.36% | 27.48 | 93.68% | 96.36% |
| UNSW-NB15_3.csv | 92.90% | 14.09 | 79.04% | 92.90% |
| UNSW-NB15_4.csv | 92.63% | 13.58 | 81.69% | 92.63% |

Bacaan: jika 'sama dengan rekaman terakhir' jauh di atas kelas mayoritas, label sangat persisten dan perlu dibahas bahwa pembanding tersebut harus dikalahkan model.
