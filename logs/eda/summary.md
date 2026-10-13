# Ringkasan EDA UNSW-NB15

- Baris mentah: **12,120**; duplikat: **111**; setelah filter ['Normal', 'Generic', 'Exploits', 'Fuzzers'] & cleaning: **10,868**
- Statistik dihitung sebelum imputasi (NaN tidak diisi) agar kondisi mentah tetap terlihat.

## Distribusi kelas

| Kelas | Jumlah | Persen |
|---|---:|---:|
| Normal | 7,247 | 66.68% |
| Generic | 1,873 | 17.23% |
| Exploits | 946 | 8.70% |
| Fuzzers | 802 | 7.38% |

Imbalance ratio (max:min) = **9.0**

## Kolom dengan null terbanyak (sebelum imputasi)

```
                  null_count  null_pct  inf_count
sbytes                 242.0     1.997        0.0
ct_flw_http_mthd       200.0     1.650        0.0
sport                  148.0     1.221        0.0
dur                      0.0     0.000        0.0
dbytes                   0.0     0.000        0.0
sttl                     0.0     0.000        0.0
dttl                     0.0     0.000        0.0
sloss                    0.0     0.000        0.0
dloss                    0.0     0.000        0.0
sload                    0.0     0.000      151.0
```

## Output

- histograms/, boxplots/, histograms_grid.png, correlation_heatmap.png, describe_overall.csv, describe_per_class.csv, data_quality_report.csv, correlation_matrix.csv
