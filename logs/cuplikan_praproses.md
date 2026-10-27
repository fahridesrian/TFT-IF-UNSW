# Cuplikan dan angka praproses (untuk Subbab 3.4)

## 0. Data mentah (setelah dibaca dari empat CSV)
- Bentuk: 2,540,040 baris, 50 kolom (49 kolom data + `source_file`).
- Rekaman per berkas: UNSW-NB15_1.csv=700,001, UNSW-NB15_2.csv=700,001, UNSW-NB15_3.csv=700,001, UNSW-NB15_4.csv=440,037
- Nilai unik `attack_cat` (apa adanya, tanda kutip menunjukkan spasi/kosong):
  [' Fuzzers', ' Fuzzers ', ' Reconnaissance ', ' Shellcode ', 'Analysis', 'Backdoor', 'Backdoors', 'DoS', 'Exploits', 'Generic', 'Reconnaissance', 'Shellcode', 'Worms', nan]

**Cuplikan mentah (nilai `attack_cat` ditampilkan dengan repr)**

```
     proto state service       dur  sbytes   sttl dttl  ct_srv_dst         stime  attack_cat
0      udp   CON     dns  0.001055   132.0   31.0   29         7.0  1.421927e+09         nan
21     tcp   FIN    http  0.240139   918.0   62.0  252         2.0  1.421927e+09  'Exploits'
58     tcp   FIN       -  0.194910   844.0  254.0  252         1.0  1.421927e+09   'Generic'
4987  ospf   INT       -  0.518061  1040.0  254.0    0         2.0  1.421928e+09  'Exploits'
```


## 1. Penyaringan empat kelas
- Sebelum: 2,540,040 baris. Sesudah: 2,503,009 baris. Dibuang: 37,031 baris.
- Rekaman per kelas sesudah penyaringan: Normal=2,218,751, Generic=215,483, Exploits=44,527, Fuzzers=24,248
- Rekaman per kategori di data mentah (setelah trim spasi; kosong = Normal):
  Generic=215,483, Exploits=44,527, Fuzzers=24,248, DoS=16,353, Reconnaissance=13,987, Analysis=2,677, Backdoor=1,795, Shellcode=1,511, Backdoors=534, Worms=174
**Cuplikan sesudah penyaringan (label sudah rapi dan Normal terisi)**

```
     attack_cat         stime
0        Normal  1.421927e+09
21     Exploits  1.421927e+09
56      Generic  1.421927e+09
4919   Exploits  1.421928e+09
```

## 2. Pembersihan data
- Nilai hilang (NaN) sesudah konversi numerik, per kolom (hanya yang > 0):
  ct_ftp_cmd=1,397,096, is_ftp_login=1,397,096, ct_flw_http_mthd=1,319,290, dsport=304, sport=8
- Nilai tak hingga (inf/-inf): 0 sel.
- Duplikat identik dibuang: 468,473 baris (2,503,009 menjadi 2,034,536).
- Rekaman per kelas sebelum -> sesudah dedup: Normal: 2,218,751 -> 1,959,764, Generic: 215,483 -> 25,378, Exploits: 44,527 -> 27,599, Fuzzers: 24,248 -> 21,795

## 3. Pengurutan kronologis
- `stime` unik: 85,348 dari 2,034,536 baris; baris yang berbagi `stime` dengan baris lain: 2,034,441 (100.0%).
- Aturan pemecah seri: `stime`, lalu berkas asal (`source_file`), lalu urutan baris asli (pengurutan stabil).
- Rentang waktu: 2015-01-22 11:49:37 sampai 2015-02-18 12:21:08 (UTC).

**Posisi cuplikan pada urutan global**

```
          source_file         stime attack_cat  posisi_urut
4     UNSW-NB15_1.csv  1.421927e+09     Normal            4
41    UNSW-NB15_1.csv  1.421927e+09   Exploits           41
118   UNSW-NB15_1.csv  1.421927e+09    Generic          118
4871  UNSW-NB15_1.csv  1.421928e+09   Exploits         4871
```

## 4. Pembagian 70:15:15
- latih: 1,424,175 baris; 2015-01-22 11:49:37 sampai 2015-02-18 05:33:34; Normal=1,387,816, Exploits=13,798, Generic=11,360, Fuzzers=11,201; berkas asal: 1=637,376, 2=631,132, 3=155,667
- validasi: 305,180 baris; 2015-02-18 05:33:34 sampai 2015-02-18 08:56:45; Normal=285,668, Generic=7,105, Exploits=6,845, Fuzzers=5,562; berkas asal: 3=305,180
- uji: 305,181 baris; 2015-02-18 08:56:45 sampai 2015-02-18 12:21:08; Normal=286,280, Exploits=6,956, Generic=6,913, Fuzzers=5,032; berkas asal: 3=5,015, 4=300,166

- Rekaman 1 (Normal): data latih, baris ke-4
- Rekaman 2 (Exploits): data latih, baris ke-41
- Rekaman 3 (Generic): data latih, baris ke-118
- Rekaman 4 (Fuzzers): data latih, baris ke-4871

## 5. Imputasi median (fit pada data latih)
| kolom | NaN latih | NaN validasi | NaN uji | median latih |
|---|---|---|---|---|
| sport | 8 | 0 | 0 | 32601.0 |
| dsport | 160 | 68 | 67 | 3561.0 |
| ct_flw_http_mthd | 366,633 | 276,420 | 272,822 | 0.0 |
| is_ftp_login | 395,533 | 298,492 | 298,695 | 0.0 |
| ct_ftp_cmd | 395,533 | 298,492 | 298,695 | 0.0 |

## 6. Encoding kategori dan 7. Normalisasi Min-Max
- Fitur akhir: 41 (kolom: proto, state, dur, sbytes, dbytes, sttl, dttl, sloss, dloss, service, sload, dload, spkts, dpkts, swin, dwin, stcpb, dtcpb, smeansz, dmeansz, trans_depth, res_bdy_len, sjit, djit, sintpkt, dintpkt, tcprtt, synack, ackdat, is_sm_ips_ports, ct_state_ttl, ct_flw_http_mthd, is_ftp_login, ct_ftp_cmd, ct_srv_src, ct_srv_dst, ct_dst_ltm, ct_src_ltm, ct_src_dport_ltm, ct_dst_sport_ltm, ct_dst_src_ltm)
- Kategori per fitur (termasuk token 'unknown'): {'proto': 136, 'state': 17, 'service': 14}
- `proto`: 136 nilai; contoh urutan kode: 3pc=0, a/n=1, aes-sp3-d=2, any=3, argus=4, aris=5, arp=6, ax.25=7, bbn-rcc=8, bna=9, br-sat-mon=10, cbt=11 ...
  - baris validasi dengan nilai belum pernah muncul di data latih (dipetakan ke 'unknown'): 0
  - baris uji dengan nilai belum pernah muncul di data latih (dipetakan ke 'unknown'): 0
- `state`: 17 nilai; contoh urutan kode: ACC=0, CLO=1, CON=2, ECO=3, ECR=4, FIN=5, INT=6, MAS=7, PAR=8, REQ=9, RST=10, TST=11 ...
  - baris validasi dengan nilai belum pernah muncul di data latih (dipetakan ke 'unknown'): 0
  - baris uji dengan nilai belum pernah muncul di data latih (dipetakan ke 'unknown'): 0
- `service`: 14 nilai; contoh urutan kode: -=0, dhcp=1, dns=2, ftp=3, ftp-data=4, http=5, irc=6, pop3=7, radius=8, smtp=9, snmp=10, ssh=11 ...
  - baris validasi dengan nilai belum pernah muncul di data latih (dipetakan ke 'unknown'): 0
  - baris uji dengan nilai belum pernah muncul di data latih (dipetakan ke 'unknown'): 0

Minimum dan maksimum data latih untuk fitur contoh:
| fitur | min latih | maks latih |
|---|---|---|
| dur | 0.0 | 8786.637695 |
| sbytes | 0.0 | 14355774.0 |
| sttl | 0.0 | 255.0 |
| dttl | 0.0 | 254.0 |
| ct_srv_dst | 1.0 | 67.0 |

## Cuplikan akhir: nilai asli, sesudah imputasi, dan sesudah encoding + Min-Max
**Rekaman 1 (Normal)**, data latih, label akhir = 0

```
           sebelum (setelah imputasi) sesudah (encoding + Min-Max)
proto                             udp                   120.000000
state                             CON                     2.000000
service                           dns                     2.000000
dur                          0.001055                     0.000000
sbytes                          132.0                     0.000009
sttl                             31.0                     0.121569
dttl                               29                     0.114173
ct_srv_dst                        7.0                     0.090909
```

**Rekaman 2 (Exploits)**, data latih, label akhir = 2

```
           sebelum (setelah imputasi) sesudah (encoding + Min-Max)
proto                             tcp                   114.000000
state                             FIN                     5.000000
service                          http                     5.000000
dur                          0.240139                     0.000027
sbytes                          918.0                     0.000064
sttl                             62.0                     0.243137
dttl                              252                     0.992126
ct_srv_dst                        2.0                     0.015152
```

**Rekaman 3 (Generic)**, data latih, label akhir = 1

```
           sebelum (setelah imputasi) sesudah (encoding + Min-Max)
proto                             tcp                   114.000000
state                             FIN                     5.000000
service                             -                     0.000000
dur                           0.19491                     0.000022
sbytes                          844.0                     0.000059
sttl                            254.0                     0.996078
dttl                              252                     0.992126
ct_srv_dst                        1.0                     0.000000
```

**Rekaman 4 (Fuzzers)**, data latih, label akhir = 2

```
           sebelum (setelah imputasi) sesudah (encoding + Min-Max)
proto                            ospf                    79.000000
state                             INT                     6.000000
service                             -                     0.000000
dur                          0.518061                     0.000059
sbytes                         1040.0                     0.000072
sttl                            254.0                     0.996078
dttl                                0                     0.000000
ct_srv_dst                        2.0                     0.015152
```
