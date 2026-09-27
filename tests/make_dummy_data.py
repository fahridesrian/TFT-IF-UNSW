"""Membuat 4 CSV dummy meniru struktur UNSW-NB15 untuk pengujian pipeline
end-to-end (bukan untuk eksperimen riil). Menyisipkan null/inf/duplikat
secara sengaja untuk menguji cleaning."""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"
))
from config import DATA

rng = np.random.default_rng(42)
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")
os.makedirs(OUT_DIR, exist_ok=True)

CLASSES_ALL = ["Normal", "Generic", "Exploits", "Fuzzers", "DoS", "Reconnaissance",
               "Analysis", "Backdoors", "Shellcode", "Worms"]
CLASS_PROBS = [0.6, 0.15, 0.08, 0.07, 0.03, 0.03, 0.015, 0.01, 0.004, 0.001]
CLASS_PROBS = [p / sum(CLASS_PROBS) for p in CLASS_PROBS]

PROTOS = ["tcp", "udp", "arp", "ospf"]
STATES = ["FIN", "CON", "INT", "REQ"]
SERVICES = ["-", "http", "dns", "ftp", "smtp"]


def make_file(n_rows: int, start_time: int, fname: str):
    t = np.sort(start_time + rng.integers(0, 5000, size=n_rows).cumsum())
    df = pd.DataFrame({
        "srcip": [f"10.0.0.{i%255}" for i in range(n_rows)],
        "sport": rng.integers(1024, 65535, n_rows),
        "dstip": [f"192.168.1.{i%255}" for i in range(n_rows)],
        "dsport": rng.integers(1, 1024, n_rows),
        "proto": rng.choice(PROTOS, n_rows),
        "state": rng.choice(STATES, n_rows),
        "dur": rng.exponential(1.0, n_rows),
        "sbytes": rng.integers(0, 10000, n_rows).astype(float),
        "dbytes": rng.integers(0, 10000, n_rows).astype(float),
        "sttl": rng.integers(0, 255, n_rows),
        "dttl": rng.integers(0, 255, n_rows),
        "sloss": rng.integers(0, 10, n_rows),
        "dloss": rng.integers(0, 10, n_rows),
        "service": rng.choice(SERVICES, n_rows),
        "sload": rng.exponential(100, n_rows),
        "dload": rng.exponential(100, n_rows),
        "spkts": rng.integers(0, 100, n_rows),
        "dpkts": rng.integers(0, 100, n_rows),
        "swin": rng.integers(0, 65535, n_rows),
        "dwin": rng.integers(0, 65535, n_rows),
        "stcpb": rng.integers(0, 2**31, n_rows),
        "dtcpb": rng.integers(0, 2**31, n_rows),
        "smeansz": rng.integers(0, 1500, n_rows),
        "dmeansz": rng.integers(0, 1500, n_rows),
        "trans_depth": rng.integers(0, 5, n_rows),
        "res_bdy_len": rng.integers(0, 5000, n_rows),
        "sjit": rng.exponential(1, n_rows),
        "djit": rng.exponential(1, n_rows),
        "stime": t,
        "ltime": t + rng.integers(0, 5, n_rows),
        "sintpkt": rng.exponential(1, n_rows),
        "dintpkt": rng.exponential(1, n_rows),
        "tcprtt": rng.exponential(0.1, n_rows),
        "synack": rng.exponential(0.05, n_rows),
        "ackdat": rng.exponential(0.05, n_rows),
        "is_sm_ips_ports": rng.integers(0, 2, n_rows),
        "ct_state_ttl": rng.integers(0, 10, n_rows),
        "ct_flw_http_mthd": rng.integers(0, 5, n_rows),
        "is_ftp_login": rng.integers(0, 2, n_rows),
        "ct_ftp_cmd": rng.integers(0, 5, n_rows),
        "ct_srv_src": rng.integers(0, 20, n_rows),
        "ct_srv_dst": rng.integers(0, 20, n_rows),
        "ct_dst_ltm": rng.integers(0, 20, n_rows),
        "ct_src_ltm": rng.integers(0, 20, n_rows),
        "ct_src_dport_ltm": rng.integers(0, 20, n_rows),
        "ct_dst_sport_ltm": rng.integers(0, 20, n_rows),
        "ct_dst_src_ltm": rng.integers(0, 20, n_rows),
    })
    attack_cat = rng.choice(CLASSES_ALL, n_rows, p=CLASS_PROBS)
    df["attack_cat"] = attack_cat
    df.loc[df["attack_cat"] == "Normal", "attack_cat"] = "Normal"
    df["label"] = (df["attack_cat"] != "Normal").astype(int)

    # sisipkan null & inf sengaja
    null_idx = rng.choice(n_rows, size=max(1, n_rows // 50), replace=False)
    df.loc[null_idx, "sbytes"] = np.nan
    inf_idx = rng.choice(n_rows, size=max(1, n_rows // 80), replace=False)
    df.loc[inf_idx, "sload"] = np.inf
    # sisipkan duplikat sengaja
    dup_rows = df.sample(n=max(1, n_rows // 100), random_state=1)
    df = pd.concat([df, dup_rows], ignore_index=True)
    # attack_cat kosong -> harus jadi Normal setelah preprocessing. Di
    # UNSW-NB15 asli baris seperti ini selalu berlabel 0; pilih hanya baris
    # label=0 agar konsisten dengan assertion di preprocessing.filter_classes.
    normal_candidates = df.index[df["label"] == 0].to_numpy()
    n_empty = max(1, len(df) // 200)
    empty_idx = rng.choice(
        normal_candidates, size=min(n_empty, len(normal_candidates)), replace=False
    )
    df.loc[empty_idx, "attack_cat"] = np.nan

    # --- reproduksi bug asli: kolom numerik berisi string liar ---
    stray_idx = rng.choice(len(df), size=max(1, len(df) // 60), replace=False)
    df["ct_flw_http_mthd"] = df["ct_flw_http_mthd"].astype(object)
    df.loc[stray_idx, "ct_flw_http_mthd"] = " "  # bare space, seperti UNSW-NB15 asli
    hex_idx = rng.choice(len(df), size=max(1, len(df) // 80), replace=False)
    df["sport"] = df["sport"].astype(object)
    df.loc[hex_idx, "sport"] = "0x000c"  # port dalam format hex
    # --- reproduksi bug casing/whitespace pada attack_cat ---
    case_idx = rng.choice(
        df.index[df["attack_cat"] == "Fuzzers"], size=5, replace=False
    ) if (df["attack_cat"] == "Fuzzers").sum() >= 5 else []
    for i in case_idx:
        df.loc[i, "attack_cat"] = " fuzzers "

    df = df[list(DATA.columns)]
    df.to_csv(os.path.join(OUT_DIR, fname), index=False)
    print(f"  wrote {fname}: {len(df)} rows")


if __name__ == "__main__":
    make_file(3000, 1_000_000, "UNSW-NB15_1.csv")
    make_file(3000, 1_100_000, "UNSW-NB15_2.csv")
    make_file(3000, 1_200_000, "UNSW-NB15_3.csv")
    make_file(3000, 1_300_000, "UNSW-NB15_4.csv")
    print("Dummy data generated at", OUT_DIR)
