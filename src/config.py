"""
config.py
Semua parameter pipeline TFT + Isolation Forest untuk deteksi intrusi
UNSW-NB15 dikumpulkan di sini agar mudah diubah tanpa menyentuh kode inti.
"""
from dataclasses import dataclass, field
from typing import List, Tuple
import os


@dataclass
class DataConfig:
    raw_dir: str = "data/raw"
    csv_files: Tuple[str, ...] = (
        "UNSW-NB15_1.csv",
        "UNSW-NB15_2.csv",
        "UNSW-NB15_3.csv",
        "UNSW-NB15_4.csv",
    )
    # Kolom diasumsikan sudah punya header ini (sesuaikan bila CSV asli tanpa header).
    columns: Tuple[str, ...] = (
        "srcip", "sport", "dstip", "dsport", "proto", "state", "dur",
        "sbytes", "dbytes", "sttl", "dttl", "sloss", "dloss", "service",
        "sload", "dload", "spkts", "dpkts", "swin", "dwin", "stcpb", "dtcpb",
        "smeansz", "dmeansz", "trans_depth", "res_bdy_len", "sjit", "djit",
        "stime", "ltime", "sintpkt", "dintpkt", "tcprtt", "synack", "ackdat",
        "is_sm_ips_ports", "ct_state_ttl", "ct_flw_http_mthd", "is_ftp_login",
        "ct_ftp_cmd", "ct_srv_src", "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm",
        "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm",
        "attack_cat", "label",
    )
    id_cols_to_drop: Tuple[str, ...] = ("srcip", "sport", "dstip", "dsport")
    categorical_cols: Tuple[str, ...] = ("proto", "state", "service")
    time_cols: Tuple[str, ...] = ("stime", "ltime")
    target_col: str = "attack_cat"
    binary_label_col: str = "label"
    classes_used: Tuple[str, ...] = ("Normal", "Generic", "Exploits", "Fuzzers")
    source_file_col: str = "source_file"


@dataclass
class PreprocConfig:
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    # Imputasi null numerik selalu median & fit-on-train (impute_missing).


@dataclass
class IFConfig:
    n_estimators: int = 100
    max_samples: str = "auto"
    contamination: str = "auto"  # tak dipakai untuk thresholding, hanya skor kontinu
    random_state: int = 42


@dataclass
class WindowConfig:
    W: int = 10   # panjang jendela historis
    T: int = 3    # panjang jendela target (multi-horizon)
    stride: int = 1


@dataclass
class ModelConfig:
    hidden_size: int = 64
    lstm_layers: int = 1
    attn_heads: int = 4
    dropout: float = 0.1
    num_classes: int = 4


@dataclass
class TrainConfig:
    batch_size: int = 1024
    epochs: int = 30
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 5           # early stopping
    lr_scheduler_patience: int = 2
    lr_scheduler_factor: float = 0.5
    device: str = "cuda" if os.environ.get("FORCE_CPU") != "1" else "cpu"
    seed: int = 42
    num_workers: int = 6  # untuk DataLoader


@dataclass
class TuningConfig:
    n_trials: int = 20
    metric: str = "f1_macro"
    direction: str = "maximize"


@dataclass
class PathsConfig:
    checkpoints_dir: str = "checkpoints"
    logs_dir: str = "logs"
    results_file: str = "logs/results_summary.json"


DATA = DataConfig()
PREPROC = PreprocConfig()
IF = IFConfig()
WINDOW = WindowConfig()
MODEL = ModelConfig()
TRAIN = TrainConfig()
TUNING = TuningConfig()
PATHS = PathsConfig()
