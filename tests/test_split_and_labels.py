"""Uji cepat (data tiruan; jalankan make_dummy_data.py dulu):
pembagian global kronologis dan pemetaan label Normal=0.
    python tests/test_split_and_labels.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from config import DATA
from data_loading import load_unsw_nb15
from preprocessing import (
    filter_classes, clean_data, sort_chronologically, chronological_split,
    run_preprocessing,
)

raw = load_unsw_nb15(os.path.join(ROOT, "data", "raw"))

# 1. Split global: semua stime latih <= val <= uji.
df = sort_chronologically(clean_data(filter_classes(raw)))
tr, va, te = chronological_split(df)
t = DATA.time_cols[0]
assert tr[t].max() <= va[t].min() and va[t].max() <= te[t].min(), "split tidak kronologis global"
assert len(tr) + len(va) + len(te) == len(df)

# 2. Pemetaan label sesuai urutan classes_used.
tr, va, te, art = run_preprocessing(raw)
assert list(art.label_encoder.classes_) == list(DATA.classes_used)
counts = tr[DATA.target_col].value_counts()
assert counts.idxmax() == DATA.classes_used.index("Normal") == 0, "Normal harus kode 0"
print("OK")
