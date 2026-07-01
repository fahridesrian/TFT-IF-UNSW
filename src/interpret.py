"""
interpret.py
Ekstraksi interpretabilitas model sesuai proposal Bagian 6.j:
  - feature importance dari Variable Selection Network (VSN)
  - attention weight untuk memahami pola temporal paling relevan
  - kontribusi skor anomali IF (kolom fitur terakhir bila model TFT-IF)

PENTING: model harus dalam mode eval() saat forward pass di sini, karena
dalam mode train() dropout ikut diterapkan pada attention weight sehingga
baris bobotnya tidak lagi berjumlah 1 (sudah diverifikasi lewat smoke test).
"""
import numpy as np
import torch

from config import TRAIN


@torch.no_grad()
def extract_interpretation(model, X_sample: np.ndarray, feature_names: list, device: str = None):
    device = device or (TRAIN.device if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)

    xb = torch.tensor(X_sample, dtype=torch.float32).to(device)
    _ = model(xb)
    interp = model.get_last_interpretation()

    vsn_weights = interp["vsn_weights"].cpu().numpy()   # (N, W, F)
    attn_weights = interp["attn_weights"].cpu().numpy()  # (N, W+T, W+T)

    mean_feature_importance = vsn_weights.mean(axis=(0, 1))  # (F,)
    ranked = sorted(
        zip(feature_names, mean_feature_importance.tolist()),
        key=lambda kv: kv[1], reverse=True,
    )

    mean_attention_over_time = attn_weights.mean(axis=0)  # (W+T, W+T)

    return {
        "feature_importance_ranked": ranked,
        "vsn_weights_raw": vsn_weights,
        "mean_attention_matrix": mean_attention_over_time,
    }


def summarize_if_contribution(interp_with_if: dict, if_feature_name: str = "if_score"):
    """Ambil ranking & skor pentingnya fitur skor anomali IF secara spesifik
    (proposal Bagian 6.j.iv)."""
    ranked = interp_with_if["feature_importance_ranked"]
    for rank, (name, score) in enumerate(ranked, start=1):
        if name == if_feature_name:
            return {"rank": rank, "importance_score": score, "total_features": len(ranked)}
    return None
