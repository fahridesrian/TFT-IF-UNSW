"""
evaluate.py
Metrik evaluasi sesuai proposal Bagian 5.8: akurasi, precision, recall,
F1-score (macro & weighted), false alarm rate (FAR, agregasi Normal-vs-Attack
per Bagian 5.8.4), dan confusion matrix multikelas.
"""
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    roc_auc_score, average_precision_score,
)

from config import DATA, MODEL, TRAIN


@torch.no_grad()
def predict(model, X: np.ndarray, batch_size: int = 256, device: str = None):
    device = device or (TRAIN.device if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)
    preds = []
    for i in range(0, len(X), batch_size):
        xb = torch.tensor(X[i:i + batch_size], dtype=torch.float32).to(device)
        logits = model(xb)
        pred = torch.argmax(logits, dim=-1).cpu().numpy()  # (b, T)
        preds.append(pred)
    return np.concatenate(preds, axis=0) if preds else np.empty((0,))


def _false_alarm_rate(y_true_flat, y_pred_flat, normal_idx: int) -> float:
    """FAR = FP / (FP + TN), agregasi Normal-vs-Attack (proposal 5.8.4):
    TN = Normal diprediksi Normal; FP = Normal diprediksi sebagai kelas serangan apa pun."""
    normal_mask = y_true_flat == normal_idx
    if normal_mask.sum() == 0:
        return float("nan")
    predicted_normal = y_pred_flat[normal_mask] == normal_idx
    tn = predicted_normal.sum()
    fp = (~predicted_normal).sum()
    return float(fp / (fp + tn)) if (fp + tn) > 0 else float("nan")


@torch.no_grad()
def predict_proba(model, X: np.ndarray, batch_size: int = 256, device: str = None):
    """Probabilitas softmax (N, T, C), float32."""
    device = device or (TRAIN.device if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)
    out = []
    for i in range(0, len(X), batch_size):
        xb = torch.tensor(X[i:i + batch_size], dtype=torch.float32).to(device)
        out.append(torch.softmax(model(xb), dim=-1).cpu().numpy())
    return np.concatenate(out, axis=0)


def threshold_free_metrics(proba: np.ndarray, Y: np.ndarray, class_names=None):
    """Metrik tanpa ambang (analisis pendukung): ROC-AUC dan PR-AUC
    (average precision) satu-lawan-semua per kelas, rata-rata makro, dan
    Normal-vs-serangan (skor = 1 - P(Normal)). `prevalence` = nilai PR-AUC
    bila skor acak, sebagai pembanding."""
    class_names = class_names or list(DATA.classes_used)
    normal_idx = class_names.index("Normal")
    C = len(class_names)
    p = proba.reshape(-1, C)
    y = Y.reshape(-1)

    def _pair(y_bin, score):
        if y_bin.sum() == 0 or y_bin.all():
            return {"roc_auc": float("nan"), "pr_auc": float("nan"),
                    "prevalence": float(y_bin.mean())}
        return {
            "roc_auc": float(roc_auc_score(y_bin, score)),
            "pr_auc": float(average_precision_score(y_bin, score)),
            "prevalence": float(y_bin.mean()),
        }

    per_class = {name: _pair(y == i, p[:, i]) for i, name in enumerate(class_names)}
    macro = {k: float(np.nanmean([v[k] for v in per_class.values()]))
             for k in ("roc_auc", "pr_auc", "prevalence")}
    return {
        "per_class": per_class,
        "macro": macro,
        "normal_vs_attack": _pair(y != normal_idx, 1.0 - p[:, normal_idx]),
    }


def evaluate_model(model, X: np.ndarray, Y: np.ndarray, class_names=None):
    proba = predict_proba(model, X)       # (N, T, C)
    y_pred = proba.argmax(axis=-1)        # (N, T), sama dengan predict()
    results = evaluate_predictions(y_pred, Y, class_names)
    results["threshold_free"] = threshold_free_metrics(proba, Y, class_names)
    return results


def baseline_predictions(Y: np.ndarray, last_labels: np.ndarray, kind: str,
                         class_names=None) -> np.ndarray:
    """Acuan sederhana (A16), BUKAN pesaing model: keduanya memakai informasi
    yang tidak dimiliki model. 'persistence' = label rekaman terakhir di
    jendela historis untuk semua T langkah; 'always_normal' = selalu Normal."""
    class_names = class_names or list(DATA.classes_used)
    if kind == "persistence":
        return np.repeat(last_labels[:, None], Y.shape[1], axis=1)
    if kind == "always_normal":
        return np.full(Y.shape, class_names.index("Normal"), dtype=np.int64)
    raise ValueError(kind)


def evaluate_predictions(y_pred: np.ndarray, Y: np.ndarray, class_names=None):
    class_names = class_names or list(DATA.classes_used)
    normal_idx = class_names.index("Normal")
    y_true_flat = Y.reshape(-1)
    y_pred_flat = y_pred.reshape(-1)

    labels = list(range(len(class_names)))
    results = {
        "accuracy": accuracy_score(y_true_flat, y_pred_flat),
        "precision_macro": precision_score(y_true_flat, y_pred_flat, labels=labels, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true_flat, y_pred_flat, labels=labels, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true_flat, y_pred_flat, labels=labels, average="macro", zero_division=0),
        "precision_weighted": precision_score(y_true_flat, y_pred_flat, labels=labels, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_true_flat, y_pred_flat, labels=labels, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_true_flat, y_pred_flat, labels=labels, average="weighted", zero_division=0),
        "false_alarm_rate": _false_alarm_rate(y_true_flat, y_pred_flat, normal_idx),
        "confusion_matrix": confusion_matrix(y_true_flat, y_pred_flat, labels=labels).tolist(),
        "class_names": class_names,
    }

    # Metrik per kelas (one-vs-rest) -- tabel pembahasan per kategori di skripsi.
    p_c = precision_score(y_true_flat, y_pred_flat, labels=labels, average=None, zero_division=0)
    r_c = recall_score(y_true_flat, y_pred_flat, labels=labels, average=None, zero_division=0)
    f_c = f1_score(y_true_flat, y_pred_flat, labels=labels, average=None, zero_division=0)
    supports = np.bincount(y_true_flat, minlength=len(labels))
    results["per_class"] = {
        name: {
            "precision": float(p_c[i]), "recall": float(r_c[i]),
            "f1": float(f_c[i]), "support": int(supports[i]),
        }
        for i, name in enumerate(class_names)
    }

    # Rincian per horizon t+1 .. t+T (metrik per langkah target, sebelum
    # dirata-ratakan pada metrik flattened di atas).
    T = Y.shape[1] if Y.ndim == 2 else 1
    per_horizon = {}
    for t in range(T):
        yt, yp = Y[:, t], y_pred[:, t]
        per_horizon[f"t+{t+1}"] = {
            "accuracy": accuracy_score(yt, yp),
            "f1_macro": f1_score(yt, yp, labels=labels, average="macro", zero_division=0),
            "false_alarm_rate": _false_alarm_rate(yt, yp, normal_idx),
        }
    results["per_horizon"] = per_horizon
    return results
