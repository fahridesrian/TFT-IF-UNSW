"""
train.py
Manual training loop (bukan pytorch-lightning) untuk menghindari kelas
masalah kompatibilitas yang pernah dialami pada proyek TFT+IF sebelumnya.
Mengimplementasikan:
  - class weighting pada cross-entropy loss (strategi utama penanganan
    imbalance, proposal Bagian 6.e.iii, karena tidak mengubah struktur
    temporal data)
  - Adam optimizer (Kingma & Ba, 2015)
  - ReduceLROnPlateau learning rate scheduler
  - validasi berkala + early stopping (proposal Bagian 6.h.vi)
  - mixed precision training (AMP, Micikevicius et al., 2018) di GPU
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.utils.class_weight import compute_class_weight
from tqdm.auto import tqdm

from config import TRAIN, MODEL
from tft_model import TemporalFusionTransformer

torch.backends.cudnn.benchmark = True


def _cpu_state_dict(model: nn.Module) -> dict:
    """Snapshot state_dict ke CPU alih-alih copy.deepcopy() langsung di GPU.
    copy.deepcopy() pada tensor CUDA dikenal rapuh di beberapa kombinasi
    driver/CUDA Windows (memicu 'CUDA error: unknown error' yang sebenarnya
    berasal dari operasi async sebelumnya, tersinkron ulang saat deepcopy).
    Mengambil state ke CPU juga menghindari duplikasi memori GPU untuk
    menyimpan checkpoint model terbaik selama training."""
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def make_loader(X: np.ndarray, Y: np.ndarray, batch_size: int, shuffle: bool,
                 num_workers: int = None):
    num_workers = TRAIN.num_workers if num_workers is None else num_workers
    X_t = torch.from_numpy(X).float()
    Y_t = torch.from_numpy(Y).long()
    ds = TensorDataset(X_t, Y_t)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )


def compute_weights(Y_train: np.ndarray, num_classes: int) -> torch.Tensor:
    flat = Y_train.reshape(-1)
    present = np.unique(flat)
    weights = np.ones(num_classes, dtype=np.float32)
    if len(present) > 1:
        cw = compute_class_weight(class_weight="balanced", classes=present, y=flat)
        for cls, w in zip(present, cw):
            weights[cls] = w
    return torch.tensor(weights, dtype=torch.float32)


def train_model(
    X_train, Y_train, X_val, Y_val, num_features: int,
    model_cfg=MODEL, train_cfg=TRAIN, use_class_weighting: bool = True,
    verbose: bool = True, desc: str = "model", show_progress: bool = True,
):
    torch.manual_seed(train_cfg.seed)
    device = torch.device(train_cfg.device if torch.cuda.is_available() else "cpu")

    model = TemporalFusionTransformer(
        num_features=num_features,
        hidden_size=model_cfg.hidden_size,
        num_classes=model_cfg.num_classes,
        T=Y_train.shape[1],
        lstm_layers=model_cfg.lstm_layers,
        attn_heads=model_cfg.attn_heads,
        dropout=model_cfg.dropout,
    ).to(device)

    class_weights = None
    if use_class_weighting:
        class_weights = compute_weights(Y_train, model_cfg.num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=train_cfg.learning_rate, weight_decay=train_cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=train_cfg.lr_scheduler_factor,
        patience=train_cfg.lr_scheduler_patience,
    )

    train_loader = make_loader(X_train, Y_train, train_cfg.batch_size, shuffle=True)
    val_loader = make_loader(X_val, Y_val, train_cfg.batch_size, shuffle=False)

    # AMP diinisialisasi SEKALI di luar loop epoch. GradScaler menyesuaikan
    # faktor skala loss secara adaptif sepanjang training; membuatnya ulang
    # tiap epoch (seperti pada revisi sebelumnya) membuang adaptasi itu dan
    # bisa memicu step yang di-skip berulang kali di awal tiap epoch.
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val_loss = float("inf")
    best_state = _cpu_state_dict(model)
    patience_counter = 0
    history = {"train_loss": [], "val_loss": []}

    epoch_bar = tqdm(
        range(train_cfg.epochs), desc=f"[{desc}] epochs", disable=not show_progress,
        leave=True,
    )
    for epoch in epoch_bar:
        model.train()
        train_losses = []
        batch_bar = tqdm(
            train_loader, desc=f"[{desc}] epoch {epoch+1}/{train_cfg.epochs} (train)",
            disable=not show_progress, leave=False,
        )
        for xb, yb in batch_bar:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(xb)  # (b, T, C)
                loss = criterion(logits.reshape(-1, model_cfg.num_classes), yb.reshape(-1))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(loss.item())
            batch_bar.set_postfix(loss=f"{loss.item():.4f}")

        model.eval()
        val_losses = []
        val_bar = tqdm(
            val_loader, desc=f"[{desc}] epoch {epoch+1}/{train_cfg.epochs} (val)",
            disable=not show_progress, leave=False,
        )
        with torch.no_grad():
            for xb, yb in val_bar:
                xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)

                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits = model(xb)
                    loss = criterion(logits.reshape(-1, model_cfg.num_classes), yb.reshape(-1))

                val_losses.append(loss.item())
                val_bar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        val_loss = float(np.mean(val_losses)) if val_losses else float("nan")
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        scheduler.step(val_loss)

        epoch_bar.set_postfix(train_loss=f"{train_loss:.4f}", val_loss=f"{val_loss:.4f}")
        if verbose:
            tqdm.write(f"  [{desc}] epoch {epoch+1}/{train_cfg.epochs} "
                       f"train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state = _cpu_state_dict(model)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= train_cfg.patience:
                if verbose:
                    tqdm.write(f"  [{desc}] Early stopping at epoch {epoch+1}")
                epoch_bar.close()
                break

    model.load_state_dict(best_state)
    model.to(device)
    return model, history