"""
hyperparameter_tuning.py
Pencarian hyperparameter memakai Optuna (Akiba et al., 2019). Objektif
yang dimaksimalkan adalah F1-macro pada data validasi (lebih representatif
untuk data tidak seimbang, proposal Bagian 5.8.3), bukan akurasi.

Parameter yang dicari: hidden_size, dropout, attn_heads, learning_rate.
attn_heads dibatasi ke nilai yang membagi habis kandidat hidden_size agar
valid untuk InterpretableMultiHeadAttention (d_attn = embed_dim / heads).
"""
import copy
import optuna
from optuna.trial import TrialState

from config import MODEL, TRAIN, TUNING
from train import train_model
from evaluate import evaluate_model


def _suggest_valid_heads(trial, hidden_size: int, options=(2, 4, 8)):
    valid = [h for h in options if hidden_size % h == 0]
    if not valid:
        valid = [1]
    return trial.suggest_categorical(f"attn_heads_{hidden_size}", valid)


def build_objective(X_train, Y_train, X_val, Y_val, num_features: int,
                    categorical_features: dict = None, quiet: bool = True):
    def objective(trial: optuna.Trial):
        hidden_size = trial.suggest_categorical("hidden_size", [32, 64, 128])
        dropout = trial.suggest_float("dropout", 0.0, 0.4)
        attn_heads = _suggest_valid_heads(trial, hidden_size)
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True)

        model_cfg = copy.deepcopy(MODEL)
        model_cfg.hidden_size = hidden_size
        model_cfg.dropout = dropout
        model_cfg.attn_heads = attn_heads

        train_cfg = copy.deepcopy(TRAIN)
        train_cfg.learning_rate = learning_rate
        train_cfg.epochs = min(train_cfg.epochs, 15)  # dipersingkat khusus saat tuning

        model, _ = train_model(
            X_train, Y_train, X_val, Y_val, num_features,
            model_cfg=model_cfg, train_cfg=train_cfg, verbose=not quiet,
            desc=f"trial{trial.number}", show_progress=False,
            categorical_features=categorical_features,
        )
        results = evaluate_model(model, X_val, Y_val)
        return results[TUNING.metric]

    return objective


def run_tuning(X_train, Y_train, X_val, Y_val, num_features: int, n_trials: int = None,
               categorical_features: dict = None):
    n_trials = n_trials or TUNING.n_trials
    study = optuna.create_study(direction=TUNING.direction)
    objective = build_objective(
        X_train, Y_train, X_val, Y_val, num_features,
        categorical_features=categorical_features,
    )
    study.optimize(objective, n_trials=n_trials)

    completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
    if not completed:
        raise RuntimeError("Tidak ada trial Optuna yang berhasil selesai.")

    return study.best_params, study
