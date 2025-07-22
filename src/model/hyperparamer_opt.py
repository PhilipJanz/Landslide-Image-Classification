import optuna
import numpy as np
import sys
from pathlib import Path

# Add src to Python path
src_path = str(Path(__file__).parent.parent)
if src_path not in sys.path:
    sys.path.append(src_path)

from config import MODEL_NAME, SEED
from model.train import train_model

def objective(trial):
    # Define hyperparameter search space
    fc_units = trial.suggest_categorical('fc_units', [64, 128, 256])
    fusioned_kernel_units = trial.suggest_categorical('fusioned_kernel_units', [64, 128, 256])
    dropout = trial.suggest_float('dropout', 0.0, 0.6)
    final_dropout = trial.suggest_float('final_dropout', 0.0, 0.6)
    lr = trial.suggest_loguniform('lr', 1e-4, 1e-2)
    weight_decay = trial.suggest_loguniform('weight_decay', 5e-6, 5e-4)
    bce_weight = 2 #trial.suggest_float('bce_weight', 1, 6)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])

    fold_metrics = train_model(fc_units=fc_units, fusioned_kernel_units=fusioned_kernel_units,
                               dropout=dropout, final_dropout=final_dropout, 
                               lr=lr, weight_decay=weight_decay, bce_weight=bce_weight, batch_size=batch_size,
                               show_process=False, save_model=False)
    return np.mean([m["val_f1"] for m in fold_metrics])


sampler = optuna.samplers.TPESampler(n_startup_trials=20, multivariate=True,
                                     warn_independent_sampling=False, seed=SEED)
study = optuna.create_study(
    storage="sqlite:///optuna_study.db",  # Specify the storage URL here.
    study_name=MODEL_NAME, 
    sampler=sampler,
    direction="maximize"
)
study.optimize(objective, n_trials=150)
print(f"Best value: {study.best_value} (params: {study.best_params})")
