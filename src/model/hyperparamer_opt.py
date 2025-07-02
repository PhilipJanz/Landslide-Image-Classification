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
    dropout = trial.suggest_float('dropout', 0.2, 0.6)
    final_dropout = trial.suggest_float('final_dropout', 0.2, 0.6)
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-2)

    fold_metrics = train_model(fc_units=fc_units, dropout=dropout, final_dropout=final_dropout, lr=lr, weight_decay=weight_decay, show_process=False, save_model=False)
    return np.mean([m["val_f1"] for m in fold_metrics])


sampler = optuna.samplers.TPESampler(n_startup_trials=50, multivariate=True,
                                     warn_independent_sampling=False, seed=SEED)
study = optuna.create_study(
    storage="sqlite:///optuna_study.db",  # Specify the storage URL here.
    study_name=MODEL_NAME, 
    sampler=sampler,
    direction="maximize"
)
study.optimize(objective, n_trials=150)
print(f"Best value: {study.best_value} (params: {study.best_params})")
