import optuna
import numpy as np
import sys
from pathlib import Path
from functools import partial

# Add src to Python path
src_path = str(Path(__file__).parent.parent)
if src_path not in sys.path:
    sys.path.append(src_path)

from config import MODEL_NAME, SEED
from model.xgb_train import load_X_y, train_model

def objective(trial, X, y):
    fold_metrics = train_model(X=X, y=y, test_prediction=False, trial=trial, params=False, save_model=False)
    return np.mean([m["val_f1"] for m in fold_metrics])

sampler = optuna.samplers.TPESampler(n_startup_trials=50, multivariate=True,
                                     warn_independent_sampling=False, seed=SEED)
study = optuna.create_study(
    storage="sqlite:///optuna_study.db",  # Specify the storage URL here.
    study_name=MODEL_NAME, 
    sampler=sampler,
    direction="maximize"
)
X, y = load_X_y()
study.optimize(partial(objective, X=X, y=y), n_trials=200)
print(f"Best value: {study.best_value} (params: {study.best_params})")
