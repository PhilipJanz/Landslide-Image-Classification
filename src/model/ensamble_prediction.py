import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to Python path
src_path = str(Path(__file__).parent.parent)
if src_path not in sys.path:
    sys.path.append(src_path)

import config

models = ["landslide_MMCNN_V6",  "landslide_MMCNN_V5_"]

dfs = {}
for model_name in models:
    df = pd.read_csv(config.MODEL_DIR / model_name / "prediction.csv").drop_duplicates().sort_values("ID")
    dfs[model_name] = df

# Union of all IDs
assert all(dfs[models[0]].ID.values == dfs[models[1]].ID.values)

# Build DataFrame with all predictions
ensemble_df = pd.DataFrame(index=df.ID)
std_df = pd.DataFrame(index=df.ID)
for model_name, df in dfs.items():
    ensemble_df[model_name] = df["pred"].values
    std_df[model_name] = df["std"].values

# make weights based on std
std_df = std_df.values
# first: convert uncertainty to confidence
confidence_mtx = 1 / (std_df + 1e-6)
# second: standardize overall confidence (XGB seems overall much more confident)
confidence_mtx = confidence_mtx / confidence_mtx.mean(axis=0)
# third: make weight for each images
confidence_mtx = (confidence_mtx.T / confidence_mtx.sum(axis=1)).T

# Compare predictions (rounded to binary for agreement)
print("Agreement matrix (rounded):")
print(ensemble_df.round().value_counts().reset_index(name="count"))

# Weighted Ensemble:
weighted_ensemble_df = ensemble_df  #* confidence_mtx 
ensemble_df["ensemble"] = (weighted_ensemble_df.sum(axis=1) > 0.5).astype(int)
"""
ensemble_df["ensemble"] = ensemble_df["landslide_MMCNN_V5_"]
xgb_conf_loc = (std_df["landslide_MMCNN_V5_"] > 0.15) & (std_df["XGB_V0"] < 0.01)
print(f"XGB updates {np.sum(xgb_conf_loc)} predictions")
ensemble_df["ensemble"][xgb_conf_loc] = ensemble_df["XGB_V0"][xgb_conf_loc]
ensemble_df["ensemble"] = (ensemble_df["ensemble"] > 0.5) * 1

"""

# Save ensembled predictions
output_path = Path(config.SUBMISSIONS_DIR / f"ensemble_prediction_{"_".join(models)}.csv")
output_path.parent.mkdir(parents=True, exist_ok=True)
ensemble_df[["ensemble"]].rename(columns={"ensemble": "label"}).to_csv(
    output_path, index_label="ID"
)
print(f"Ensembled predictions saved to {output_path}")
print(f"\nPrediction Statistics:")
print(f"Total predictions: {len(ensemble_df["ensemble"])}")
print(f"Positive predictions (landslide): {sum(ensemble_df["ensemble"])}")
print(f"Positive rate: {sum(ensemble_df["ensemble"]) / len(ensemble_df["ensemble"]) * 100:.2f}%")
