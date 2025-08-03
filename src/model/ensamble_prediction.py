import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to Python path
src_path = str(Path(__file__).parent.parent)
if src_path not in sys.path:
    sys.path.append(src_path)

import config

models = ["landslide_MMCNN_final42",  "landslide_MMCNN_sar_final42"]

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

# predefine final labels with prediction of full model
ensemble_df["label"] = (ensemble_df[models[0]] > 0.5) * 1

# load cloud coverage data
cc_df = pd.read_csv(config.PROCESSED_DATA_DIR / f"test_cloud_coverage.csv")
assert all(dfs[models[0]].ID.values == cc_df.ID.values)
# set threshold on TODO
cc_threshold = .1
clouded_idx = cc_df.cloud_coverage > cc_threshold
print(f"Marked {np.sum(clouded_idx)} out of {len(clouded_idx)} of test-images as significantly cloud covered.")

clouded_ensemble_df = ensemble_df.reset_index()[clouded_idx]
pred_mtx = (clouded_ensemble_df.values[:, [1, 2]] > 0.5) * 1
diff_pred_idx = pred_mtx[:, 0] != pred_mtx[:, 1]
print(f"... the models disagree in {np.sum(diff_pred_idx)} cases.")
confident_pred_idx = std_df[models[1]].values[clouded_idx][diff_pred_idx] < .05
print(f"... the cloud models is confident in {np.sum(confident_pred_idx)} cases.\n")
switch_ids = clouded_ensemble_df[diff_pred_idx][confident_pred_idx].ID.values
ensemble_df.loc[switch_ids, "label"] = (ensemble_df.loc[switch_ids, "label"] - 1) * (-1)
assert np.all(ensemble_df.loc[switch_ids, "label"].values == pred_mtx[diff_pred_idx, 1][confident_pred_idx])

# Save ensembled predictions
output_path = Path(config.SUBMISSIONS_DIR / f"cloudy_prediction_{"_".join(models)}.csv")
output_path.parent.mkdir(parents=True, exist_ok=True)
ensemble_df[["label"]].to_csv(
    output_path, index_label="ID"
)
print(f"Ensembled predictions saved to {output_path}")
print(f"\nPrediction Statistics:")
print(f"Total predictions: {len(ensemble_df["label"])}")
print(f"Positive predictions (landslide): {sum(ensemble_df["label"])}")
print(f"Positive rate: {sum(ensemble_df["label"]) / len(ensemble_df["label"]) * 100:.2f}%")
