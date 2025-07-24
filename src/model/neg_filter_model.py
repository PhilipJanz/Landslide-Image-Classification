from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, confusion_matrix, f1_score
from sklearn.metrics import precision_recall_curve
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from sklearn.model_selection import StratifiedKFold

# Add src to path to import config
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import PROCESSED_FEATURE_PATH, TRAIN_CSV_PATH, SEED

TARGET_PRECISION = 1

# Assume features and labels are loaded
# Load features
X = pd.read_csv(PROCESSED_FEATURE_PATH / 'train.csv', index_col=0) # Load test features
X_test = pd.read_csv(PROCESSED_FEATURE_PATH / 'test.csv', index_col=0)

# Load labels
labels_df = pd.read_csv(TRAIN_CSV_PATH).sort_values("ID", ignore_index=True)
# Assume the label column is named 'label'
y = labels_df['label']
assert np.all(labels_df["ID"] == X.index.values)

# Prepare for cross-validation
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=False)

# Store out-of-fold predictions and test predictions
oof_pred = np.zeros(X.shape[0])
test_pred = np.zeros((X_test.shape[0], n_splits))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    clf = XGBClassifier(
        max_depth=5,
        min_child_weight=10,
        eval_metric='logloss',
        random_state=SEED + fold
    )
    clf.fit(X_train, y_train)
    val_probs = clf.predict_proba(X_val)[:, 1]
    precision, recall, thresholds_fold = precision_recall_curve(y_val, val_probs)
    # Find threshold for desired precision
    for i, (r, t) in enumerate(zip(recall[1:], thresholds_fold)):
        if r < TARGET_PRECISION:
            threshold = t
            break
    oof_pred[val_idx] = (val_probs >= threshold).astype(int)
    # Predict on test set for this fold
    test_probs = clf.predict_proba(X_test)[:, 1]
    test_pred[:, fold] = (test_probs >= threshold).astype(int)

# Majority threshold for test set
test_pred_mean = np.mean(test_pred, axis=1)

# Save train predictions (out-of-fold)
train_pred_df = pd.DataFrame({
    'ID': X.index,
    'label': oof_pred.astype(int)
})
train_pred_df.to_csv(PROCESSED_FEATURE_PATH / 'train_prediction.csv', index=False)

# Save test predictions
test_pred_df = pd.DataFrame({
    'ID': X_test.index.values,
    'label': test_pred_mean
})
test_pred_df.to_csv(PROCESSED_FEATURE_PATH / 'test_prediction.csv', index=False)

# Print evaluation for train (out-of-fold)
print("Confusion matrix (OOF):")
print(confusion_matrix(y, oof_pred))
print(f"Negatives in train.py: {np.sum(oof_pred == 0.0)} ({np.round(100 * np.sum(oof_pred == 0.0) / len(oof_pred))}% of all test samples)")
print("Precision (OOF):", precision_score(y, oof_pred))
print("Recall (OOF):", recall_score(y, oof_pred))
print("F1 (OOF):", f1_score(y, oof_pred))

print(f"\nNegatives in test.py: {np.sum(test_pred_mean == 0.0)} ({np.round(100 * np.sum(test_pred_mean == 0.0) / len(test_pred_mean))}% of all test samples)")
