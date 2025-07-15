from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, confusion_matrix, f1_score
from sklearn.metrics import precision_recall_curve
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path to import config
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import PROCESSED_FEATURE_PATH, TRAIN_CSV_PATH, SEED

# Assume features and labels are loaded
# Load features
X = pd.read_csv(PROCESSED_FEATURE_PATH / 'train.csv', index_col=0)
# Load labels
labels_df = pd.read_csv(TRAIN_CSV_PATH).sort_values("ID")
# Assume the label column is named 'label'
y = labels_df['label']
assert np.all(labels_df["ID"] == X.index.values)

# Split with stratification
X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.25, random_state=6)

# Train a small decision tree
clf = XGBClassifier(
    max_depth=3,
    min_child_weight=10,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=SEED
)
clf.fit(X_train, y_train)

probs = clf.predict_proba(X_val)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_val, probs)

# Find threshold for desired precision
target_precision = 0.999
for p, r, t in zip(precision, recall, thresholds):
    if r <= target_precision:
        threshold = t
        break
# Apply this threshold
y_pred = (probs >= threshold).astype(int)
#y_pred = X["ndvi_mean"] < -1.5

# Evaluate
print("Confusion matrix:")
print(confusion_matrix(y_val, y_pred))
print("Precision:", precision_score(y_val, y_pred))
print("Recall:", recall_score(y_val, y_pred))
print("F1:", f1_score(y_val, y_pred))
