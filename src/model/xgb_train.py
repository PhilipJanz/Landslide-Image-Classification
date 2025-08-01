import numpy as np
from pathlib import Path
import sys
import time
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from tqdm import tqdm
import warnings
import math
from sklearn.model_selection import KFold
from codecarbon import track_emissions
from xgboost import XGBRegressor



#warnings.filterwarnings('ignore')

# Add src to Python path
src_path = str(Path(__file__).parent.parent)
if src_path not in sys.path:
    sys.path.append(src_path)

import config
from utils.visualizations import create_summary_plot
from utils.pred_postprocessing import anchored_sigmoid
from model.model_config import XGB_PARAMS


def init_model(trial=None, params=None):
    # check if inputs are reasonable: xor on trial and params
    assert (not trial) ^ (not params), "Choose either params OR give an optuna trial (it's xor)"

    if trial:
        model_params = {
            'max_depth': trial.suggest_int('max_depth', 2, 20),
            'learning_rate': trial.suggest_float('learning_rate', 1e-2, 1, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 1.0),
            'gamma': trial.suggest_float('gamma', 1e-9, 1, log=True),
            'alpha': trial.suggest_float('alpha', 1e-9, 100.0, log=True),
            "lambda": 0,
            "n_estimators": trial.suggest_int('n_estimators', 100, 500)
        }
    else:
        model_params = {key: params[key] for key in
                        ["max_depth", "learning_rate", "subsample", 'colsample_bytree', 'gamma', "alpha"]}
    #model_params['objective'] = 'binary:logistic'
    return XGBRegressor(**model_params, random_state=config.SEED, objective="binary:logistic", importance_type="gain"), model_params
    

def calculate_metrics(y_true, y_pred, threshold=.5):
    """Calculate accuracy and F1 score."""
    # Convert to numpy arrays    
    if not threshold: # optimize based on f1
        threshold_ls = np.arange(20, 81) / 100
        f1_ls = [f1_score(y_true, (y_pred > threshold).astype(int), average='binary') for threshold in threshold_ls]
        threshold = threshold_ls[np.argmax(f1_ls)]
        print(f"F1-optimal threshold: {threshold:.4f} F1: {np.max(f1_ls):.4f}")

    # Convert probabilities to binary predictions
    y_pred_binary = (y_pred > threshold).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred_binary)
    f1 = f1_score(y_true, y_pred_binary, average='binary')
    recall = recall_score(y_true, y_pred_binary, average='binary')
    precision = precision_score(y_true, y_pred_binary, average='binary')
    return accuracy, recall, precision, f1, threshold

def load_X_y(filter_obvious_negs=True):
    df_train = pd.read_csv(config.PROCESSED_FEATURE_PATH / 'train.csv', index_col=0) # Load test features

    # Load labels
    labels_df = pd.read_csv(config.TRAIN_CSV_PATH).sort_values("ID", ignore_index=True)
    # Assume the label column is named 'label'
    y = labels_df['label'].values

    assert np.all(labels_df["ID"] == df_train.index.values)

    if filter_obvious_negs:
        feature_model_preds = pd.read_csv(config.PROCESSED_FEATURE_PATH / "train_prediction.csv")
        obvious_neg_image_ids = feature_model_preds.ID.values[feature_model_preds.label == 0]
        obvious_neg_image_loc = [id not in obvious_neg_image_ids for id in df_train.index]

        df_train = df_train.iloc[obvious_neg_image_loc]

    y = y[obvious_neg_image_loc]

    X = df_train.values
    return X, y

#@track_emissions()
def train_model(X, y, 
                test_prediction=True,
                save_model=True,
                trial=None,
                params=XGB_PARAMS,
                ):
    """Main training function with 5-fold cross-validation and no early stopping."""
    print(f"Set random seed to {config.SEED} for reproducibility")
    
    # Assume features and labels are loaded
    # Load features
    #X_test = df_test.values
    total_size = len(X)
    indices = np.arange(total_size)
    
    kf = KFold(n_splits=10, shuffle=True, random_state=config.SEED)

    # Prepare model directory
    model_dir = config.MODEL_DIR / config.MODEL_NAME
    model_dir.mkdir(parents=True, exist_ok=True)

    if test_prediction:
        df_test = pd.read_csv(config.PROCESSED_FEATURE_PATH / 'test.csv', index_col=0)
        X_test = df_test.values
        test_pred_ls = []

    fold_metrics = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(indices)):
        print(f"\nFOLD {fold+1}/5")
        print("-" * 50)

        X_train = X[train_idx]
        y_train = y[train_idx]
        X_val = X[val_idx]
        y_val = y[val_idx]
        
        model, _ = init_model(trial=trial, params=params)

        model.fit(X_train, y_train)
        train_preds = model.predict(X_train)
        val_preds = model.predict(X_val)

        train_acc, train_recall, train_precision, train_f1, f1_opt_threshold = calculate_metrics(y_train, train_preds, threshold=None)
        val_acc, val_recall, val_precision, val_f1, _ = calculate_metrics(y_val, val_preds, threshold=.5)
    
        print(f"Final Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f} (at threshold {f1_opt_threshold})")
        fold_metrics.append({
            'fold': fold,
            'val_acc': val_acc,
            'val_f1': val_f1,
            'val_recall': val_recall,
            'val_precision': val_precision
        })

        if test_prediction:
            test_preds = model.predict(X_test)
            # respect f1 optimal threshold
            #test_preds = [anchored_sigmoid(pred, t=f1_opt_threshold) for pred in test_preds]
            test_pred_ls.append(test_preds)
        
    
    # Calculate and display average metrics across all folds
    print("\n" + "="*80)
    print("CROSS-VALIDATION RESULTS SUMMARY")
    print("="*80)
    
    # Individual fold results
    print("\nIndividual Fold Results:")
    print("-" * 60)
    for m in fold_metrics:
        print(f"Fold {m['fold']+1}: Acc={m['val_acc']:.4f}, F1={m['val_f1']:.4f}, "
              f"Recall={m['val_recall']:.4f}, Precision={m['val_precision']:.4f}")
    
    # Calculate averages
    avg_acc = np.mean([m['val_acc'] for m in fold_metrics])
    avg_recall = np.mean([m['val_recall'] for m in fold_metrics])
    avg_precision = np.mean([m['val_precision'] for m in fold_metrics])
    avg_f1 = np.mean([m['val_f1'] for m in fold_metrics])
    
    # Calculate standard deviations
    std_acc = np.std([m['val_acc'] for m in fold_metrics])
    std_f1 = np.std([m['val_f1'] for m in fold_metrics])
    std_recall = np.std([m['val_recall'] for m in fold_metrics])
    std_precision = np.std([m['val_precision'] for m in fold_metrics])
    
    # Display average results
    print("\nAverage Results Across All Folds:")
    print("-" * 60)
    print(f"Validation Accuracy: {avg_acc:.4f} ± {std_acc:.4f}")
    print(f"Validation Recall:   {avg_recall:.4f} ± {std_recall:.4f}")
    print(f"Validation Precision:{avg_precision:.4f} ± {std_precision:.4f}")
    print(f"Validation F1-Score: {avg_f1:.4f} ± {std_f1:.4f}")
    
    # Create summary plot
    #create_summary_plot(fold_metrics, model_dir)

    if test_prediction:
        test_pred_mtx = np.stack(test_pred_ls).T
        mean_prediction = np.mean(test_pred_mtx, axis=1)
        std_prediction = np.std(test_pred_mtx, axis=1)
        binary_prediction = (mean_prediction > 0.5).astype(float)
        prediction_df = pd.DataFrame({
            'ID': df_test.index,
            'pred': mean_prediction,
            "std": std_prediction
        })
        submission_df = pd.DataFrame({
            'ID': df_test.index,
            'label': binary_prediction
        })
        prediction_df.to_csv(model_dir / f"prediction_{config.MODEL_NAME}.csv", index=False)
        submission_df.to_csv(config.SUBMISSIONS_DIR / f"submission_{config.MODEL_NAME}.csv", index=False)
        print(f"\nPrediction Statistics:")
        print(f"Total predictions: {len(binary_prediction)}")
        print(f"Positive predictions (landslide): {sum(binary_prediction)}")
        print(f"Positive rate: {sum(binary_prediction) / len(binary_prediction) * 100:.2f}%")
    
    return fold_metrics

if __name__ == "__main__":
    X, y = load_X_y()
    train_model(X, y)
