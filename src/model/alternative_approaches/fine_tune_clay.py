import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent))
import config

SEED = config.SEED
np.random.seed(SEED)
torch.manual_seed(SEED)

# ---- Config ----
EMBEDDING_DIR = config.EMBEDDING_DATA_DIR
TRAIN_CSV = config.TRAIN_CSV_PATH
BATCH_SIZE = 64
EPOCHS = 100
LR = 1e-3
N_SPLITS = 5

# ---- Model ----
class LinearHead(nn.Module):
    def __init__(self, in_dim, hidden_dim=512, dropout=0.25):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# ---- Data Loading ----
def load_embeddings_and_labels():
    # Load train embeddings
    train_emb_path = EMBEDDING_DIR / "train" / "train_sentinel-2_embeddings.pt"
    X_train = torch.load(train_emb_path)
    # Load test embeddings
    test_emb_path = EMBEDDING_DIR / "test" / "test_sentinel-2_embeddings.pt"
    X_test = torch.load(test_emb_path)
    # Load train labels
    df = pd.read_csv(TRAIN_CSV)
    y_train = torch.tensor(df['label'].values, dtype=torch.float32)  # adjust column name if needed
    return X_train, y_train, X_test

# ---- Training and Evaluation ----
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(xb).squeeze(1)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    return total_loss / len(loader.dataset)

def eval_epoch(model, loader, device):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            out = torch.sigmoid(model(xb).squeeze(1)).cpu().numpy()
            preds.append(out)
            targets.append(yb.numpy())
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    pred_labels = (preds > 0.5).astype(int)
    acc = accuracy_score(targets, pred_labels)
    f1 = f1_score(targets, pred_labels)
    recall = recall_score(targets, pred_labels)
    precision = precision_score(targets, pred_labels)
    return acc, f1, recall, precision, preds, targets

def make_loader(X, y=None, batch_size=32, shuffle=False):
    if y is not None:
        dataset = torch.utils.data.TensorDataset(X, y)
    else:
        dataset = torch.utils.data.TensorDataset(X)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

# ---- Main CV Loop ----
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train, y_train, X_test = load_embeddings_and_labels()
    in_dim = X_train.shape[1]
    print(f"Train embeddings: {X_train.shape}, Test embeddings: {X_test.shape}")

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof_preds = np.zeros(len(X_train))
    test_preds = np.zeros(len(X_test))
    metrics = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        print(f"\nFOLD {fold+1}/{N_SPLITS}")
        model = LinearHead(in_dim).to(device)
        optimizer = optim.Adam(model.parameters(), lr=LR)
        criterion = nn.BCEWithLogitsLoss()

        X_tr, y_tr = X_train[train_idx], y_train[train_idx]
        X_val, y_val = X_train[val_idx], y_train[val_idx]
        train_loader = make_loader(X_tr, y_tr, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = make_loader(X_val, y_val, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = make_loader(X_test, batch_size=BATCH_SIZE, shuffle=False)

        best_f1 = 0
        for epoch in range(EPOCHS):
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
            train_acc, train_f1, train_recall, train_precision, _, _ = eval_epoch(model, train_loader, device)
            val_acc, val_f1, val_recall, val_precision, val_preds, val_targets = eval_epoch(model, val_loader, device)
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_model_state = model.state_dict()
            if (epoch+1) % 5 == 0 or epoch == 0:
                print(
                    f"Epoch {epoch+1}: "
                    f"loss={train_loss:.4f}, "
                    f"train_acc={train_acc:.4f}, train_f1={train_f1:.4f}, "
                    f"val_acc={val_acc:.4f}, val_f1={val_f1:.4f}"
                )

        # Load best model for this fold
        model.load_state_dict(best_model_state)
        acc, f1, recall, precision, val_preds, val_targets = eval_epoch(model, val_loader, device)
        print(f"FOLD {fold+1} FINAL: acc={acc:.4f}, f1={f1:.4f}, recall={recall:.4f}, precision={precision:.4f}")
        metrics.append({'acc': acc, 'f1': f1, 'recall': recall, 'precision': precision})

        # OOF predictions
        oof_preds[val_idx] = val_preds

        # Test predictions (average over folds)
        test_fold_preds = []
        model.eval()
        with torch.no_grad():
            for (xb,) in test_loader:
                xb = xb.to(device)
                out = torch.sigmoid(model(xb).squeeze(1)).cpu().numpy()
                test_fold_preds.append(out)
        test_preds += np.concatenate(test_fold_preds) / N_SPLITS

    # Print CV summary
    print("\nCV SUMMARY:")
    for i, m in enumerate(metrics):
        print(f"Fold {i+1}: acc={m['acc']:.4f}, f1={m['f1']:.4f}, recall={m['recall']:.4f}, precision={m['precision']:.4f}")
    print("Mean acc: {:.4f}, Mean f1: {:.4f}".format(
        np.mean([m['acc'] for m in metrics]), np.mean([m['f1'] for m in metrics])
    ))

    # Save OOF and test predictions
    np.save(EMBEDDING_DIR / "oof_preds.npy", oof_preds)
    np.save(EMBEDDING_DIR / "test_preds.npy", test_preds)
    print(f"Saved OOF and test predictions to {EMBEDDING_DIR}")

    # Optionally, create a submission file
    submission = pd.DataFrame({
        "id": np.arange(len(test_preds)),  # replace with actual test IDs if available
        "prediction": test_preds
    })
    submission_path = EMBEDDING_DIR / "submission.csv"
    submission.to_csv(submission_path, index=False)
    print(f"Saved submission to {submission_path}")

if __name__ == "__main__":
    main()