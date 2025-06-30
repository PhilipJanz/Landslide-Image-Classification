
# Custom surrogate loss maximizing F1
import torch


def f1_loss(y_pred, y_true, smooth=1e-6):
    y_pred = torch.sigmoid(y_pred)
    TP = (y_pred * y_true).sum()
    precision = TP / (y_pred.sum() + smooth)
    recall = TP / (y_true.sum() + smooth)
    f1 = 2 * (precision * recall) / (precision + recall + smooth)
    return 1 - f1
