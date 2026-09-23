"""Shared evaluation helpers: Macro-F1 (primary) + accuracy + confusion matrix (secondary)."""

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def evaluate(y_true, y_pred, labels=None):
    return {
        "macro_f1": f1_score(y_true, y_pred, average="macro", labels=labels),
        "accuracy": accuracy_score(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels),
    }
