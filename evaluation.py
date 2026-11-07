"""Shared evaluation contract: malignant=1, benign=0."""
import warnings
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef


def _mcc(y, predictions):
    # matthews_corrcoef has no `labels` argument; for one-class uploads it warns and returns 0.0.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        return float(matthews_corrcoef(y, predictions))


def evaluate(model, X, y):
    predictions = model.predict(X)
    if hasattr(model, 'predict_proba'):
        positive_column = list(model.classes_).index(1)
        scores = model.predict_proba(X)[:, positive_column]
    else:
        scores = model.decision_function(X)
    return {
        'Accuracy': float(accuracy_score(y, predictions)),
        'AUC': float(roc_auc_score(y, scores)) if len(np.unique(y)) == 2 else None,
        'Precision': float(precision_score(y, predictions, pos_label=1, zero_division=0)),
        'Recall': float(recall_score(y, predictions, pos_label=1, zero_division=0)),
        'F1': float(f1_score(y, predictions, pos_label=1, zero_division=0)),
        'MCC': _mcc(y, predictions),
    }, predictions
