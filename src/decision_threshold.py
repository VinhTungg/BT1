"""Decision thresholds fitted using training out-of-fold scores only."""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.metrics import f1_score, recall_score


def select_threshold(labels, scores):
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=float)
    if labels.shape != scores.shape or labels.ndim != 1 or not len(labels):
        raise ValueError("Labels and scores must be matching nonempty vectors")
    if set(labels) != {0, 1} or not np.isfinite(scores).all():
        raise ValueError("Threshold selection requires both classes and finite scores")
    candidates = np.unique(np.r_[0.0, np.quantile(scores, np.linspace(0.5, 0.999, 101))])
    records = []
    for threshold in candidates:
        predicted = (scores >= threshold).astype(int)
        records.append(dict(threshold=float(threshold),
                            macro_f1=float(f1_score(labels, predicted, average='macro', zero_division=0)),
                            f1_toxic=float(f1_score(labels, predicted, zero_division=0)),
                            recall_toxic=float(recall_score(labels, predicted, zero_division=0))))
    best = max(records, key=lambda r: (r['macro_f1'], r['f1_toxic'], r['recall_toxic'], -abs(r['threshold'])))
    return best, records


class DecisionThresholdClassifier(ClassifierMixin, BaseEstimator):
    """Wrap a fitted binary estimator without refitting its feature extraction."""

    def __init__(self, estimator, threshold=0.0):
        self.estimator = estimator
        self.threshold = threshold

    @property
    def classes_(self):
        return getattr(self, 'estimator_', self.estimator).classes_

    def fit(self, X, y):
        self.estimator_ = clone(self.estimator).fit(X, y)
        return self

    def decision_function(self, X):
        scores = getattr(self, 'estimator_', self.estimator).decision_function(X)
        if np.asarray(scores).ndim != 1:
            raise ValueError('Only binary decision functions are supported')
        return scores

    def predict(self, X):
        if list(self.classes_) != [0, 1]:
            raise ValueError('Expected classes [0, 1]')
        return (self.decision_function(X) >= self.threshold).astype(int)
