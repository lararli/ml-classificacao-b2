"""Custom models. Must implement fit(X,y) and predict(X). Reference in YAML as models.custom_models.ClassName."""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class RuleBasedClassifier(BaseEstimator, ClassifierMixin):
    """Deterministic: rejects if has loan defaults, high debt ratio, or low credit score."""

    def __init__(self, income_threshold=0.35, min_credit_score=600, random_state=None):
        self.income_threshold = income_threshold
        self.min_credit_score = min_credit_score
        self.random_state = random_state

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        return self

    def predict(self, X):
        X = np.asarray(X)
        preds = np.ones(X.shape[0], dtype=int)
        if X.shape[1] > 8:
            preds[X[:, -1] > 0] = 0          # loan defaults = yes
        if X.shape[1] > 5:
            preds[X[:, 5] > self.income_threshold] = 0   # high debt
        if X.shape[1] > 7:
            preds[X[:, 7] < self.min_credit_score] = 0   # low score
        return preds

    def predict_proba(self, X):
        p = self.predict(X).astype(float)
        return np.column_stack([1 - p, p])
