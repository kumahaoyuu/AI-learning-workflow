"""Bernoulli Naive Bayes implemented with NumPy; no classifier library."""

import numpy as np


def _validate_X(X):
    X = np.asarray(X)
    if X.ndim != 2 or X.shape[1] != 784:
        raise ValueError("X must have shape (n_samples, 784).")
    if not np.issubdtype(X.dtype, np.number) or np.iscomplexobj(X):
        raise ValueError("X must contain real numeric pixels.")
    if not np.isfinite(X).all() or (X.size and (X.min() < 0 or X.max() > 255)):
        raise ValueError("X pixels must be finite and in [0, 255].")
    return X


def _validate_y(y, n):
    y = np.asarray(y)
    if y.shape != (n,) or n == 0:
        raise ValueError("y must have shape (n_samples,); training cannot be empty.")
    if not np.issubdtype(y.dtype, np.number) or np.iscomplexobj(y):
        raise ValueError("y must contain integer labels in [0, 9].")
    if not np.isfinite(y).all() or np.any((y < 0) | (y > 9) | (y != np.floor(y))):
        raise ValueError("y must contain integer labels in [0, 9].")
    return y.astype(np.int64, copy=False)


class NaiveBayesClassifier:
    """Protocol v1. Raw pixels > 128 are foreground, all other pixels background."""

    def __init__(self, alpha: float = 1.0):
        if not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha must be finite and strictly positive.")
        self.alpha = float(alpha)

    def fit(self, X, y):
        X = _validate_X(X)
        y = _validate_y(y, len(X))
        foreground = X > 128
        self.classes_ = np.arange(10, dtype=np.int64)
        self.class_count_ = np.bincount(y, minlength=10)
        # Both values of every Bernoulli variable receive alpha pseudocounts.
        counts = np.zeros((10, 784), dtype=np.float64)
        for digit in self.classes_:
            counts[digit] = foreground[y == digit].sum(axis=0)
        self.feature_prob_ = (counts + self.alpha) / (
            self.class_count_[:, None] + 2 * self.alpha
        )
        log_denominator = np.log(self.class_count_[:, None] + 2 * self.alpha)
        # Derive both logs from counts: 1 - feature_prob_ may round to zero when
        # alpha is tiny, even though the smoothed background count is positive.
        self.feature_log_prob_ = np.log(counts + self.alpha) - log_denominator
        self.feature_log_neg_prob_ = np.log(
            self.class_count_[:, None] - counts + self.alpha
        ) - log_denominator
        self.class_log_prior_ = np.log(
            (self.class_count_ + self.alpha) / (len(y) + 10 * self.alpha)
        )
        self.n_features_in_ = 784
        return self

    def _joint_log_likelihood(self, X):
        if not hasattr(self, "class_log_prior_"):
            raise RuntimeError("Call fit before prediction.")
        X = _validate_X(X)
        scores = np.empty((len(X), 10), dtype=np.float64)
        weight = self.feature_log_prob_ - self.feature_log_neg_prob_
        bias = self.class_log_prior_ + self.feature_log_neg_prob_.sum(axis=1)
        # This includes background evidence as well as foreground evidence.
        for start in range(0, len(X), 2048):
            binary = (X[start : start + 2048] > 128).astype(np.float64)
            scores[start : start + 2048] = binary @ weight.T + bias
        return scores

    def predict(self, X):
        scores = self._joint_log_likelihood(X)
        return self.classes_[np.argmax(scores, axis=1)]

    def predict_proba(self, X):
        scores = self._joint_log_likelihood(X)
        scores -= scores.max(axis=1, keepdims=True)
        probabilities = np.exp(scores)
        return probabilities / probabilities.sum(axis=1, keepdims=True)
