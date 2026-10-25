"""Eight-state pixel Naive Bayes with deterministic moment-based deskew.

All probability estimation and prediction are implemented with NumPy. The public
constructor retains the course-required default alpha=1.0; main.py explicitly
uses the validation-selected alpha=0.1.
"""

import numpy as np


N_GRAY_STATES = 8


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


def deskew_pixels(X):
    """Remove horizontal slant using gray-level second moments, without labels."""
    X = np.asarray(X)
    output = np.empty(X.shape, dtype=np.float32)
    xx = np.arange(28, dtype=np.float64)[None, None, :]
    yy = np.arange(28, dtype=np.float64)[None, :, None]
    for start in range(0, len(X), 1024):
        images = X[start:start + 1024].reshape(-1, 28, 28).astype(np.float64)
        mass = images.sum(axis=(1, 2))
        safe_mass = np.maximum(mass, 1)
        cx = (images * xx).sum(axis=(1, 2)) / safe_mass
        cy = (images * yy).sum(axis=(1, 2)) / safe_mass
        dx, dy = xx - cx[:, None, None], yy - cy[:, None, None]
        mu11 = (images * dx * dy).sum(axis=(1, 2))
        mu02 = (images * dy * dy).sum(axis=(1, 2))
        slope = np.divide(mu11, mu02, out=np.zeros_like(mu11), where=mu02 > 1e-12)
        slope = np.clip(slope, -1, 1)
        source_x = xx + slope[:, None, None] * dy
        left = np.floor(source_x).astype(np.int64)
        right = left + 1
        fraction = source_x - left
        a = np.take_along_axis(images, np.clip(left, 0, 27), axis=2)
        b = np.take_along_axis(images, np.clip(right, 0, 27), axis=2)
        result = a * (1 - fraction) * ((left >= 0) & (left < 28))
        result += b * fraction * ((right >= 0) & (right < 28))
        output[start:start + len(images)] = result.reshape(-1, 784)
    return output


def discretize_pixels(X):
    """Fixed gray bins [0,32), [32,64), ..., [224,256), after deskew."""
    pixels = deskew_pixels(X)
    return np.clip(np.floor(pixels / 32.0), 0, N_GRAY_STATES - 1).astype(np.uint8)


class NaiveBayesClassifier:
    """Protocol v1. Categorical NB on 784 independently modeled pixel variables."""

    def __init__(self, alpha: float = 1.0):
        if not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha must be finite and strictly positive.")
        self.alpha = float(alpha)

    def fit(self, X, y):
        X = _validate_X(X)
        y = _validate_y(y, len(X))
        bins = discretize_pixels(X)
        self.classes_ = np.arange(10, dtype=np.int64)
        self.class_count_ = np.bincount(y, minlength=10)
        counts = np.zeros((10, 784, N_GRAY_STATES), dtype=np.int64)
        offsets = np.arange(784) * N_GRAY_STATES
        for digit in self.classes_:
            encoded = bins[y == digit].astype(np.int64) + offsets
            counts[digit] = np.bincount(encoded.ravel(), minlength=784 * N_GRAY_STATES).reshape(784, N_GRAY_STATES)
        # Each of the eight possible states receives alpha pseudocounts.
        self.feature_prob_ = (counts + self.alpha) / (
            self.class_count_[:, None, None] + N_GRAY_STATES * self.alpha
        )
        log_denominator = np.log(self.class_count_[:, None, None] + N_GRAY_STATES * self.alpha)
        self.feature_log_prob_ = np.log(counts + self.alpha) - log_denominator
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
        bins = discretize_pixels(X)
        columns = np.arange(784)[None, :]
        # Gather one state likelihood for EVERY pixel, including zero/background.
        for start in range(0, len(X), 512):
            batch = bins[start:start + 512]
            scores[start:start + len(batch)] = (
                self.feature_log_prob_[:, columns, batch].sum(axis=2).T + self.class_log_prior_
            )
        return scores

    def predict(self, X):
        scores = self._joint_log_likelihood(X)
        return self.classes_[np.argmax(scores, axis=1)]

    def predict_proba(self, X):
        scores = self._joint_log_likelihood(X)
        scores -= scores.max(axis=1, keepdims=True)
        probabilities = np.exp(scores)
        return probabilities / probabilities.sum(axis=1, keepdims=True)
