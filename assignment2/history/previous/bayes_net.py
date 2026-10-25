"""Hand-designed, fully observed discrete Bayesian network, using only NumPy.

Every feature has Y as a parent. FEATURE_PARENTS gives its optional additional
feature parent. Indices are zero based here; the report uses F1 through F12.
"""

import numpy as np


FEATURE_NAMES = (
    "normalized_top_left_density", "normalized_top_center_density", "normalized_top_right_density",
    "normalized_middle_left_density", "normalized_middle_center_density", "normalized_middle_right_density",
    "normalized_bottom_left_density", "normalized_bottom_center_density", "normalized_bottom_right_density",
    "bounding_box_aspect_ratio", "upper_enclosed_background_fraction", "lower_enclosed_background_fraction",
)
FEATURE_PARENTS = (None, 0, 1, None, 3, 4, None, 6, 7, None, 9, 10)
N_STATES = 5
DENSITY_EDGES = (0.05, 0.15, 0.30, 0.50)
ASPECT_EDGES = (0.45, 0.60, 0.75, 0.90)
HOLE_EDGES = (0.001, 0.03, 0.10, 0.22)
REGION_BOUNDARIES = (0, 7, 14, 21)
FOREGROUND_THRESHOLD = 64
NORMALIZED_SIZE = 21


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


def _enclosed_background(binary):
    """Find background not 4-connected to the image boundary; no external library."""
    background = ~binary
    outside = np.zeros_like(background)
    outside[:, 0, :] = background[:, 0, :]
    outside[:, -1, :] = background[:, -1, :]
    outside[:, :, 0] = background[:, :, 0]
    outside[:, :, -1] = background[:, :, -1]
    while True:
        grown = outside.copy()
        grown[:, 1:, :] |= outside[:, :-1, :]
        grown[:, :-1, :] |= outside[:, 1:, :]
        grown[:, :, 1:] |= outside[:, :, :-1]
        grown[:, :, :-1] |= outside[:, :, 1:]
        grown &= background
        if np.array_equal(grown, outside):
            break
        outside = grown
    return background & ~outside


def _continuous_features(X):
    """Nine box-normalized gray densities, original aspect, upper/lower holes.

    The >64 foreground box is sampled onto 21x21 by nearest-neighbor sampling.
    Hole connectivity is computed on the original 28x28 thresholded image, then
    the resulting mask is sampled using the exact same box coordinate mapping.
    """
    X = _validate_X(X)
    features = np.empty((len(X), len(FEATURE_NAMES)), dtype=np.float64)
    for start in range(0, len(X), 1024):
        raw = X[start:start + 1024].reshape(-1, 28, 28)
        gray = raw.astype(np.float64) / 255.0
        binary = raw > FOREGROUND_THRESHOLD
        occupied_rows = binary.any(axis=2)
        occupied_cols = binary.any(axis=1)
        nonempty = occupied_rows.any(axis=1)
        top, left = occupied_rows.argmax(axis=1), occupied_cols.argmax(axis=1)
        height = 28 - top - occupied_rows[:, ::-1].argmax(axis=1)
        width = 28 - left - occupied_cols[:, ::-1].argmax(axis=1)
        aspect = np.where(nonempty, width / height, 0.0)
        sample_r = top[:, None] + np.arange(NORMALIZED_SIZE)[None, :] * height[:, None] // NORMALIZED_SIZE
        sample_c = left[:, None] + np.arange(NORMALIZED_SIZE)[None, :] * width[:, None] // NORMALIZED_SIZE
        ids = np.arange(len(raw))[:, None, None]
        normalized = gray[ids, sample_r[:, :, None], sample_c[:, None, :]]
        block = features[start:start + len(raw)]
        for row in range(3):
            for col in range(3):
                region = normalized[:, REGION_BOUNDARIES[row]:REGION_BOUNDARIES[row + 1],
                                    REGION_BOUNDARIES[col]:REGION_BOUNDARIES[col + 1]]
                block[:, row * 3 + col] = region.mean(axis=(1, 2))
        block[:, 9] = aspect
        holes = _enclosed_background(binary)
        normalized_holes = holes[ids, sample_r[:, :, None], sample_c[:, None, :]]
        block[:, 10] = normalized_holes[:, :10, :].mean(axis=(1, 2))
        block[:, 11] = normalized_holes[:, 10:, :].mean(axis=(1, 2))
    return features


def extract_features(X):
    """Return fixed, batch-independent integer states, shape (n_samples,12).

    Each feature has five states. Ties go to the higher bin. No learned state,
    labels, quantiles or caller-side feature preprocessing are required.
    """
    continuous = _continuous_features(X)
    edges = [DENSITY_EDGES] * 9 + [ASPECT_EDGES, HOLE_EDGES, HOLE_EDGES]
    return np.column_stack([np.digitize(continuous[:, i], row) for i, row in enumerate(edges)])


class BayesNetClassifier:
    """Protocol v1. Y is the root; each feature has at most two total parents."""

    def __init__(self, alpha: float = 1.0):
        if not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha must be finite and strictly positive.")
        self.alpha = float(alpha)

    def fit(self, X, y):
        X = _validate_X(X)
        y = _validate_y(y, len(X))
        features = extract_features(X)
        self.classes_ = np.arange(10, dtype=np.int64)
        self.class_count_ = np.bincount(y, minlength=10)
        self.class_log_prior_ = np.log(
            (self.class_count_ + self.alpha) / (len(y) + 10 * self.alpha)
        )
        self.feature_parents_ = FEATURE_PARENTS
        self.cpts_ = []
        for index, parent in enumerate(self.feature_parents_):
            if parent is None:
                encoded = y * N_STATES + features[:, index]
                shape = (10, N_STATES)
            else:
                encoded = (y * N_STATES + features[:, parent]) * N_STATES + features[:, index]
                shape = (10, N_STATES, N_STATES)
            counts = np.bincount(encoded, minlength=int(np.prod(shape))).reshape(shape)
            # Normalize along the CHILD-state axis for every parent assignment.
            table = (counts + self.alpha) / (
                counts.sum(axis=-1, keepdims=True) + N_STATES * self.alpha
            )
            self.cpts_.append(table)
        self.log_cpts_ = [np.log(table) for table in self.cpts_]
        self.n_features_in_ = 784
        return self

    def _joint_log_likelihood(self, X):
        if not hasattr(self, "class_log_prior_"):
            raise RuntimeError("Call fit before prediction.")
        features = extract_features(X)
        scores = np.broadcast_to(self.class_log_prior_, (len(features), 10)).copy()
        for index, parent in enumerate(self.feature_parents_):
            table = self.log_cpts_[index]
            if parent is None:
                scores += table[:, features[:, index]].T
            else:
                scores += table[:, features[:, parent], features[:, index]].T
        return scores

    def predict(self, X):
        scores = self._joint_log_likelihood(X)
        return self.classes_[np.argmax(scores, axis=1)]

    def predict_proba(self, X):
        scores = self._joint_log_likelihood(X)
        scores -= scores.max(axis=1, keepdims=True)
        probabilities = np.exp(scores)
        return probabilities / probabilities.sum(axis=1, keepdims=True)
