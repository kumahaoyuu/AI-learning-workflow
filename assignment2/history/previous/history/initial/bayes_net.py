"""Hand-designed, fully observed discrete Bayesian network, using only NumPy.

Every feature has Y as a parent. FEATURE_PARENTS gives its optional additional
feature parent. Indices are zero based here; the report uses F1 through F12.
"""

import numpy as np


FEATURE_NAMES = (
    "top_left_density", "top_center_density", "top_right_density",
    "middle_left_density", "middle_center_density", "middle_right_density",
    "bottom_left_density", "bottom_center_density", "bottom_right_density",
    "bounding_box_aspect_ratio", "left_right_asymmetry", "top_bottom_asymmetry",
)
FEATURE_PARENTS = (None, 0, 1, None, 3, 4, None, 6, 7, None, 9, 10)
N_STATES = 5
DENSITY_EDGES = (0.02, 0.08, 0.20, 0.40)
ASPECT_EDGES = (0.45, 0.60, 0.75, 0.90)
ASYMMETRY_EDGES = (0.25, 0.40, 0.55, 0.70)
REGION_BOUNDARIES = (0, 9, 19, 28)


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


def extract_features(X):
    """Return integer states, shape (n_samples, 12), each in {0,1,2,3,4}.

    Thresholds are fixed constants, independent of the supplied batch or labels.
    Bins are [0,e0), [e0,e1), ..., [e3,infinity); ties go to the higher bin.
    Empty images have aspect ratio zero and both asymmetry values zero.
    """
    X = _validate_X(X)
    features = np.empty((len(X), len(FEATURE_NAMES)), dtype=np.int64)
    for start in range(0, len(X), 2048):
        images = X[start : start + 2048].reshape(-1, 28, 28).astype(np.float64) / 255.0
        block = features[start : start + len(images)]
        for row in range(3):
            for col in range(3):
                region = images[:, REGION_BOUNDARIES[row] : REGION_BOUNDARIES[row + 1],
                                REGION_BOUNDARIES[col] : REGION_BOUNDARIES[col + 1]]
                block[:, row * 3 + col] = np.digitize(region.mean(axis=(1, 2)), DENSITY_EDGES)
        binary = images > (128.0 / 255.0)
        occupied_rows = binary.any(axis=2)
        occupied_cols = binary.any(axis=1)
        nonempty = occupied_rows.any(axis=1)
        height = 28 - occupied_rows.argmax(axis=1) - occupied_rows[:, ::-1].argmax(axis=1)
        width = 28 - occupied_cols.argmax(axis=1) - occupied_cols[:, ::-1].argmax(axis=1)
        aspect = np.where(nonempty, width / height, 0.0)
        block[:, 9] = np.digitize(aspect, ASPECT_EDGES)
        total_ink = images.sum(axis=(1, 2))
        for feature_index, reflected in ((10, images[:, :, ::-1]), (11, images[:, ::-1, :])):
            difference = np.abs(images - reflected).sum(axis=(1, 2))
            asymmetry = np.divide(difference, 2 * total_ink,
                                  out=np.zeros_like(total_ink), where=total_ink > 0)
            block[:, feature_index] = np.digitize(asymmetry, ASYMMETRY_EDGES)
    return features


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
