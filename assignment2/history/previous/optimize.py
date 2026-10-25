"""Training-only model selection. Never reads the original MNIST test files.

Candidate families, one fixed hand-written DAG, seed and ranking metric were
declared in optimization_protocol.json before the validation experiments.
Run: python optimize.py --data-dir ../assignment2-data/mnist
"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
from mnist_loader import _read_idx


SEED = 20261019
ALPHAS = (0.1, 1.0, 5.0)
PARENTS = (None, 0, 1, None, 3, 4, None, 6, 7, None, 9, 10)


def deskew_pixels(X):
    """Remove row-dependent horizontal slant using gray-level second moments.

    Bilinear interpolation in x only; off-image intensity is zero. Empty images
    have zero skew. Clipping skew to [-1,1] limits extreme degenerate inputs.
    """
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


def enclosed_background(binary):
    """Background pixels not 4-connected to the image boundary (NumPy flood fill)."""
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


def continuous_features(X, geometry="normalized", summaries="holes"):
    """Twelve semantic summaries; fixed geometry, no labels or learned structure.

    For normalized geometry, the >64 foreground box is sampled onto 21x21 by
    nearest-neighbor sampling. The first nine features are 7x7 cell gray means.
    The last two hole summaries are upper/lower enclosed-background fractions.
    """
    X = np.asarray(X)
    output = np.empty((len(X), 12), dtype=np.float64)
    for start in range(0, len(X), 1024):
        raw = X[start:start + 1024].reshape(-1, 28, 28)
        gray = raw.astype(np.float64) / 255.0
        binary = raw > 64
        rows, cols = binary.any(axis=2), binary.any(axis=1)
        nonempty = rows.any(axis=1)
        top, left = rows.argmax(axis=1), cols.argmax(axis=1)
        height = 28 - top - rows[:, ::-1].argmax(axis=1)
        width = 28 - left - cols[:, ::-1].argmax(axis=1)
        aspect = np.where(nonempty, width / height, 0.0)
        sample_r = top[:, None] + np.arange(21)[None, :] * height[:, None] // 21
        sample_c = left[:, None] + np.arange(21)[None, :] * width[:, None] // 21
        ids = np.arange(len(raw))[:, None, None]
        if geometry == "normalized":
            gray = gray[ids, sample_r[:, :, None], sample_c[:, None, :]]
            bounds = (0, 7, 14, 21)
        elif geometry == "raw":
            # Exactly the initial model's geometric definitions.
            bounds = (0, 9, 19, 28)
            binary128 = raw > 128
            r, c = binary128.any(axis=2), binary128.any(axis=1)
            h = 28 - r.argmax(axis=1) - r[:, ::-1].argmax(axis=1)
            w = 28 - c.argmax(axis=1) - c[:, ::-1].argmax(axis=1)
            aspect = np.where(r.any(axis=1), w / h, 0.0)
        else:
            raise ValueError("Unknown feature geometry")
        block = output[start:start + len(raw)]
        for row in range(3):
            for col in range(3):
                region = gray[:, bounds[row]:bounds[row + 1], bounds[col]:bounds[col + 1]]
                block[:, 3 * row + col] = region.mean(axis=(1, 2))
        block[:, 9] = aspect
        if summaries == "asymmetry":
            total = gray.sum(axis=(1, 2))
            for i, reflected in ((10, gray[:, :, ::-1]), (11, gray[:, ::-1, :])):
                difference = np.abs(gray - reflected).sum(axis=(1, 2))
                block[:, i] = np.divide(difference, 2 * total,
                                        out=np.zeros_like(total), where=total > 0)
        elif summaries == "holes" and geometry == "normalized":
            holes = enclosed_background(binary)
            normalized_holes = holes[ids, sample_r[:, :, None], sample_c[:, None, :]]
            block[:, 10] = normalized_holes[:, :10, :].mean(axis=(1, 2))
            block[:, 11] = normalized_holes[:, 10:, :].mean(axis=(1, 2))
        else:
            raise ValueError("Unknown shape summaries")
    return output


def bin_edges(values, mode, summaries):
    if mode == "quantile":
        edges = []
        for column in values.T:
            if np.mean(column == 0) >= 0.05 and np.any(column > 0):
                row = np.r_[1e-8, np.quantile(column[column > 0], [0.25, 0.5, 0.75])]
            else:
                row = np.quantile(column, [0.2, 0.4, 0.6, 0.8])
            # Quantile ties keep a documented deterministic five-state space.
            for i in range(1, 4):
                row[i] = max(row[i], np.nextafter(row[i - 1], np.inf))
            edges.append(row)
        return np.asarray(edges)
    density = (0.02, 0.08, 0.20, 0.40) if mode == "initial" else (0.05, 0.15, 0.30, 0.50)
    summary = (0.001, 0.03, 0.10, 0.22) if summaries == "holes" else (0.25, 0.40, 0.55, 0.70)
    return np.asarray([density] * 9 + [(0.45, 0.60, 0.75, 0.90), summary, summary])


def discretize_features(values, edges):
    return np.column_stack([np.digitize(values[:, i], row) for i, row in enumerate(edges)])


def scores_from_features(train_F, y, valid_F, alpha, parents=PARENTS):
    classes = np.bincount(y, minlength=10)
    scores = np.broadcast_to(np.log((classes + alpha) / (len(y) + 10 * alpha)), (len(valid_F), 10)).copy()
    for i, parent in enumerate(parents):
        if parent is None:
            code, shape = y * 5 + train_F[:, i], (10, 5)
        else:
            code, shape = (y * 5 + train_F[:, parent]) * 5 + train_F[:, i], (10, 5, 5)
        counts = np.bincount(code, minlength=int(np.prod(shape))).reshape(shape)
        table = np.log((counts + alpha) / (counts.sum(axis=-1, keepdims=True) + 5 * alpha))
        if parent is None:
            scores += table[:, valid_F[:, i]].T
        else:
            scores += table[:, valid_F[:, parent], valid_F[:, i]].T
    return scores


def metrics(y, prediction):
    cm = np.bincount(y * 10 + prediction, minlength=100).reshape(10, 10)
    tp = np.diag(cm).astype(float)
    denominator = cm.sum(axis=0) + cm.sum(axis=1)
    f1 = np.divide(2 * tp, denominator, out=np.zeros(10), where=denominator > 0)
    return {"accuracy": float(tp.sum() / len(y)), "macro_f1": float(f1.mean())}


def pixel_counts(X, y, n_states, threshold=None):
    if threshold is not None:
        bins = (X > threshold).astype(np.uint8)
    else:
        bins = np.minimum(np.floor(X.astype(np.float64) * n_states / 256), n_states - 1).astype(np.uint8)
    counts = np.zeros((10, 784, n_states), dtype=np.int64)
    offsets = np.arange(784) * n_states
    for digit in range(10):
        encoded = bins[y == digit].astype(np.int64) + offsets
        counts[digit] = np.bincount(encoded.ravel(), minlength=784 * n_states).reshape(784, n_states)
    return counts


def pixel_predict(counts, y, valid, alpha, n_states, threshold=None):
    class_counts = np.bincount(y, minlength=10)
    table = np.log((counts + alpha) / (class_counts[:, None, None] + n_states * alpha))
    prior = np.log((class_counts + alpha) / (len(y) + 10 * alpha))
    prediction = np.empty(len(valid), dtype=np.int64)
    columns = np.arange(784)[None, :]
    for start in range(0, len(valid), 512):
        batch = valid[start:start + 512]
        bins = ((batch > threshold).astype(np.int64) if threshold is not None else
                np.minimum(np.floor(batch.astype(float) * n_states / 256), n_states - 1).astype(np.int64))
        scores = table[:, columns, bins].sum(axis=2).T + prior
        prediction[start:start + len(batch)] = scores.argmax(axis=1)
    return prediction


def load_training_only(data_dir):
    # Intentionally do not call load_mnist: it would read the held-out test files.
    X = _read_idx(data_dir / "train-images-idx3-ubyte.gz")
    y = _read_idx(data_dir / "train-labels-idx1-ubyte.gz").astype(np.int64)
    if X.shape != (60000, 28, 28) or y.shape != (60000,):
        raise ValueError("Expected the full original MNIST training split")
    return X.reshape(-1, 784), y


def validation_split(y):
    rng = np.random.default_rng(SEED)
    valid = []
    for digit in range(10):
        valid.extend(rng.permutation(np.flatnonzero(y == digit))[:1000].tolist())
    valid = np.sort(valid)
    mask = np.ones(len(y), dtype=bool)
    mask[valid] = False
    return np.flatnonzero(mask), valid


def run(data_dir, output_dir):
    X, y = load_training_only(data_dir)
    train_idx, valid_idx = validation_split(y)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(output_dir / "validation_indices.csv", valid_idx, fmt="%d", header="training_set_index", comments="")
    rows = []
    for deskew in (False, True):
        pixels = deskew_pixels(X) if deskew else X
        for n_states, threshold in ((2, 0), (2, 64), (2, 128), (2, 192), (4, None), (8, None)):
            counts = pixel_counts(pixels[train_idx], y[train_idx], n_states, threshold)
            for alpha in ALPHAS:
                pred = pixel_predict(counts, y[train_idx], pixels[valid_idx], alpha, n_states, threshold)
                row = {"model": "naive_bayes", "deskew": deskew, "n_states": n_states,
                       "threshold": threshold, "alpha": alpha, **metrics(y[valid_idx], pred)}
                rows.append(row)
                print(json.dumps(row), flush=True)
    for geometry, summaries, modes in (
        ("raw", "asymmetry", ("initial", "quantile")),
        ("normalized", "asymmetry", ("uniform", "quantile")),
        ("normalized", "holes", ("uniform", "quantile")),
    ):
        features = continuous_features(X, geometry, summaries)
        for mode in modes:
            edges = bin_edges(features[train_idx], mode, summaries)
            states = discretize_features(features, edges)
            for alpha in ALPHAS:
                pred = scores_from_features(states[train_idx], y[train_idx], states[valid_idx], alpha).argmax(axis=1)
                row = {"model": "bayes_net", "geometry": geometry, "summaries": summaries,
                       "binning": mode, "alpha": alpha, "edges": edges.tolist(), **metrics(y[valid_idx], pred)}
                rows.append(row)
                print(json.dumps({k:v for k,v in row.items() if k != 'edges'}), flush=True)
    ranking = lambda row: (row["macro_f1"], row["accuracy"])
    selected = {model: max((row for row in rows if row["model"] == model), key=ranking)
                for model in ("naive_bayes", "bayes_net")}
    best_bn = selected['bayes_net']
    selected_values = continuous_features(X, best_bn['geometry'], best_bn['summaries'])
    selected_states = discretize_features(selected_values, np.asarray(best_bn['edges']))
    no_edges = scores_from_features(selected_states[train_idx], y[train_idx], selected_states[valid_idx],
                                    best_bn['alpha'], parents=(None,) * 12).argmax(axis=1)
    result = {
        "seed": SEED, "training_samples": len(train_idx), "validation_samples": len(valid_idx),
        "validation_class_counts": np.bincount(y[valid_idx], minlength=10).tolist(),
        "ranking": "macro_f1, then accuracy", "test_files_read": False,
        "manual_feature_parents_zero_based": list(PARENTS), "automatic_structure_learning": False,
        "candidate_count": len(rows), "candidates": rows, "selected": selected,
        "selected_bn_no_feature_edges_ablation": metrics(y[valid_idx], no_edges),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (output_dir / "selection.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("SELECTED", json.dumps(selected), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "results/optimization")
    args = parser.parse_args()
    run(args.data_dir, args.output_dir)
