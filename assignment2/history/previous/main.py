"""Reproduce the frozen optimized MNIST experiment; training-only selection.

Run: python main.py --data-dir ../assignment2-data/mnist
The original course-provided MNIST loader is included unchanged and attributed
in README.md and report.md. The two classifiers are independently loadable.
"""

import argparse
import csv
from datetime import datetime, timezone, timedelta
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np

from naive_bayes import NaiveBayesClassifier, N_GRAY_STATES
from bayes_net import (
    BayesNetClassifier, extract_features, FEATURE_NAMES, FEATURE_PARENTS,
    DENSITY_EDGES, ASPECT_EDGES, HOLE_EDGES, REGION_BOUNDARIES, N_STATES,
    FOREGROUND_THRESHOLD, NORMALIZED_SIZE,
)
from mnist_loader import load_mnist


DATA_HASHES = {
    "train-images-idx3-ubyte.gz": "440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609",
    "train-labels-idx1-ubyte.gz": "3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c",
    "t10k-images-idx3-ubyte.gz": "8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6",
    "t10k-labels-idx1-ubyte.gz": "f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6",
}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_data(data_dir):
    """Check against fixed hashes from the course manifest, before training."""
    verified = {}
    for name, expected in DATA_HASHES.items():
        actual = sha256_file(data_dir / name)
        if actual != expected:
            raise ValueError(f"SHA-256 mismatch for {name}: {actual}")
        verified[name] = actual
    return verified


def classification_metrics(y_true, y_pred):
    """Rows = true labels, columns = predictions. Macro average over all 10 digits.

    Undefined precision/recall/F1 is explicitly assigned zero. F1 is computed per
    class before averaging, rather than from the two macro-averaged rates.
    """
    true = np.asarray(y_true)
    pred = np.asarray(y_pred)
    if true.ndim != 1 or pred.shape != true.shape or not len(true):
        raise ValueError("Metrics need equal, nonempty one-dimensional label arrays.")
    for labels in (true, pred):
        if (not np.issubdtype(labels.dtype, np.number) or np.iscomplexobj(labels)
                or not np.isfinite(labels).all()
                or np.any((labels < 0) | (labels > 9) | (labels != np.floor(labels)))):
            raise ValueError("Labels must be integers in [0, 9].")
    true, pred = true.astype(np.int64), pred.astype(np.int64)
    cm = np.bincount(true * 10 + pred, minlength=100).reshape(10, 10)
    tp = np.diag(cm).astype(np.float64)
    actual, predicted = cm.sum(axis=1), cm.sum(axis=0)
    precision = np.divide(tp, predicted, out=np.zeros(10), where=predicted > 0)
    recall = np.divide(tp, actual, out=np.zeros(10), where=actual > 0)
    f1 = np.divide(2 * tp, actual + predicted, out=np.zeros(10), where=(actual + predicted) > 0)
    return {
        "accuracy": float(tp.sum() / len(true)),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "confusion_matrix": cm.tolist(),
        "precision_per_class": precision.tolist(),
        "recall_per_class": recall.tolist(),
        "f1_per_class": f1.tolist(),
        "support_per_class": actual.tolist(),
        "correct": int(tp.sum()),
        "errors": int(len(true) - tp.sum()),
    }


def most_confused_pairs(cm, limit=5):
    cm = np.asarray(cm)
    pairs = [(int(cm[a, b] + cm[b, a]), a, b) for a in range(10) for b in range(a + 1, 10)]
    pairs.sort(key=lambda item: (-item[0], item[1], item[2]))
    return [{"digits": [a, b], "total": total, "a_to_b": int(cm[a, b]),
             "b_to_a": int(cm[b, a])} for total, a, b in pairs[:limit]]


def _plotting():
    # Cache stays inside the assignment directory; the core models need only NumPy.
    os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "svg.fonttype": "none", "axes.spines.top": False,
                         "axes.spines.right": False})
    return plt, matplotlib.__version__


def _save_figure(fig, base):
    """Fail on clipped text or overlapping tick labels before export."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    canvas = fig.bbox
    texts = list(fig.texts)
    tick_groups = []
    for ax in fig.axes:
        texts.extend([*ax.texts, ax.title])
        if not ax.axison:
            continue
        for axis in (ax.xaxis, ax.yaxis):
            texts.extend([axis.label, axis.get_offset_text()])
            low, high = sorted(axis.get_view_interval())
            # Matplotlib can create off-range tick objects that it never draws.
            # Ignore those objects instead of mistaking them for clipped labels.
            for side in ("label1", "label2"):
                labels = [getattr(tick, side) for tick in axis.get_major_ticks()
                          if low <= tick.get_loc() <= high]
                texts.extend(labels)
                tick_groups.append(labels)
    for text in texts:
        if not text.get_visible() or not text.get_text().strip():
            continue
        box = text.get_window_extent(renderer)
        if box.x0 < canvas.x0 - 2 or box.y0 < canvas.y0 - 2 or box.x1 > canvas.x1 + 2 or box.y1 > canvas.y1 + 2:
            raise RuntimeError(f"Figure text outside canvas: {text.get_text()}")
    for labels in tick_groups:
        visible = [label.get_window_extent(renderer) for label in labels
                   if label.get_visible() and label.get_text()]
        if any(a.overlaps(b) for a, b in zip(visible, visible[1:])):
            raise RuntimeError("Overlapping tick labels in figure.")
    fig.savefig(base.with_suffix(".png"), dpi=300)
    fig.savefig(base.with_suffix(".svg"))


def plot_confusion_matrices(plt, results, figure_dir):
    vmax = max(max(max(row) for row in result["confusion_matrix"]) for result in results.values())
    for key, result in results.items():
        fig, ax = plt.subplots(figsize=(6.3, 5.5), layout="constrained")
        cm = np.asarray(result["confusion_matrix"])
        image = ax.imshow(cm, cmap="viridis", vmin=0, vmax=vmax)
        for row in range(10):
            for col in range(10):
                ax.text(col, row, str(cm[row, col]), ha="center", va="center", fontsize=8,
                        color="black" if cm[row, col] > 0.55 * vmax else "white")
        ax.set(xticks=np.arange(10), yticks=np.arange(10), xlabel="Predicted digit",
               ylabel="True digit", title=f"{result['display_name']} | n = 10,000")
        fig.colorbar(image, ax=ax, label="Number of test images", shrink=0.92)
        _save_figure(fig, figure_dir / f"confusion_{key}")
        plt.close(fig)


def plot_errors(plt, images, y_true, predictions, results, figure_dir):
    for key, y_pred in predictions.items():
        errors = np.flatnonzero(y_pred != y_true)[:6]
        results[key]["example_indices"] = errors.tolist()
        fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.2), layout="constrained")
        fig.suptitle(f"{results[key]['display_name']}: first six errors (test order)", fontsize=12)
        for ax, index in zip(axes.flat, errors):
            ax.imshow(images[index], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
            ax.set_title(f"Index {index}: true {y_true[index]}, predicted {y_pred[index]}", fontsize=9)
            ax.axis("off")
        for ax in list(axes.flat)[len(errors):]:
            ax.axis("off")
        _save_figure(fig, figure_dir / f"errors_{key}")
        plt.close(fig)


def plot_dag(plt, figure_dir):
    from matplotlib.patches import Circle, FancyArrowPatch
    fig, ax = plt.subplots(figsize=(16, 4.2), layout="constrained")
    ax.set(xlim=(-0.6, 11.6), ylim=(-0.05, 2.9))
    ax.axis("off")
    root = (5.5, 2.38)
    leaf_y = 0.62
    labels = ["TL", "TC", "TR", "ML", "MC", "MR", "BL", "BC", "BR", "Aspect", "Upper hole", "Lower hole"]
    for index in range(12):
        ax.add_patch(FancyArrowPatch(root, (index, leaf_y), arrowstyle="-|>", mutation_scale=10,
                                    shrinkA=16, shrinkB=16, linewidth=0.85, color="0.60", zorder=1))
    for child, parent in enumerate(FEATURE_PARENTS):
        if parent is not None:
            ax.add_patch(FancyArrowPatch((parent + 0.20, leaf_y), (child - 0.20, leaf_y),
                                        arrowstyle="-|>", mutation_scale=13, color="#0072B2",
                                        linewidth=1.8, zorder=2))
    ax.add_patch(Circle(root, 0.18, fc="#F0E442", ec="0.2", zorder=3))
    ax.text(*root, "Y", ha="center", va="center", weight="bold")
    for index, label in enumerate(labels):
        ax.add_patch(Circle((index, leaf_y), 0.20, fc="#E6F2F8", ec="#0072B2", zorder=3))
        ax.text(index, leaf_y, f"F{index + 1}", ha="center", va="center", fontsize=9, zorder=4)
        ax.text(index, 0.23, label, ha="center", va="center", fontsize=9)
    ax.text(-0.45, 2.7, "Hand-designed DAG: 13 nodes, 20 edges, no hidden variables", fontsize=13)
    ax.text(-0.45, 2.3, "Gray: Y to each feature (12 edges)\nBlue: feature dependencies (8 edges)",
            fontsize=10, color="0.2", va="top")
    _save_figure(fig, figure_dir / "bayes_network")
    plt.close(fig)


def find_data_dir():
    here = Path(__file__).resolve().parent
    candidates = (Path.cwd() / "assignment2-data/mnist", here.parent / "assignment2-data/mnist",
                  here.parent.parent / "assignment2-data/mnist")
    return next((p for p in candidates if p.is_dir()), candidates[1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=find_data_dir())
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--alpha", type=float, default=0.1,
                        help="Explicit validation-selected alpha (class constructors still default to 1.0)")
    args = parser.parse_args()
    if not np.isfinite(args.alpha) or args.alpha <= 0:
        parser.error("--alpha must be finite and positive")
    source_dir = Path(__file__).resolve().parent
    frozen = json.loads((source_dir / "selected_config.json").read_text(encoding="utf-8"))
    for name, expected in frozen["core_source_sha256"].items():
        if sha256_file(source_dir / name) != expected:
            raise ValueError(f"Core source changed after configuration freeze: {name}")
    if sha256_file(source_dir / "results/optimization/selection.json") != frozen["selection_sha256"]:
        raise ValueError("Validation selection record changed after configuration freeze")
    result_dir, figure_dir = args.output_dir / "results", args.output_dir / "figures"
    result_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    hashes = verify_data(args.data_dir)
    X_train, y_train, X_test, y_test = load_mnist(args.data_dir)
    train_flat, test_flat = X_train.reshape(-1, 784), X_test.reshape(-1, 784)
    results, predictions = {}, {}
    for key, name, model in (
        ("naive_bayes", "Deskewed Categorical NB", NaiveBayesClassifier(args.alpha)),
        ("bayes_net", "Small Bayesian Network", BayesNetClassifier(args.alpha)),
    ):
        start = time.perf_counter()
        model.fit(train_flat, y_train)
        fitted = time.perf_counter()
        pred = model.predict(test_flat)
        predicted = time.perf_counter()
        predictions[key] = pred
        results[key] = classification_metrics(y_test, pred)
        results[key].update({"display_name": name, "fit_seconds": fitted - start,
                             "predict_seconds": predicted - fitted})
        results[key]["most_confused_pairs"] = most_confused_pairs(results[key]["confusion_matrix"])
        print(f"{name}: accuracy={results[key]['accuracy']:.4%}, "
              f"macro_f1={results[key]['macro_f1']:.6f}, "
              f"fit={fitted-start:.3f}s, predict={predicted-fitted:.3f}s", flush=True)
    plt, matplotlib_version = _plotting()
    plot_confusion_matrices(plt, results, figure_dir)
    plot_errors(plt, X_test, y_test, predictions, results, figure_dir)
    plot_dag(plt, figure_dir)
    # Test labels are used only for evaluation and error analysis above.
    features = extract_features(train_flat)
    profile = {
        "train_shape": list(X_train.shape), "test_shape": list(X_test.shape),
        "image_dtype": str(X_train.dtype), "pixel_range": [int(X_train.min()), int(X_train.max())],
        "missing_pixels": int(np.count_nonzero(~np.isfinite(X_train))),
        "train_class_counts": np.bincount(y_train, minlength=10).tolist(),
        "test_class_counts": np.bincount(y_test, minlength=10).tolist(),
        "train_feature_state_counts": {name: np.bincount(features[:, i], minlength=N_STATES).tolist()
                                       for i, name in enumerate(FEATURE_NAMES)},
    }
    metadata = {
        "created_at_shanghai": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "python": platform.python_version(), "numpy": np.__version__, "matplotlib": matplotlib_version,
        "platform": platform.platform(), "alpha": args.alpha,
        "train_samples": len(y_train), "test_samples": len(y_test),
        "data_sha256": hashes, "data_profile": profile,
        "source_sha256": {name: sha256_file(source_dir / name) for name in
                          ("naive_bayes.py", "bayes_net.py", "main.py", "mnist_loader.py", "requirements.txt", "optimize.py")},
        "naive_bayes_preprocessing": "Moment deskew; 8 equal-width gray bins",
        "naive_bayes_n_states": N_GRAY_STATES,
        "features": list(FEATURE_NAMES), "feature_parents_zero_based": list(FEATURE_PARENTS),
        "region_boundaries": list(REGION_BOUNDARIES), "density_bin_edges": list(DENSITY_EDGES),
        "aspect_bin_edges": list(ASPECT_EDGES), "hole_bin_edges": list(HOLE_EDGES),
        "foreground_threshold": FOREGROUND_THRESHOLD, "normalized_size": NORMALIZED_SIZE,
        "structure_learning": False, "test_set_parameter_selection": False,
        "test_set_seen_in_initial_work": True,
        "selection_sha256": sha256_file(source_dir / "results/optimization/selection.json"),
        "selected_config_sha256": sha256_file(source_dir / "selected_config.json"),
        "matches_selected_alpha": args.alpha == frozen["main_explicit_alpha"],
        "initial_report_sha256": sha256_file(source_dir / "history/initial/report.md"),
        "randomness": "Seed 20261019 for validation split only; final fitting and prediction are deterministic.",
        "invocation": sys.argv,
    }
    for name, content in (("metrics.json", results), ("run_metadata.json", metadata)):
        (result_dir / name).write_text(json.dumps(content, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (result_dir / "predictions.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["test_index", "true_label", "naive_bayes_prediction", "bayes_net_prediction"])
        writer.writerows(zip(range(len(y_test)), y_test, predictions["naive_bayes"], predictions["bayes_net"]))
    for key, result in results.items():
        np.savetxt(result_dir / f"confusion_{key}.csv", result["confusion_matrix"], fmt="%d", delimiter=",")
    lines = ["# Experiment summary", "", "Full MNIST train/test: 60,000 / 10,000. Alpha = " + str(args.alpha), "",
             "| Model | Accuracy | Macro precision | Macro recall | Macro F1 |",
             "|---|---:|---:|---:|---:|"]
    for result in results.values():
        lines.append(f"| {result['display_name']} | {result['accuracy']:.6f} | {result['macro_precision']:.6f} | "
                     f"{result['macro_recall']:.6f} | {result['macro_f1']:.6f} |")
    (result_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved experiment artifacts to {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
