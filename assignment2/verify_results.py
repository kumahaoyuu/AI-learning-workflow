"""Verify saved experiments without MNIST raw data; run the numerical unit tests.

Run: python verify_results.py
Writes results/audit.json and results/test_results.txt. CSV metrics below use an
independent pure-Python calculation, not main.classification_metrics.
"""

import ast
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import unittest

from PIL import Image


INITIAL_REPORT_SHA256 = "76e44351555726ef04c2f17b4ae8dda5d1b9982f22fae41560982fef56e8387a"
PREVIOUS_REPORT_SHA256 = "67220e10e0928ab7bc627fc1dcbf40cdee38e247411b506f91675af0bdfbddc4"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path(__file__).resolve().parent
    result_dir = root / "results"
    metadata = json.loads((result_dir / "run_metadata.json").read_text(encoding="utf-8"))
    metrics = json.loads((result_dir / "metrics.json").read_text(encoding="utf-8"))
    frozen = json.loads((root / "selected_config.json").read_text(encoding="utf-8"))
    selection = json.loads((result_dir / "optimization/selection.json").read_text(encoding="utf-8"))
    initial = root / "history/initial"
    assert sha256(initial / "report.md") == INITIAL_REPORT_SHA256
    original_manifest = json.loads((initial / "results/delivery_sha256.json").read_text(encoding="utf-8"))
    for name, expected in original_manifest.items():
        assert sha256(initial / name) == expected, ("initial snapshot", name)
    assert (root / "mnist_loader.py").read_bytes() == (initial / "mnist_loader.py").read_bytes()
    previous = root / "history/previous"
    assert sha256(previous / "report.md") == PREVIOUS_REPORT_SHA256
    previous_manifest = json.loads((previous / "results/delivery_sha256.json").read_text(encoding="utf-8"))
    for name, expected in previous_manifest.items():
        assert sha256(previous / name) == expected, ("previous version", name)
    for name, expected in metadata["source_sha256"].items():
        assert sha256(root / name) == expected, ("experiment source", name)
    for name, expected in frozen["core_source_sha256"].items():
        assert sha256(root / name) == expected, ("frozen model", name)
    assert sha256(root / "selected_config.json") == metadata["selected_config_sha256"]
    assert sha256(result_dir / "optimization/selection.json") == frozen["selection_sha256"]
    for name, expected in selection["source_sha256"].items():
        assert sha256(root / name) == expected, ("validation source", name)
    assert selection["test_files_read"] is False
    assert selection["automatic_structure_learning"] is False
    assert selection["candidate_count"] == 208
    assert metadata["matches_selected_alpha"]
    assert metadata["initial_report_sha256"] == INITIAL_REPORT_SHA256
    assert metadata["previous_report_sha256"] == PREVIOUS_REPORT_SHA256

    with (result_dir / "predictions.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 10000
    assert [int(row["test_index"]) for row in rows] == list(range(10000))
    reconstructed = {}
    for key in ("naive_bayes", "bayes_net"):
        cm = [[0] * 10 for _ in range(10)]
        for row in rows:
            cm[int(row["true_label"])][int(row[key + "_prediction"])] += 1
        assert cm == metrics[key]["confusion_matrix"]
        with (result_dir / f"confusion_{key}.csv").open(encoding="utf-8", newline="") as stream:
            stored = [[int(value) for value in row] for row in csv.reader(stream)]
        assert stored == cm
        precision, recall, f1 = [], [], []
        for digit in range(10):
            tp = cm[digit][digit]
            actual = sum(cm[digit])
            predicted = sum(row[digit] for row in cm)
            precision.append(tp / predicted if predicted else 0)
            recall.append(tp / actual if actual else 0)
            f1.append(2 * tp / (actual + predicted) if actual + predicted else 0)
        reconstructed[key] = {
            "accuracy": sum(cm[i][i] for i in range(10)) / 10000,
            "macro_precision": sum(precision) / 10,
            "macro_recall": sum(recall) / 10,
            "macro_f1": sum(f1) / 10,
        }
        for metric, value in reconstructed[key].items():
            assert math.isclose(value, metrics[key][metric], abs_tol=1e-12)

    core_imports = {}
    for name in ("naive_bayes.py", "bayes_net.py"):
        tree = ast.parse((root / name).read_text(encoding="utf-8"))
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module)
        assert imports == ["numpy"], (name, imports)
        core_imports[name] = imports
    figure_checks = {}
    for path in sorted((root / "figures").glob("*.png")):
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            dpi = im.info.get("dpi")
            assert dpi and min(dpi) >= 299.5
            figure_checks[path.name] = {"pixels": list(im.size), "dpi": list(dpi)}
        assert path.with_suffix(".svg").is_file()
    assert len(figure_checks) == 5
    assert not list(root.rglob("*.gz")), "Raw MNIST data must not be part of the delivery"

    suite = unittest.defaultTestLoader.discover(str(root / "tests"))
    buffer = io.StringIO()
    test_run = unittest.TextTestRunner(stream=buffer, verbosity=2).run(suite)
    (result_dir / "test_results.txt").write_text(buffer.getvalue(), encoding="utf-8")
    print(buffer.getvalue())
    assert test_run.wasSuccessful()
    report = (root / "report.md").read_text(encoding="utf-8")
    for scores in reconstructed.values():
        for value in scores.values():
            assert f"{value:.6f}" in report
    audit = {
        "unit_tests_run": test_run.testsRun, "unit_tests_passed": test_run.wasSuccessful(),
        "prediction_rows": len(rows), "independently_recomputed_metrics": reconstructed,
        "confusion_csvs_match_predictions": True, "source_hashes_match_experiment": True,
        "frozen_core_sources_match": True, "initial_snapshot_hashes_match": True,
        "initial_report_sha256": INITIAL_REPORT_SHA256, "course_loader_byte_identical": True,
        "previous_report_sha256": PREVIOUS_REPORT_SHA256, "previous_snapshot_hashes_match": True,
        "core_model_imports": core_imports, "training_only_selection_candidates": 208,
        "raw_mnist_data_in_delivery": False, "figures": figure_checks,
    }
    (result_dir / "audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print("FINAL_AUDIT_PASSED: both earlier versions, frozen models, predictions, metrics, sources and 23 tests")


if __name__ == "__main__":
    main()
