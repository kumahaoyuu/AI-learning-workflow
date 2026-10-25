"""Independent arithmetic checks and protocol/edge-case checks. Standard unittest."""

import importlib.util
import inspect
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))
import bayes_net as bn
from bayes_net import BayesNetClassifier, extract_features
from naive_bayes import NaiveBayesClassifier, deskew_pixels, discretize_pixels
from main import classification_metrics, most_confused_pairs


def toy_images():
    X = np.zeros((4, 784), dtype=np.uint8)
    X[:, :2] = [[255, 255], [255, 0], [0, 255], [0, 0]]
    return X, np.array([0, 0, 1, 1])


class NaiveBayesArithmeticTests(unittest.TestCase):
    def test_likelihood_matches_explicit_categorical_formula(self):
        X, y = toy_images()
        model = NaiveBayesClassifier().fit(X, y)
        test = np.vstack([X, np.full((1, 784), 255, dtype=np.uint8)])
        expected = np.empty((len(test), 10))
        for row, image in enumerate(test):
            for digit in range(10):
                # All toy foreground lies on one row, so its vertical variance
                # is zero and deskew is exactly the identity transformation.
                members = (X[y == digit] // 32).astype(int)
                expected[row, digit] = np.log((len(members) + 1) / 14) + sum(
                    np.log((np.count_nonzero(members[:, j] == int(image[j]) // 32) + 1)
                           / (len(members) + 8)) for j in range(784)
                )
        np.testing.assert_allclose(model._joint_log_likelihood(test), expected, atol=1e-9)

    def test_known_prior_and_conditional_probability(self):
        X, y = toy_images()
        model = NaiveBayesClassifier().fit(X, y)
        self.assertAlmostEqual(np.exp(model.class_log_prior_[0]), 3 / 14)
        self.assertAlmostEqual(model.feature_prob_[0, 0, 7], 3 / 10)
        self.assertAlmostEqual(model.feature_prob_[0, 1, 7], 2 / 10)
        self.assertAlmostEqual(model.feature_prob_[9, 0, 0], 1 / 8)
        np.testing.assert_allclose(model.feature_prob_.sum(axis=-1), 1)

    def test_tiny_smoothing_does_not_create_negative_infinity(self):
        X, y = toy_images()
        model = NaiveBayesClassifier(1e-20).fit(X, y)
        self.assertTrue(np.isfinite(model._joint_log_likelihood(X)).all())
        self.assertTrue(np.isfinite(model.predict_proba(X)).all())

    def test_gray_bin_boundary(self):
        X, y = toy_images()
        model = NaiveBayesClassifier().fit(X, y)
        test = np.zeros((3, 784), dtype=np.uint8)
        test[:, 0] = [0, 31, 32]
        scores = model._joint_log_likelihood(test)
        np.testing.assert_array_equal(scores[0], scores[1])
        self.assertFalse(np.array_equal(scores[1], scores[2]))

    def test_eight_gray_bins_on_known_single_row_input(self):
        image = np.zeros((1, 784), dtype=np.uint8)
        image[0, :7] = [31, 32, 63, 64, 223, 224, 255]
        np.testing.assert_array_equal(discretize_pixels(image)[0, :7], [0, 1, 1, 2, 6, 7, 7])

    def test_deskew_reduces_known_synthetic_slant(self):
        image = np.zeros((28, 28), dtype=float)
        for row in range(4, 24):
            image[row, 7 + (row - 4) // 2] = 255
        def slope(a):
            yy, xx = np.indices(a.shape)
            total = a.sum()
            cy, cx = (a * yy).sum() / total, (a * xx).sum() / total
            return (a * (xx - cx) * (yy - cy)).sum() / (a * (yy - cy)**2).sum()
        corrected = deskew_pixels(image.reshape(1, 784)).reshape(28, 28)
        self.assertGreater(abs(slope(image)), 0.4)
        self.assertLess(abs(slope(corrected)), 1e-3)
        self.assertAlmostEqual(float(corrected.sum()), float(image.sum()), places=2)


class BayesNetworkArithmeticTests(unittest.TestCase):
    def setUp(self):
        self.F = np.zeros((4, 12), dtype=np.int64)
        self.F[:, :2] = [[0, 0], [0, 1], [1, 1], [1, 0]]
        self.y = np.array([0, 0, 0, 1])
        self.X = np.zeros((4, 784), dtype=np.uint8)
        with patch.object(bn, "extract_features", return_value=self.F):
            self.model = BayesNetClassifier().fit(self.X, self.y)

    def test_cpt_uses_parent_configuration_denominator(self):
        self.assertAlmostEqual(self.model.cpts_[1][0, 0, 1], 2 / 7)
        self.assertAlmostEqual(self.model.cpts_[1][0, 1, 1], 2 / 6)

    def test_all_parent_configurations_are_smoothed_and_normalized(self):
        for cpt in self.model.cpts_:
            self.assertTrue((cpt > 0).all())
            np.testing.assert_allclose(cpt.sum(axis=-1), 1)
            np.testing.assert_allclose(cpt[9], 1 / 5)

    def test_joint_matches_independent_counting_for_every_class(self):
        expected = np.zeros((4, 10))
        for row, features in enumerate(self.F):
            for digit in range(10):
                expected[row, digit] = np.log(((self.y == digit).sum() + 1) / 14)
                for i, parent in enumerate(bn.FEATURE_PARENTS):
                    mask = self.y == digit
                    if parent is not None:
                        mask = mask & (self.F[:, parent] == features[parent])
                    child_count = np.count_nonzero(mask & (self.F[:, i] == features[i]))
                    expected[row, digit] += np.log((child_count + 1) / (mask.sum() + 5))
        with patch.object(bn, "extract_features", return_value=self.F):
            actual = self.model._joint_log_likelihood(self.X)
        np.testing.assert_allclose(actual, expected)

    def test_graph_is_acyclic_with_at_most_two_total_parents(self):
        parents = bn.FEATURE_PARENTS
        self.assertTrue(6 <= len(parents) <= 12)
        self.assertGreaterEqual(sum(p is not None for p in parents), 2)
        for child, parent in enumerate(parents):
            self.assertLessEqual(1 + (parent is not None), 2)  # includes Y
            if parent is not None:
                self.assertTrue(0 <= parent < child)  # explicit topological order
        self.assertTrue(2 <= bn.N_STATES <= 5)


class FeatureTests(unittest.TestCase):
    def test_blank_and_full_images_have_known_semantics(self):
        X = np.vstack([np.zeros((1, 784)), np.full((1, 784), 255)])
        F = extract_features(X)
        np.testing.assert_array_equal(F[0], np.zeros(12))
        np.testing.assert_array_equal(F[1, :10], np.full(10, 4))
        np.testing.assert_array_equal(F[1, 10:], [0, 0])

    def test_aspect_ratio_and_hole_features(self):
        image = np.zeros((1, 28, 28), dtype=np.uint8)
        image[:, 4:24, 8:20] = 255  # width/height = 0.6; no enclosed background
        F = extract_features(image.reshape(1, 784))
        self.assertEqual(F[0, 9], 2)  # boundary goes to higher bin
        np.testing.assert_array_equal(F[0, 10:], [0, 0])
        image[:] = 0
        image[:, 0, 0] = 255
        np.testing.assert_array_equal(extract_features(image.reshape(1, 784))[0, 10:], [0, 0])

    def test_enclosed_background_detects_ring_and_opening(self):
        ring = np.zeros((1, 28, 28), dtype=bool)
        ring[:, 5:12, 5] = True
        ring[:, 5:12, 11] = True
        ring[:, 5, 5:12] = True
        ring[:, 11, 5:12] = True
        self.assertEqual(int(bn._enclosed_background(ring).sum()), 25)
        features = extract_features(ring.reshape(1, 784).astype(np.uint8) * 255)
        np.testing.assert_array_equal(features[0, 10:], [4, 4])
        ring[:, 5, 8] = False  # opening joins the interior to boundary background
        self.assertEqual(int(bn._enclosed_background(ring).sum()), 0)

    def test_box_normalization_is_translation_invariant(self):
        images = np.zeros((2, 28, 28), dtype=np.uint8)
        pattern = np.array([[255, 0, 255], [255, 255, 255], [0, 0, 255]], dtype=np.uint8)
        images[0, 3:6, 4:7] = pattern
        images[1, 14:17, 19:22] = pattern
        features = extract_features(images.reshape(2, 784))
        np.testing.assert_array_equal(features[0], features[1])

    def test_features_are_batch_independent_integer_states(self):
        X, _ = toy_images()
        F = extract_features(X)
        self.assertEqual(F.shape, (4, 12))
        self.assertTrue(np.issubdtype(F.dtype, np.integer))
        self.assertTrue(((F >= 0) & (F < 5)).all())
        np.testing.assert_array_equal(F, np.vstack([extract_features(row[None, :]) for row in X]))


class ProtocolTests(unittest.TestCase):
    def test_exact_constructor_signature_and_independent_file_loading(self):
        for filename, classname in (("naive_bayes.py", "NaiveBayesClassifier"), ("bayes_net.py", "BayesNetClassifier")):
            spec = importlib.util.spec_from_file_location("isolated_" + classname, SOURCE / filename)
            module = importlib.util.module_from_spec(spec)
            with patch.object(sys, "path", []):
                spec.loader.exec_module(module)
            cls = getattr(module, classname)
            self.assertEqual(list(inspect.signature(cls).parameters), ["alpha"])
            self.assertEqual(inspect.signature(cls).parameters["alpha"].default, 1.0)
            for method, params in (("fit", ["self", "X", "y"]), ("predict", ["self", "X"])):
                self.assertEqual(list(inspect.signature(getattr(cls, method)).parameters), params)

    def test_fit_returns_self_prediction_shape_and_probabilities(self):
        X, y = toy_images()
        for cls in (NaiveBayesClassifier, BayesNetClassifier):
            model = cls()
            self.assertIs(model.fit(X, y), model)
            pred, proba = model.predict(X), model.predict_proba(X)
            self.assertEqual(pred.shape, (4,))
            self.assertTrue(np.issubdtype(pred.dtype, np.integer))
            self.assertEqual(proba.shape, (4, 10))
            self.assertTrue(np.isfinite(proba).all())
            np.testing.assert_allclose(proba.sum(axis=1), 1)
            np.testing.assert_array_equal(pred, proba.argmax(axis=1))
            self.assertEqual(model.predict(X[:0]).shape, (0,))
            self.assertEqual(model.predict_proba(X[:0]).shape, (0, 10))

    def test_training_and_prediction_do_not_modify_inputs_or_parameters(self):
        X, y = toy_images()
        original = X.copy()
        for cls in (NaiveBayesClassifier, BayesNetClassifier):
            model = cls().fit(X, y)
            prior = model.class_log_prior_.copy()
            model.predict(X)
            np.testing.assert_array_equal(model.class_log_prior_, prior)
            np.testing.assert_array_equal(X, original)
            np.testing.assert_array_equal(y, [0, 0, 1, 1])

    def test_invalid_inputs_and_unfitted_prediction_fail_clearly(self):
        X, y = toy_images()
        for cls in (NaiveBayesClassifier, BayesNetClassifier):
            with self.assertRaises(RuntimeError):
                cls().predict(X)
            for alpha in (0, -1, np.nan, np.inf):
                with self.assertRaises(ValueError):
                    cls(alpha)
            for bad_X in (np.zeros((2, 28, 28)), np.zeros((2, 783)), np.full((2, 784), np.nan),
                          np.full((2, 784), 256), np.full((2, 784), -1)):
                with self.assertRaises(ValueError):
                    cls().fit(bad_X, [0, 1])
            for bad_y in ([0, 1], [0, 1, 2, 10], [0, 1, 2, 1.5], [0, 1, 2, np.nan]):
                with self.assertRaises(ValueError):
                    cls().fit(X, bad_y)
            with self.assertRaises(ValueError):
                cls().fit(X[:0], y[:0])


class MetricsTests(unittest.TestCase):
    def test_metrics_match_hand_computed_example_including_absent_classes(self):
        result = classification_metrics([0, 0, 1, 1], [0, 1, 1, 1])
        self.assertEqual(result["accuracy"], 0.75)
        self.assertAlmostEqual(result["macro_precision"], (1 + 2 / 3) / 10)
        self.assertAlmostEqual(result["macro_recall"], (0.5 + 1) / 10)
        self.assertAlmostEqual(result["macro_f1"], (2 / 3 + 4 / 5) / 10)
        np.testing.assert_array_equal(np.asarray(result["confusion_matrix"])[:2, :2], [[1, 1], [0, 2]])

    def test_confused_pairs_aggregate_both_directions(self):
        cm = np.zeros((10, 10), dtype=int)
        cm[2, 3], cm[3, 2], cm[0, 1] = 4, 5, 8
        first = most_confused_pairs(cm, 1)[0]
        self.assertEqual(first, {"digits": [2, 3], "total": 9, "a_to_b": 4, "b_to_a": 5})


if __name__ == "__main__":
    unittest.main(verbosity=2)
