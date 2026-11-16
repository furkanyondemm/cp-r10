"""Independent checks of the set rule, source losses, features and ridge fit."""

import math
import unittest

import numpy as np

from cpr10 import controls, core, oracle
from cpr10.numerics import (
    evaluate_plugin,
    metrics_from_log_threshold,
    plugin_log_thresholds,
    temperature_log_probs,
)


class NumericalTests(unittest.TestCase):
    def test_zero_beta_includes_every_tied_class(self):
        p = np.array([[0.9, 0.1], [0.1, 0.9], [0.7, 0.3]])
        for alpha in (0.05, 0.1, 0.5):
            result = evaluate_plugin(p, np.array([0, 1, 1]), 0.0, alpha)
            self.assertEqual(result["coverage"], 1.0)
            self.assertEqual(result["mean_set_size"], 2.0)
            self.assertAlmostEqual(result["abs_coverage_error"], alpha)

    def test_tiny_positive_deficits_are_not_rounded_away(self):
        p = np.array([[1.0, 1e-12], [1.0, 1e-11], [0.6, 0.4]])
        logp = temperature_log_probs(p, 20.0)
        (threshold,) = plugin_log_thresholds(logp, [0.5])
        # Required mass is 2. The two largest weights are strictly less than 2.
        self.assertEqual(threshold, logp[2, 0])
        metrics = metrics_from_log_threshold(logp, np.zeros(3, dtype=int), threshold)
        self.assertEqual(metrics["coverage"], 1.0)
        self.assertEqual(metrics["mean_set_size"], 1.0)

    def test_finite_sample_rank_is_capped_at_total_mass(self):
        p = np.array([[0.8, 0.2], [0.6, 0.4]])
        logp = temperature_log_probs(p, 1.0)
        (threshold,) = plugin_log_thresholds(logp, [0.05])
        self.assertEqual(threshold, logp.min())
        self.assertTrue((logp >= threshold).all())

    def test_input_floor_does_not_cap_post_temperature_loss(self):
        p = np.array([[1.0, 0.0]])
        logp = temperature_log_probs(p, 20.0)
        self.assertAlmostEqual(-logp[0, 1], 240 * math.log(10), places=10)
        self.assertGreater(-logp[0, 1], -math.log(1e-12))

    def test_reported_nll_loss_conventions_remain_distinct(self):
        p = np.tile([0.8, 0.2], (100, 1))
        p[0] = [1.0 - 1e-12, 1e-12]
        y = np.zeros(100, dtype=int)
        y[0] = 1
        # Solve the untruncated binary NLL derivative independently by bisection.
        ordinary_log_odds = math.log(4)
        rare_log_odds = math.log((1 - 1e-12) / 1e-12)
        low, high = 0.0, 20.0
        for _ in range(100):
            beta = (low + high) / 2
            derivative = -99 * ordinary_log_odds / (
                1 + math.exp(ordinary_log_odds * beta)
            ) + rare_log_odds / (1 + math.exp(-rare_log_odds * beta))
            if derivative > 0:
                high = beta
            else:
                low = beta
        optimum = (low + high) / 2
        for system in ("c10_resnet18", "imagenet_resnet50"):
            self.assertAlmostEqual(
                core.original_nll_target(p, y, system), optimum, places=6
            )
        # The inherited CIFAR-100 loss clips at 1e-15 after temperature scaling.
        self.assertEqual(core.original_nll_target(p, y, "c100_vit_b16"), 20.0)

    def test_nll_endpoint_for_incorrect_classifier(self):
        p = np.tile([0.8, 0.2], (30, 1))
        y = np.ones(30, dtype=int)
        for system in ("c10_resnet18", "c100_vit_b16", "imagenet_resnet50"):
            self.assertEqual(core.original_nll_target(p, y, system), 0.0)


class FeatureAndRegressionTests(unittest.TestCase):
    def test_ten_features_against_two_image_calculation(self):
        p = np.array([[0.8, 0.2], [0.6, 0.4]])
        first_h = -0.8 * math.log(0.8) - 0.2 * math.log(0.2)
        second_h = -0.6 * math.log(0.6) - 0.4 * math.log(0.4)
        first_ratio, second_ratio = math.log(4), math.log(1.5)
        expected = [
            0.7,
            (first_h + second_h) / 2,
            0.4,
            0.62,
            0.1 * first_h + 0.9 * second_h,
            0.24,
            -1e-12 * math.log(1e-12),
            (first_ratio + second_ratio) / 2,
            0.1 * first_ratio + 0.9 * second_ratio,
            (first_ratio + second_ratio) / 4,
        ]
        features = core.r10_features(p)
        self.assertEqual(list(features), core.FEATURES)
        np.testing.assert_allclose(
            list(features.values()), expected, rtol=1e-14, atol=1e-15
        )

    def test_feature_floor_does_not_renormalize(self):
        features = core.r10_features(np.array([[1.0, 0.0], [1.0, 0.0]]))
        self.assertEqual(features["msp_mean"], 1.0)
        self.assertEqual(features["margin_mean"], 1.0 - 1e-12)
        self.assertEqual(features["entropy_mean"], -1e-12 * math.log(1e-12))

    def test_ridge_penalty_and_unpenalized_intercept(self):
        X = np.ones((3, 10))
        X[:, 0] = [-1.0, 0.0, 1.0]
        y = 2.0 + 3.0 * X[:, 0]
        model = core.ridge_fit(X, y)
        self.assertEqual(model["ridge_intercept"], 2.0)
        np.testing.assert_array_equal(model["scaler_scale"][1:], np.ones(9))
        np.testing.assert_array_equal(model["ridge_coef"][1:], np.zeros(9))
        # Sum z^2 = 3; coefficient shrinkage is 3/(3 + .1).
        expected = np.clip(2.0 + 9.0 / 3.1 * X[:, 0], 0.0, 20.0)
        np.testing.assert_allclose(core.ridge_predict(model, X), expected, atol=1e-14)
        extreme = np.ones((2, 10))
        extreme[:, 0] = [-100.0, 100.0]
        np.testing.assert_array_equal(core.ridge_predict(model, extreme), [0.0, 20.0])


class SourceSelectionTests(unittest.TestCase):
    def test_original_and_common_grid_ties_have_different_declared_rules(self):
        rows = []
        for beta, size in ((1.0, 5.0), (2.0, 1.0)):
            for alpha in core.ALPHAS:
                rows.append(
                    dict(
                        beta=beta,
                        alpha=alpha,
                        nll=1.0,
                        abs_coverage_error=0.01,
                        mean_set_size=size,
                    )
                )
        self.assertEqual(oracle.best_alpha(rows, 0.05)["beta"], 2.0)
        targets, _ = controls.select_targets(rows)
        self.assertEqual(targets["cp_grid"]["0.05"], 1.0)
        self.assertEqual(targets["nll_grid"]["shared"], 1.0)
        self.assertEqual(targets["cp_shared_grid"]["shared"], 1.0)

    def test_common_grid_uses_loss_tolerance_before_smallest_beta(self):
        rows = []
        for beta, error in ((1.0, 0.01 + 0.5e-12), (2.0, 0.01)):
            for alpha in core.ALPHAS:
                rows.append(
                    dict(
                        beta=beta,
                        alpha=alpha,
                        nll=error,
                        abs_coverage_error=error,
                        mean_set_size=3.0 - beta,
                    )
                )
        targets, _ = controls.select_targets(rows)
        self.assertEqual(targets["cp_grid"]["0.05"], 1.0)
        self.assertEqual(targets["nll_grid"]["shared"], 1.0)

    def test_shared_target_uses_maximum_error_then_mean_error(self):
        rows = []
        for beta, errors in (
            (1.0, [0.03, 0.03]),
            (2.0, [0.03, 0.01]),
            (3.0, [0.04, 0.0]),
        ):
            for alpha, error in zip(core.ALPHAS, errors):
                rows.append(
                    dict(
                        beta=beta,
                        alpha=alpha,
                        nll=1.0,
                        abs_coverage_error=error,
                        mean_set_size=1.0,
                    )
                )
        targets, _ = controls.select_targets(rows)
        self.assertEqual(targets["cp_shared_grid"]["shared"], 2.0)

    def test_grid_and_scalar_search_boundaries(self):
        grid = controls.coarse_grid()
        self.assertEqual(len(grid), 93)
        self.assertEqual((grid[0], grid[-1]), (0.0, 20.0))
        np.testing.assert_array_equal(grid, oracle.coarse_grid())
        self.assertEqual(controls.d_refine(0.5)[-1], 0.51)
        self.assertEqual(controls.choose([-0.1, 0.2], [0.01, 0.01], [2.0, 1.0]), 1)

    def test_cifar_final_offset_keeps_contiguous_column_reduction(self):
        rng = np.random.default_rng(21)
        first = rng.integers(0, 1001, 320) / 1000
        second = first[rng.permutation(len(first))]
        tables = [
            {
                0.0: {"coverage": a, "mean_set_size": 1.0},
                0.1: {"coverage": b, "mean_set_size": 1.0},
            }
            for a, b in zip(first, second)
        ]
        result = controls._aggregate(tables, [0.0, 0.1], 0.05, columnwise=True)
        # Each candidate has the same error multiset; the final tie uses |d|.
        expected = np.array(
            [np.mean(np.abs(first - 0.95)), np.mean(np.abs(second - 0.95))]
        )
        np.testing.assert_array_equal(result["source_mean_A"], expected)
        self.assertEqual(expected[0], expected[1])
        self.assertEqual(
            controls.choose(
                [0.0, 0.1], result["source_mean_A"], result["source_mean_S"]
            ),
            0,
        )


if __name__ == "__main__":
    unittest.main()
