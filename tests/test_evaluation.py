"""Protocol invariants, label isolation, tie diagnostics and paired reporting."""

import hashlib
import unittest

import numpy as np

from cpr10 import core
from cpr10.evaluation import (
    CORRUPTIONS,
    disjoint_pools,
    empirical_threshold,
    main_draws,
    requests,
    score_requests,
)
from cpr10.numerics import metrics_from_log_threshold
from cpr10.reporting import CELL, _bootstrap, _mean, paired_intervals, validate_rows


class EvaluationTests(unittest.TestCase):
    def test_saved_paper_split_fingerprints(self):
        # Checked against the archived CIFAR Test4 and ImageNet Test4 split files.
        cases = [
            (
                9744,
                "c10_resnet18",
                "64959bf67491f08fd9aced584e0e2490bf3dbfa122f3106fe29d9c271da58ea5",
            ),
            (
                10000,
                "c100_vit_b_16",
                "aa943bfc7a8603e5cb4b49746a3e38cb619d2886e1d9d43ed5c6140fa5023fb8",
            ),
            (
                45000,
                "imagenet_resnet50",
                "cf6303c35532f905f25ae7c6a1ce50e6a25bb635358f3c27b5bacd888326071f",
            ),
        ]
        for n, system, expected in cases:
            ev, full, anchors = disjoint_pools(n, system)
            digest = hashlib.sha256(
                np.concatenate([ev, full, *anchors]).astype("<i8").tobytes()
            ).hexdigest()
            self.assertEqual(digest, expected)
            self.assertEqual(np.intersect1d(ev, full).size, 0)
            self.assertTrue(
                all(len(a) == 1500 and np.isin(a, full).all() for a in anchors)
            )

    def test_main_draw_order_matches_published_generators(self):
        # ImageNet preserves random order; CIFAR sorts each draw after sampling.
        rng = np.random.default_rng(20260908)
        expected = [np.sort(rng.choice(9744, 1500, replace=False)) for _ in range(50)]
        for actual, reference in zip(main_draws(9744, "c10_resnet18"), expected):
            np.testing.assert_array_equal(actual, reference)
        actual = main_draws(45000, "imagenet_resnet50")[49]
        expected = np.random.default_rng(np.random.SeedSequence([20260912, 49])).choice(
            45000, 1500, replace=False
        )
        np.testing.assert_array_equal(actual, expected)
        self.assertFalse(np.array_equal(actual, np.sort(actual)))

    def test_extra_sources_are_excluded_before_sampling(self):
        eligible = np.arange(100, 4100)
        ev, full, anchors = disjoint_pools(
            45000, "imagenet_resnet50", source_repeats=True, eligible=eligible
        )
        np.testing.assert_array_equal(np.sort(np.concatenate([ev, full])), eligible)
        self.assertTrue(all(np.isin(a, eligible).all() for a in anchors))

    def test_labels_do_not_change_decisions(self):
        p = np.tile([0.95, 0.05], (100, 1))
        head = core.ridge_fit(np.zeros((2, 10)), np.ones(2))
        models = {"cp": {"0.05": head, "0.1": head}, "nll": head}
        a, b = requests(p, models, ("CP", "NLL")), requests(p, models, ("CP", "NLL"))
        score_requests(a, p, p, np.zeros(100, dtype=int))
        score_requests(b, p, p, np.ones(100, dtype=int))
        for left, right in zip(a, b):
            self.assertEqual(left["beta"], right["beta"])
            self.assertEqual(left["log_threshold"], right["log_threshold"])
            self.assertEqual(left["mean_set_size"], right["mean_set_size"])
        self.assertGreater(a[-1]["coverage"], b[-1]["coverage"])

    def test_equal_coverage_ties_are_not_hidden(self):
        logs = np.log(np.array([[0.8, 0.2], [0.8, 0.2], [0.7, 0.3]]))
        labels = np.zeros(3, dtype=int)
        threshold = empirical_threshold(logs, labels, 1)
        self.assertEqual(threshold, logs[0, 0])
        result = metrics_from_log_threshold(logs, labels, threshold)
        self.assertAlmostEqual(result["coverage"], 2 / 3)
        self.assertEqual(
            metrics_from_log_threshold(
                logs, labels, empirical_threshold(logs, labels, 0)
            )["mean_set_size"],
            0,
        )


class ReportingTests(unittest.TestCase):
    def test_absolute_error_is_averaged_after_each_draw(self):
        rows = []
        for coverage in (0.8, 1.0):
            error = coverage - 0.9
            rows.append(
                dict(
                    system="c10_resnet18",
                    dataset="corruption",
                    regime="main",
                    arm="1500",
                    source_seed="20260908",
                    alpha=0.1,
                    method="CP",
                    corruption="fog",
                    severity=1,
                    coverage=coverage,
                    abs_coverage_error=abs(error),
                    undercoverage=max(-error, 0),
                    overcoverage=max(error, 0),
                    mean_set_size=2.0,
                )
            )
        (result,) = _mean(rows, CELL)
        self.assertAlmostEqual(result["coverage"], 0.9)
        self.assertAlmostEqual(result["abs_coverage_error"], 0.1)
        self.assertAlmostEqual(result["undercoverage"] + result["overcoverage"], 0.1)

    def test_paired_cluster_bootstrap_and_units(self):
        rows = []
        stratum = dict(
            system="imagenet_vit_b_16",
            dataset="corruption",
            regime="grid",
            arm="1500",
            source_seed="20260911",
            alpha=0.05,
        )
        for i, corruption in enumerate(CORRUPTIONS):
            for method, offset in (("CP-grid", 0.0), ("NLL-grid", 0.02)):
                rows.append(
                    dict(
                        **stratum,
                        method=method,
                        corruption=corruption,
                        coverage=0.95,
                        abs_coverage_error=0.01 + offset + 0.001 * i,
                        undercoverage=0.005,
                        overcoverage=0.005 + offset + 0.001 * i,
                        mean_set_size=10.0 + 100 * offset
                    )
                )
        intervals = paired_intervals(rows)
        self.assertEqual(len(intervals), 4)
        for row in intervals:
            self.assertEqual(row["bootstrap_resamples"], 2000)
        result = next(r for r in intervals if r["metric"] == "abs_coverage_error")
        self.assertAlmostEqual(result["estimate"], -2.0)
        self.assertAlmostEqual(result["ci_low"], -2.0)
        self.assertAlmostEqual(result["ci_high"], -2.0)
        self.assertEqual(result["units"], "pp")
        draws, _ = _bootstrap(tuple(stratum.values()), "CP-grid", "NLL-grid")
        self.assertEqual(draws.shape, (2000, 15))
        self.assertTrue(np.all((0 <= draws) & (draws < 15)))

    def test_natural_and_cifar_grid_have_no_invented_intervals(self):
        for system, dataset, regime in (
            ("imagenet_resnet50", "imagenetv2", "clean"),
            ("c100_vit_b_16", "corruption", "grid"),
        ):
            draws, _ = _bootstrap(
                (system, dataset, regime, "1500", "20260911", 0.05), "CP", "NLL"
            )
            self.assertIsNone(draws)

    def test_complete_grid_requires_all_levels_methods_conditions_and_draws(self):
        rows = []
        for alpha in (0.05, 0.1):
            for method in ("CP-grid", "NLL-grid", "Shared-CP"):
                for corruption in CORRUPTIONS:
                    for severity in range(1, 6):
                        for repeat in range(20):
                            rows.append(
                                dict(
                                    system="c10_resnet18",
                                    dataset="corruption",
                                    regime="grid",
                                    arm="1500",
                                    source_seed="20260908",
                                    alpha=alpha,
                                    method=method,
                                    corruption=corruption,
                                    severity=severity,
                                    repeat=repeat,
                                    coverage=1 - alpha,
                                    abs_coverage_error=0.0,
                                    undercoverage=0.0,
                                    overcoverage=0.0,
                                    mean_set_size=2.0,
                                )
                            )
        validate_rows(rows)
        for incomplete in (
            rows[:-1],
            [r for r in rows if r["corruption"] != "fog"],
            [r for r in rows if r["alpha"] == 0.05],
            [r for r in rows if r["method"] != "Shared-CP"],
        ):
            with self.assertRaises(ValueError):
                validate_rows(incomplete)


if __name__ == "__main__":
    unittest.main()
