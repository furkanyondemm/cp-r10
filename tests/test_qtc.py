"""QTC numerical and boundary tests; run after fetching the pinned sources."""

import inspect
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from cpr10.qtc import DEFAULT_UPSTREAM_ROOT, QTCSource, SOURCE_SHA256, verify_upstream
import fetch_qtc


UPSTREAM = Path(os.environ.get("CPR10_QTC_ROOT", DEFAULT_UPSTREAM_ROOT))


@unittest.skipUnless(
    all((UPSTREAM / path).is_file() for path in SOURCE_SHA256),
    "Run python fetch_qtc.py to enable QTC tests",
)
class QTCTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def source(self, boundary="infinity"):
        p = np.tile(np.array([0.6, 0.4], dtype=np.float32), (20, 1))
        return QTCSource(
            p, np.zeros(len(p), dtype=int), upstream_root=UPSTREAM, boundary=boundary
        )

    def test_linear_quantile_and_inclusive_sets(self):
        rng = np.random.default_rng(17)
        p = rng.dirichlet(np.ones(4), 103)
        labels = rng.integers(0, 4, len(p))
        source = QTCSource(p, labels, upstream_root=UPSTREAM)
        result = source._calibrate(0.2)
        recovered = torch.softmax(
            torch.log(torch.tensor(p, dtype=torch.float32)), 1
        ).numpy()
        scores = np.array([1.0 - row[label] for row, label in zip(recovered, labels)])
        self.assertEqual(result["q"], np.quantile(scores, 0.8 * (1 + 1 / len(p))))
        self.assertFalse(result["boundary_applied"])
        rows = source.evaluate(p)
        for row in rows:
            self.assertEqual(row["gamma"], min(row["gamma_T"], row["gamma_S"]))
            np.testing.assert_array_equal(row["mask"], recovered >= 1.0 - row["q"])

    def test_upper_boundary_and_cap(self):
        regular, capped = self.source(), self.source("cap")
        for gamma in (0.0, 0.5 / 21):
            a, b = regular._calibrate(gamma), capped._calibrate(gamma)
            self.assertTrue(a["boundary_applied"])
            self.assertTrue(b["boundary_applied"])
            self.assertEqual(a["q"], np.inf)
            self.assertEqual(b["q"], capped._maximum_source_score())
            self.assertEqual(a["gamma"], b["gamma"])
        target = np.tile(np.array([0.5, 0.5], dtype=np.float32), (8, 1))
        original = regular.evaluate(target)
        cap = capped.evaluate(target)
        for a, b in zip(original, cap):
            self.assertTrue(a["mask"].all())
            self.assertFalse(b["mask"].any())
            self.assertEqual(a["gamma"], b["gamma"])

    def test_zero_probabilities_are_not_floored(self):
        p = np.tile(np.array([1.0, 0.0], dtype=np.float32), (20, 1))
        source = QTCSource(p, np.zeros(len(p), dtype=int), upstream_root=UPSTREAM)
        self.assertTrue(torch.isneginf(source._source_logits[:, 1]).all())
        for row in source.evaluate(p):
            self.assertTrue(row["mask"].all())

    def test_other_errors_are_not_hidden(self):
        source = self.source()
        with patch.object(
            source._model, "_get_Q_hat", side_effect=ValueError("unrelated error")
        ):
            with self.assertRaisesRegex(ValueError, "unrelated error"):
                source._calibrate(0.0)
        for gamma in (-0.1, 1.1, np.nan, np.inf):
            with self.assertRaises(ValueError):
                source._calibrate(gamma)

    def test_api_excludes_target_labels(self):
        self.assertEqual(
            list(inspect.signature(self.source().evaluate).parameters),
            ["p_target", "alphas"],
        )

    def test_changed_upstream_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / next(iter(SOURCE_SHA256))
            path.parent.mkdir(parents=True)
            path.write_text("# modified\n")
            with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
                verify_upstream(root)

    def test_fetch_authenticates_every_file_and_resumes(self):
        def local_response(url, timeout):
            self.assertEqual(timeout, 60)
            relative = next(path for path in SOURCE_SHA256 if url.endswith("/" + path))
            return io.BytesIO((UPSTREAM / relative).read_bytes())

        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            with patch.object(
                fetch_qtc.urllib.request, "urlopen", side_effect=local_response
            ) as download:
                fetch_qtc.fetch(destination)
                self.assertEqual(download.call_count, 3)
                fetch_qtc.fetch(destination)
                self.assertEqual(download.call_count, 3)
            self.assertEqual(
                verify_upstream(destination)["source_sha256"], SOURCE_SHA256
            )

    def test_fetch_rejects_corrupted_download(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(
                fetch_qtc.urllib.request,
                "urlopen",
                return_value=io.BytesIO(b"bad source"),
            ):
                with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
                    fetch_qtc.fetch(Path(directory))
            self.assertFalse(any(Path(directory).rglob("*.py")))


if __name__ == "__main__":
    unittest.main()
