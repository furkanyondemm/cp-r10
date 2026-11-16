"""One-condition integration; generated inputs are not benchmark results."""

import os
from pathlib import Path
import tempfile
import unittest
import numpy as np
from cpr10 import core
from cpr10.datasets import _write_npz
from cpr10.evaluation import _worker, CORRUPTIONS
from cpr10.io import write_json
from cpr10.qtc import DEFAULT_UPSTREAM_ROOT, SOURCE_SHA256
from cpr10.reporting import write_report
from cpr10.source import _initial

UPSTREAM = Path(os.environ.get("CPR10_QTC_ROOT", DEFAULT_UPSTREAM_ROOT))


@unittest.skipUnless(
    all((UPSTREAM / p).is_file() for p in SOURCE_SHA256),
    "Run python fetch_qtc.py to enable the integration test",
)
class PipelineTests(unittest.TestCase):
    def test_source_to_sets_reporting_and_integrity(self):
        import torch

        previous = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache, fits, output = root / "cache", root / "fits", root / "output"
            seed = "20260908"
            rng = np.random.default_rng(91)

            def predictions(n):
                p = rng.dirichlet(np.ones(10), n).astype(np.float32)
                y = np.argmax(p, axis=1).astype(np.int64)
                y[::4] = rng.integers(0, 10, len(y[::4]))
                return p, y

            pd, yd = predictions(1000)
            pf, _ = predictions(1500)
            source_path = cache / "source" / seed / "syn_0000.npz"
            _write_npz(source_path, p_design=pd, y_design=yd, p_feature=pf)
            job = (
                str(source_path),
                "c10_resnet18",
                str(root / "targets.json"),
                True,
                "gaussian_noise",
            )
            row = _initial(job)
            self.assertEqual(row, _initial(job))
            self.assertEqual(
                set(row["grid"]), {"cp_grid", "nll_grid", "cp_shared_grid"}
            )
            X = np.asarray([row["features"], row["features"]])

            def head(beta):
                return core.ridge_fit(X, np.array([beta, beta]))

            model = {
                "cp": {a: head(b) for a, b in row["cp"].items()},
                "nll": head(row["nll"]),
                "cp_grid": {a: head(b) for a, b in row["grid"]["cp_grid"].items()},
                "nll_grid": head(row["grid"]["nll_grid"]["shared"]),
                "shared_cp": head(row["grid"]["cp_shared_grid"]["shared"]),
                "offsets": {
                    "0.05": {"beta": 0.01, "alpha": 0.001},
                    "0.1": {"beta": 0.01, "alpha": 0.001},
                },
            }
            write_json(fits / seed / "models.json", model)
            calp, caly = predictions(2500)
            _write_npz(cache / "source" / seed / "calibration.npz", p=calp, y=caly)
            p, y = predictions(10000)
            target = cache / "targets" / "gaussian_noise_1.npz"
            _write_npz(target, p=p, y=y, ids=np.arange(10000))
            write_json(
                cache / "manifest.json",
                dict(
                    system="c10_resnet18",
                    main_source_seed=int(seed),
                    excluded_target_ids=np.linspace(0, 9999, 256, dtype=int).tolist(),
                    conditions=["gaussian_noise_1"],
                ),
            )
            task = (
                str(cache),
                str(fits),
                str(output),
                "c10_resnet18",
                "gaussian_noise_1",
                ("main", "grid", "equal_coverage"),
                str(UPSTREAM),
            )
            rows = _worker(task)
            self.assertEqual(rows, _worker(task))
            self.assertEqual(len(rows), 50 * 5 * 2 + 20 * 3 * 2 + 20 * 3 * 2)
            # Duplicate fixture rows only to exercise the complete report validator.
            fake_grid = []
            for corruption in CORRUPTIONS:
                for severity in range(1, 6):
                    fake_grid.extend(
                        {
                            **r,
                            "corruption": corruption,
                            "severity": severity,
                            "condition": f"{corruption}_{severity}",
                        }
                        for r in rows
                    )
            macro = write_report(output, fake_grid)
            self.assertTrue(macro)
            self.assertTrue((output / "equal_coverage.csv").is_file())
            self.assertIn("not a deployable method", (output / "README.md").read_text())
            with target.open("ab") as stream:
                stream.write(b"changed")
            with self.assertRaisesRegex(ValueError, "checksum"):
                _worker(task)
