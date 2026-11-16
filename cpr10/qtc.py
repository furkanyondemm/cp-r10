"""Paper QTC: pinned author routines on reconstructed float32 log probabilities.

The only extension is the declared upper-quantile boundary: infinity in the
main comparison, or the largest source score in the cap sensitivity check.
The caller controls input normalization. No target labels enter this module.
"""

from __future__ import annotations

import ast
from collections import Counter
import hashlib
import math
from pathlib import Path
from typing import Any, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

UPSTREAM_REPOSITORY = "https://github.com/MLI-lab/recalibrating_conformal_prediction"
UPSTREAM_COMMIT = "25b4b821539cce5dc6d0082da4f7db41858a4f41"
DEFAULT_UPSTREAM_ROOT = Path(__file__).resolve().parents[1] / "third_party/qtc"
SOURCE_SHA256 = {
    "utils/pytorch_utils/inference.py": "37a6defb841c4b480b30fc5f0bf88035535f9d846e48179edded7326b1b0c3b4",
    "utils/pytorch_utils/prediction.py": "1ef9d167afbde5c149072a431e475f7f9f1fa13624a8c98a4a2cbde68b3bfcdb",
    "utils/conformal_utils/conformal.py": "3349c85d0e6c6fb36d0a6464487e1ea152e2fd570f41eae05bbea2f72f6d1e82",
}
_SELECTIONS = {
    "utils/pytorch_utils/inference.py": ["get_sorted_softmax"],
    "utils/pytorch_utils/prediction.py": ["QuantileConfidenceThreshold"],
    "utils/conformal_utils/conformal.py": [
        "ConformalModel",
        "ThresholdedConformalModel",
    ],
}
_QUANTILE_DOMAIN_MESSAGE = "Quantiles must be in the range [0, 1]"


def verify_upstream(upstream_root=None):
    """Authenticate executed files by content, including an explicitly given checkout."""
    root = Path(upstream_root or DEFAULT_UPSTREAM_ROOT).resolve()
    sources = {}
    for relative, expected in SOURCE_SHA256.items():
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing pinned QTC source: {path}. Run python fetch_qtc.py."
            )
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"Upstream QTC source hash mismatch: {path}")
        sources[relative] = actual
    return {
        "root": str(root),
        "repository": UPSTREAM_REPOSITORY,
        "commit": UPSTREAM_COMMIT,
        "source_sha256": sources,
    }


def _load_upstream(upstream_root):
    manifest = verify_upstream(upstream_root)
    namespace = dict(
        np=np, torch=torch, nn=nn, F=F, Any=Any, Union=Union, Counter=Counter
    )
    for relative, names in _SELECTIONS.items():
        path = Path(manifest["root"]) / relative
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256[relative]:
            raise RuntimeError(f"Upstream QTC source changed during loading: {path}")
        parsed = ast.parse(raw.decode("utf-8"), filename=str(path))
        selected = [
            node
            for node in parsed.body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names
        ]
        if [node.name for node in selected] != names:
            raise RuntimeError(f"Unexpected QTC source definitions: {path}")
        # Execute unchanged author definitions without importing unused training code.
        module = ast.Module(body=selected, type_ignores=[])
        exec(compile(module, str(path), "exec", dont_inherit=True), namespace)
    return namespace, manifest


def _probabilities(p, classes=None):
    values = np.asarray(p)
    if (
        values.ndim != 2
        or min(values.shape) < 1
        or values.shape[1] < 2
        or (classes is not None and values.shape[1] != classes)
    ):
        raise ValueError(
            "Expected nonempty N x K probabilities with matching class count"
        )
    if not np.issubdtype(values.dtype, np.number) or np.iscomplexobj(values):
        raise ValueError("Probabilities must be real numbers")
    if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
        raise ValueError("Invalid probabilities")
    if not np.allclose(values.sum(axis=1, dtype=np.float64), 1.0, atol=1e-4, rtol=0):
        raise ValueError("Probability rows must sum to one")
    return np.array(values, dtype=np.float32, copy=True), values.astype(
        np.float64, copy=False
    )


def _reconstruct(p32, original64):
    # Preserve zeros: upstream log(0)=-inf. CP's probability floor does not apply.
    logits = torch.log(torch.from_numpy(p32))
    recovered = F.softmax(logits / torch.Tensor([1.0]), dim=1)
    if not torch.isfinite(recovered).all():
        raise ValueError("Reconstructed QTC probabilities are nonfinite")
    metadata = {
        "input_cast_max_abs_difference": float(
            np.max(np.abs(p32.astype(np.float64) - original64))
        ),
        "reconstruction_max_abs_difference": float(
            np.max(np.abs(recovered.numpy().astype(np.float64) - original64))
        ),
        "input_zero_probability_count": int(np.sum(p32 == 0)),
        "sample_count": len(p32),
        "class_count": p32.shape[1],
    }
    return logits, recovered, metadata


class QTCSource:
    """Source calibration for min(QTC-T, QTC-S), with no target-label argument."""

    def __init__(self, p_source, y_source, *, upstream_root=None, boundary="infinity"):
        if boundary not in ("infinity", "cap"):
            raise ValueError("boundary must be 'infinity' or 'cap'")
        self.boundary = boundary
        self._namespace, upstream = _load_upstream(upstream_root)
        p32, original64 = _probabilities(p_source)
        labels = np.asarray(y_source)
        if (
            labels.ndim != 1
            or len(labels) != len(p32)
            or not np.issubdtype(labels.dtype, np.integer)
            or np.any(labels < 0)
            or np.any(labels >= p32.shape[1])
        ):
            raise ValueError("Expected one valid integer source label per sample")
        self._source_labels = torch.tensor(labels.astype(np.int64), dtype=torch.long)
        self._source_logits, _, reconstruction = _reconstruct(p32, original64)
        self._source_dataset = torch.utils.data.TensorDataset(
            self._source_logits, self._source_labels
        )
        # Upstream's constructor forces CUDA; these are its original CPU attributes.
        self._model = object.__new__(self._namespace["ThresholdedConformalModel"])
        self._model.batch_size = 512
        self._model.num_workers = 0
        self._model.is_softmax = False
        self._model.softmax_temp = torch.Tensor([1.0])
        self._model.randomized = True
        self._model.allow_zero_sets = True
        self._source_qtc = {}
        self._calibration_cache = {}
        self._cap = None
        self.calibration_calls = 0
        self.last_target_metadata = None
        self.metadata = {
            "wrapper_version": "paper-min-pinned-25b4-v1",
            "upstream": upstream,
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "device": "cpu",
            "dtype": "float32",
            "boundary": boundary,
            "reconstruction": "cached probabilities -> float32 -> torch.log -> upstream softmax; no clipping or extra input normalization",
            "original_classifier_logits_available": False,
            "source_reconstruction": reconstruction,
            "gamma": "min(gamma_T, gamma_S)",
            "source_quantile": "upstream NumPy linear interpolation",
        }

    def _maximum_source_score(self):
        if self._cap is None:
            scores = np.zeros(len(self._source_labels))
            loader = torch.utils.data.DataLoader(
                self._source_dataset,
                batch_size=512,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
            )
            with torch.inference_mode():
                for index, (logits, labels) in enumerate(loader):
                    probabilities = F.softmax(
                        logits / self._model.softmax_temp, dim=1
                    ).numpy()
                    start = index * 512
                    # Preserve upstream scalar arithmetic, rather than vectorized 1-p.
                    scores[start : start + len(logits)] = np.array(
                        [
                            self._model.get_tau(row, label)
                            for row, label in zip(probabilities, labels.numpy())
                        ]
                    )
            self._cap = float(np.quantile(scores, 1.0))
        return self._cap

    def _calibrate(self, gamma):
        gamma = float(gamma)
        if not math.isfinite(gamma) or not 0 <= gamma <= 1:
            raise ValueError("QTC gamma must be finite and in [0, 1]")
        if gamma not in self._calibration_cache:
            self._model.alpha = gamma
            level = (1 - gamma) * (1 + 1 / len(self._source_labels))
            self.calibration_calls += 1
            boundary_applied = False
            upstream_error = None
            try:
                q = float(self._model._get_Q_hat(self._source_dataset))
            except ValueError as error:
                if not (level > 1 and str(error) == _QUANTILE_DOMAIN_MESSAGE):
                    raise
                boundary_applied = True
                upstream_error = {"type": "ValueError", "message": str(error)}
                q = (
                    math.inf
                    if self.boundary == "infinity"
                    else self._maximum_source_score()
                )
            self._calibration_cache[gamma] = {
                "gamma": gamma,
                "q": q,
                "status": "ok",
                "error": None,
                "conformal_quantile_level": level,
                "boundary_applied": boundary_applied,
                "upstream_status": (
                    "upstream_quantile_domain_error" if boundary_applied else "ok"
                ),
                "upstream_error": upstream_error,
            }
        return self._calibration_cache[gamma]

    def evaluate(self, p_target, *, alphas=(0.05, 0.1)):
        """Return one threshold, boolean prediction mask and diagnostics per alpha."""
        levels = tuple(float(alpha) for alpha in alphas)
        if not levels or any(
            not math.isfinite(alpha) or not 0 < alpha < 1 for alpha in levels
        ):
            raise ValueError("alphas must be strictly between zero and one")
        if len(set(levels)) != len(levels):
            raise ValueError("Duplicate alpha values")
        p32, original64 = _probabilities(p_target, self._source_logits.shape[1])
        target_logits, recovered, metadata = _reconstruct(p32, original64)
        self.last_target_metadata = metadata
        dummy_labels = torch.zeros(len(p32), dtype=torch.long)
        cls = self._namespace["QuantileConfidenceThreshold"]
        rows, masks = [], {}
        with torch.inference_mode():
            for alpha in levels:
                target_qtc = cls(
                    target_logits, dummy_labels, quantile=alpha, softmax_temperature=1.0
                )
                if alpha not in self._source_qtc:
                    self._source_qtc[alpha] = cls(
                        self._source_logits,
                        self._source_labels,
                        quantile=1 - alpha,
                        softmax_temperature=1.0,
                    )
                source_qtc = self._source_qtc[alpha]
                gamma_t = target_qtc.predict(self._source_logits)
                gamma_s = 1.0 - source_qtc.predict(target_logits)
                gamma = min(gamma_t, gamma_s)
                calibrated = self._calibrate(gamma)
                if gamma not in masks:
                    mask = np.zeros(p32.shape, dtype=bool)
                    for i, probabilities in enumerate(recovered.numpy()):
                        selected = self._model._compute_S(
                            probabilities,
                            calibrated["q"],
                            randomized=True,
                            allow_zero_sets=True,
                        )
                        mask[i, selected] = True
                    masks[gamma] = mask
                rows.append(
                    {
                        "method": "qtc" if self.boundary == "infinity" else "qtc_cap",
                        "alpha": alpha,
                        "gamma_T": gamma_t,
                        "gamma_S": gamma_s,
                        "quantile_T": float(target_qtc.q.item()),
                        "quantile_S": float(source_qtc.q.item()),
                        "target_reconstruction_max_abs_difference": metadata[
                            "reconstruction_max_abs_difference"
                        ],
                        **calibrated,
                        "mask": masks[gamma],
                    }
                )
        return rows
