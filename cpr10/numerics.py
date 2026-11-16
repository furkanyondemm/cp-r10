"""Numerically stable temperature scaling and posterior-weighted LAC.

The conformal rule is unchanged: sort posterior probabilities in descending
order and include the first probability whose cumulative mass reaches
``min(n, ceil((n + 1) * (1 - alpha)))``.  Sets use the corresponding *log*
probability threshold.  A reported ``q = 1 - exp(log_threshold)`` is for display
only: near one, converting it back into a probability destroys information.

In the weighted quantile, probabilities above one half are represented as
``1 - deficit`` with ``deficit = -expm1(log_probability)``.  Integer counts,
deficits, and smaller probabilities are accumulated separately.  This avoids
both rounded-to-one probabilities and cumulative masses rounded to integers.
An adaptive Decimal calculation checks the rare ambiguous mass comparisons.
"""

from __future__ import annotations

from decimal import Decimal, localcontext
import math
from typing import Iterable

import numpy as np

NUMERICAL_VERSION = "stable-log-lac-count-deficit-v2"


class _TemperatureLogProbabilities(np.ndarray):
    """Keep source logits solely for exceptional high-precision comparisons."""

    def __new__(cls, values, source_logs, beta):
        obj = np.asarray(values).view(cls)
        obj._temperature_reference = (source_logs, float(beta))
        return obj

    def __array_finalize__(self, obj):
        # Slices and arithmetic may change the sample rows or temperature;
        # callers can still use those as ordinary normalized log probabilities.
        self._temperature_reference = None


def _matrix(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 2 or min(result.shape) < 1:
        raise ValueError("Expected a nonempty samples-by-classes matrix")
    return result


def _required_mass(n: int, alpha: float) -> int:
    if not math.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    return min(n, math.ceil((n + 1) * (1 - alpha)))


def temperature_log_probs(
    p: np.ndarray, beta: float, *, log_floor: float = 1e-12
) -> np.ndarray:
    """Scale clipped cached probabilities, retaining tiny deficits from one.

    ``log_floor`` is explicit because historical pipelines used different
    declared clipping floors.  At beta zero all classes remain exactly tied.
    """
    p = _matrix(p)
    if not np.isfinite(p).all() or np.any(p < 0) or np.any(p > 1 + 1e-7):
        raise ValueError("Invalid probabilities")
    if not math.isfinite(beta) or beta < 0:
        raise ValueError("beta must be finite and nonnegative")
    if not math.isfinite(log_floor) or not 0 < log_floor <= 1:
        raise ValueError("log_floor must lie in (0, 1]")
    if beta == 0:
        return np.full(p.shape, -math.log(p.shape[1]), dtype=np.float64)
    logs = np.log(np.clip(p, log_floor, 1.0))
    centered = (logs - logs.max(axis=1, keepdims=True)) * float(beta)
    maximum_indices = centered.argmax(axis=1)
    other_exponentials = np.exp(centered)
    other_exponentials[np.arange(len(p)), maximum_indices] = 0.0
    tail_mass = other_exponentials.sum(axis=1)
    if not np.isfinite(centered).all() or (p.shape[1] > 1 and np.any(tail_mass == 0)):
        raise ArithmeticError(
            "Temperature scaling exceeds float64 log-probability range; "
            "a maximum's positive tail underflowed. Use a declared supported "
            "clipping floor or a higher-precision evaluation."
        )
    log_normalizer = np.log1p(tail_mass)
    return _TemperatureLogProbabilities(centered - log_normalizer[:, None], logs, beta)


def _precise_prefix_compare(
    logp: np.ndarray, order: np.ndarray, length: int, target: int, reference=None
) -> int:
    """Compare a prefix mass with an integer without adding tiny tails to 1.

    Each normalized row contributes an exact integer (if its maximum is
    selected), minus its omitted mass, or its selected nonmaximum mass.  This
    preserves tiny deficits even when ordinary arbitrary-precision softmax
    would round its maximum probability to one.  Renormalization removes
    rounding in the supplied float log normalizers.
    """
    selected = np.zeros(logp.size, dtype=bool)
    selected[order[:length]] = True
    selected = selected.reshape(logp.shape)
    maxima = logp.argmax(axis=1)
    selected_maxima = selected[np.arange(len(logp)), maxima]
    integer_part = int(selected_maxima.sum()) - target
    active = np.flatnonzero(selected.any(axis=1) & ~selected.all(axis=1))
    if not len(active):
        return (integer_part > 0) - (integer_part < 0)
    # Agreement at increasing precision guards cancellation of substantial
    # terms; tiny-only deficits are retained at every precision.
    previous = None
    for precision in (80, 160, 320):
        with localcontext() as context:
            context.prec = precision
            positive = Decimal(0)
            negative = Decimal(0)
            for i in active:
                row = logp[i] if reference is None else reference[0][i]
                beta = (
                    Decimal(1)
                    if reference is None
                    else Decimal.from_float(reference[1])
                )
                maximum = Decimal.from_float(float(row[maxima[i]]))
                exponentials = [
                    ((Decimal.from_float(float(value)) - maximum) * beta).exp()
                    for value in row
                ]
                denominator = sum(exponentials, Decimal(0))
                if selected_maxima[i]:
                    negative += (
                        sum(
                            (
                                value
                                for j, value in enumerate(exponentials)
                                if not selected[i, j]
                            ),
                            Decimal(0),
                        )
                        / denominator
                    )
                else:
                    positive += (
                        sum(
                            (
                                value
                                for j, value in enumerate(exponentials)
                                if selected[i, j]
                            ),
                            Decimal(0),
                        )
                        / denominator
                    )
            difference = Decimal(integer_part) + positive - negative
            sign = int(difference > 0) - int(difference < 0)
            scale = abs(Decimal(integer_part)) + positive + negative
            error = scale * Decimal(10) ** (-(precision - 12))
            if abs(difference) > error and sign == previous:
                return sign
            # Repeated zeros at 80/160 digits do not certify equality: the
            # beta<=20, floor>=1e-12 protocol can retain 1e-240 residuals.
            previous = sign
    if difference == 0:
        return 0
    if abs(difference) <= error:
        raise ArithmeticError("Weighted quantile mass remains ambiguous at 320 digits")
    return sign


def plugin_log_thresholds(logp: np.ndarray, alphas: Iterable[float]) -> list[float]:
    """Return aligned log thresholds, sharing a single sort across alphas."""
    reference = getattr(logp, "_temperature_reference", None)
    logp = _matrix(logp)
    if not np.isfinite(logp).all() or np.any(logp > 0):
        raise ValueError("Expected finite nonpositive log probabilities")
    targets = [_required_mass(len(logp), float(alpha)) for alpha in alphas]
    if not targets:
        return []
    # Stable sorting keeps a row's first argmax before its other tied maxima.
    flat = logp.ravel()
    order = np.argsort(-flat, kind="stable")
    ordered = flat[order]
    if ordered[0] == ordered[-1]:
        return [float(ordered[0])] * len(targets)
    high_count = int(np.searchsorted(-ordered, math.log(2.0), side="right"))
    deficits = -np.expm1(ordered[:high_count])
    small = np.exp(ordered[high_count:])
    deficit_prefix = np.cumsum(deficits, dtype=np.longdouble)
    small_prefix = np.cumsum(small, dtype=np.longdouble)
    total_deficit = deficit_prefix[-1] if high_count else np.longdouble(0)
    eps = np.finfo(np.float64).eps
    accumulation_eps = np.finfo(np.longdouble).eps

    def compare(index: int, target: int) -> int:
        length = index + 1
        if length <= high_count:
            deficit, positive = deficit_prefix[index], np.longdouble(0)
            integer = length - target
        else:
            deficit, positive = total_deficit, small_prefix[index - high_count]
            integer = high_count - target
        difference = np.longdouble(integer) + positive - deficit
        # The scale deliberately excludes an artificial unit floor: a tiny
        # positive deficit certifies mass < target even at 1e-200.
        error = (16 * eps + 4 * length * accumulation_eps) * (positive + deficit)
        if abs(difference) <= error:
            return _precise_prefix_compare(logp, order, length, target, reference)
        return int(difference > 0) - int(difference < 0)

    thresholds = []
    for target in targets:
        if target == len(logp):
            # Every clipped posterior is positive, and the total is exactly n.
            thresholds.append(float(ordered[-1]))
            continue
        # Prefixes in the two regions are monotone.  Separate arithmetic keeps
        # an integer target from absorbing the deficit of a near-unit weight.
        left, right = 0, len(ordered) - 1
        while left < right:
            middle = (left + right) // 2
            if compare(middle, target) >= 0:
                right = middle
            else:
                left = middle + 1
        thresholds.append(float(ordered[left]))
    return thresholds


def plugin_log_threshold(logp: np.ndarray, alpha: float) -> float:
    return plugin_log_thresholds(logp, [alpha])[0]


def _labels(logp: np.ndarray, y: np.ndarray) -> np.ndarray:
    y = np.asarray(y)
    if y.ndim != 1 or len(y) != len(logp) or not np.issubdtype(y.dtype, np.integer):
        raise ValueError("Expected one integer label per sample")
    if np.any(y < 0) or np.any(y >= logp.shape[1]):
        raise ValueError("Label outside the class range")
    return y


def metrics_from_log_threshold(
    logp: np.ndarray, y: np.ndarray, threshold: float
) -> dict:
    logp = _matrix(logp)
    y = _labels(logp, y)
    if math.isnan(threshold):
        raise ValueError("NaN threshold")
    mask = logp >= threshold
    sizes = mask.sum(axis=1)
    return {
        "coverage": float(mask[np.arange(len(y)), y].mean()),
        "mean_set_size": float(sizes.mean()),
        "empty_rate": float((sizes == 0).mean()),
    }


def evaluate_plugin(
    p: np.ndarray, y: np.ndarray, beta: float, alpha: float, *, log_floor: float = 1e-12
) -> dict:
    logp = temperature_log_probs(p, beta, log_floor=log_floor)
    threshold = plugin_log_threshold(logp, alpha)
    metrics = metrics_from_log_threshold(logp, y, threshold)
    error = metrics["coverage"] - (1 - alpha)
    return {
        "beta": float(beta),
        "alpha": float(alpha),
        "q": float(-np.expm1(threshold)),
        "log_threshold": threshold,
        **metrics,
        "abs_coverage_error": abs(error),
        "signed_coverage_error": error,
    }
