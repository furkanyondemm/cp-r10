"""Ten batch features, ridge regression and original source NLL targets."""

from __future__ import annotations
import numpy as np

FEATURES = [
    "msp_mean",
    "entropy_mean",
    "margin_mean",
    "msp_q10",
    "entropy_q90",
    "margin_q10",
    "prediction_hist_entropy",
    "pred_log_top1_top2_ratio_mean",
    "pred_log_top1_top2_ratio_q10",
    "pred_logprob_class_sd_mean",
]


CORRUPTIONS = [
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    "defocus_blur",
    "glass_blur",
    "motion_blur",
    "zoom_blur",
    "snow",
    "frost",
    "fog",
    "brightness",
    "contrast",
    "elastic_transform",
    "pixelate",
    "jpeg_compression",
]


ALPHAS = (0.05, 0.10)


def validate_probs(p, y=None, k=None):
    p = np.asarray(p, dtype=np.float64)
    if p.ndim != 2 or not len(p) or (k is not None and p.shape[1] != k):
        raise ValueError("Wrong probability shape")
    if not np.isfinite(p).all() or np.min(p) < -1e-7 or np.max(p) > 1 + 1e-7:
        raise ValueError("Invalid probabilities")
    if not np.allclose(p.sum(1), 1, atol=1e-4, rtol=0):
        raise ValueError("Unnormalized probabilities")
    if y is not None:
        y = np.asarray(y, dtype=np.int64).reshape(-1)
        if len(y) != len(p) or np.any(y < 0) or np.any(y >= p.shape[1]):
            raise ValueError("Invalid labels")
    return p


def r10_features(p):
    p = np.clip(validate_probs(p), 1e-12, 1)
    sp = np.sort(p, axis=1)[:, ::-1]
    msp = sp[:, 0]
    margin = sp[:, 0] - sp[:, 1]
    ent = -(p * np.log(p)).sum(1)
    hist = np.bincount(p.argmax(1), minlength=p.shape[1]).astype(float)
    hist /= hist.sum()
    hh = np.clip(hist, 1e-12, 1)
    lr = np.log(sp[:, 0]) - np.log(sp[:, 1])
    vals = [
        msp.mean(),
        ent.mean(),
        margin.mean(),
        np.quantile(msp, 0.1),
        np.quantile(ent, 0.9),
        np.quantile(margin, 0.1),
        -(hh * np.log(hh)).sum(),
        lr.mean(),
        np.quantile(lr, 0.1),
        np.log(p).std(1, ddof=0).mean(),
    ]
    return dict(zip(FEATURES, map(float, vals)))


def ridge_fit(X, y, alpha=0.1):
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    if (
        X.ndim != 2
        or X.shape[1] != 10
        or len(X) != len(y)
        or not np.isfinite(X).all()
        or not np.isfinite(y).all()
    ):
        raise ValueError("Bad Ridge data")
    mean = X.mean(0)
    scale = X.std(0, ddof=0)
    scale = np.where(scale < 1e-12, 1.0, scale)
    Z = (X - mean) / scale
    ybar = float(y.mean())
    coef = np.linalg.solve(Z.T @ Z + float(alpha) * np.eye(10), Z.T @ (y - ybar))
    return {
        "scaler_mean": mean.tolist(),
        "scaler_scale": scale.tolist(),
        "ridge_coef": coef.tolist(),
        "ridge_intercept": ybar,
        "ridge_alpha": float(alpha),
    }


def ridge_predict(model, X):
    X = np.asarray(X, float)
    mean = np.asarray(model["scaler_mean"], float)
    scale = np.asarray(model["scaler_scale"], float)
    coef = np.asarray(model["ridge_coef"], float)
    pred = float(model["ridge_intercept"]) + ((X - mean) / scale) @ coef
    return np.clip(pred, 0.0, 20.0)


def softmax(z):
    z = np.asarray(z, dtype=np.float64)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def probs_from_beta(p, beta):
    return softmax(
        np.log(np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1.0)) * float(beta)
    )


def fit_beta(p, y, lo=0.0, hi=20.0, iters=80):
    p = validate_probs(p, y)
    lp = np.log(np.clip(p, 1e-12, 1.0))
    yt = np.asarray(y, int)

    def obj(b):
        z = lp * b
        z -= z.max(1, keepdims=True)
        e = np.exp(z)
        logp = z - np.log(e.sum(1, keepdims=True))
        return float(-logp[np.arange(len(yt)), yt].mean())

    phi = (1 + 5**0.5) / 2
    a = float(lo)
    b = float(hi)
    c = b - (b - a) / phi
    d = a + (b - a) / phi
    fc = obj(c)
    fd = obj(d)
    for _ in range(iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - (b - a) / phi
            fc = obj(c)
        else:
            a, c, fc = c, d, fd
            d = a + (b - a) / phi
            fd = obj(d)
    mid = (a + b) / 2
    return float(min([(obj(lo), lo), (obj(hi), hi), (obj(mid), mid)])[1])


def original_nll_target(p, y, system):
    """Retain each reported main baseline's loss and endpoint conventions.

    Input probabilities always use 1e-12. The historical post-temperature
    loss safeguards are separate from this input floor.
    """
    if system.startswith("imagenet"):
        return fit_beta(p, y)
    p = validate_probs(p, y)
    y = np.asarray(y, dtype=int)
    output_floor = 1e-300 if system == "c10_resnet18" else 1e-15

    def objective(beta):
        true_p = probs_from_beta(p, beta)[np.arange(len(y)), y]
        return float(-np.log(np.clip(true_p, output_floor, 1.0)).mean())

    phi = (1 + 5**0.5) / 2
    a, b = 0.0, 20.0
    c, d = b - (b - a) / phi, a + (b - a) / phi
    fc, fd = objective(c), objective(d)
    for _ in range(80):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - (b - a) / phi
            fc = objective(c)
        else:
            a, c, fc = c, d, fd
            d = a + (b - a) / phi
            fd = objective(d)
    mid = (a + b) / 2
    if system == "c10_resnet18":
        selected = min(
            [(objective(mid), mid), (objective(0.0), 0.0), (objective(20.0), 20.0)],
            key=lambda row: row[0],
        )[1]
        return 0.0 if objective(0.0) <= objective(selected) + 1e-10 else float(selected)
    return float(
        min([(objective(0.0), 0.0), (objective(20.0), 20.0), (objective(mid), mid)])[1]
    )


def fit_original_nll(X, y, system):
    if system != "c10_resnet18":
        return ridge_fit(X, y)
    # Preserve the original CIFAR-10 NLL fit's sklearn centering/solver path.
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge

    pipeline = make_pipeline(StandardScaler(), Ridge(alpha=0.1)).fit(X, y)
    scaler, ridge = (
        pipeline.named_steps["standardscaler"],
        pipeline.named_steps["ridge"],
    )
    return dict(
        scaler_mean=scaler.mean_.tolist(),
        scaler_scale=scaler.scale_.tolist(),
        ridge_coef=ridge.coef_.tolist(),
        ridge_intercept=float(ridge.intercept_),
        ridge_alpha=0.1,
    )
