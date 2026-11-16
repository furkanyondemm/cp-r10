"""Synthetic source conditions and the paper's fixed image transformations."""

from __future__ import annotations
import hashlib
from typing import Sequence
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import InterpolationMode, functional as TF

FAMILIES = [
    "gaussian_noise",
    "impulse_noise",
    "blur",
    "photometric",
    "color",
    "geometry",
    "occlusion",
    "resolution",
]


def make_specs(per_family=40, seed=20260908):
    rng = np.random.default_rng(seed)
    specs = []
    cid = 0
    for family in FAMILIES:
        for _ in range(per_family):
            u = float(rng.uniform(0.08, 1.0))
            spec = {
                "condition_id": f"syn_{cid:04d}",
                "family": family,
                "strength": u,
                "seed": int(rng.integers(0, 2**31 - 1)),
            }
            if family == "gaussian_noise":
                spec.update({"variant": "gaussian", "sigma": float(0.008 + 0.18 * u)})
            elif family == "impulse_noise":
                spec.update({"variant": "salt_pepper", "prob": float(0.002 + 0.16 * u)})
            elif family == "blur":
                v = str(rng.choice(["gaussian", "motion_like"]))
                spec.update({"variant": v, "sigma": float(0.15 + 2.2 * u)})
            elif family == "photometric":
                v = str(rng.choice(["brightness", "contrast", "gamma"]))
                direction = float(rng.choice([-1, 1]))
                if v == "brightness":
                    value = max(0.35, float(1 + direction * (0.08 + 0.52 * u)))
                elif v == "contrast":
                    value = max(0.25, float(1 + direction * (0.10 + 0.65 * u)))
                else:
                    value = float(np.exp(direction * (0.12 + 0.65 * u)))
                spec.update({"variant": v, "value": value})
            elif family == "color":
                v = str(rng.choice(["saturation", "channel_gain", "grayscale_mix"]))
                if v == "saturation":
                    direction = float(rng.choice([-1, 1]))
                    spec.update(
                        {
                            "variant": v,
                            "value": max(
                                0.05, float(1 + direction * (0.15 + 0.75 * u))
                            ),
                        }
                    )
                elif v == "channel_gain":
                    gains = np.clip(
                        1 + rng.normal(0, 0.08 + 0.28 * u, size=3), 0.45, 1.65
                    )
                    spec.update({"variant": v, "gains": [float(x) for x in gains]})
                else:
                    spec.update({"variant": v, "mix": float(0.08 + 0.82 * u)})
            elif family == "geometry":
                v = str(rng.choice(["rotate", "translate", "scale"]))
                if v == "rotate":
                    spec.update(
                        {
                            "variant": v,
                            "degrees": float(rng.choice([-1, 1]) * (1.5 + 18 * u)),
                        }
                    )
                elif v == "translate":
                    dx = int(np.sign(rng.normal()) * max(1, round(1 + 5 * u)))
                    dy = int(np.sign(rng.normal()) * max(1, round(1 + 5 * u)))
                    spec.update({"variant": v, "dx": dx, "dy": dy})
                else:
                    direction = float(rng.choice([-1, 1]))
                    spec.update(
                        {
                            "variant": v,
                            "scale": max(
                                0.72, float(1 + direction * (0.03 + 0.18 * u))
                            ),
                        }
                    )
            elif family == "occlusion":
                side = int(round(2 + 15 * u))
                cx = int(rng.integers(side // 2, 32 - max(1, side // 2)))
                cy = int(rng.integers(side // 2, 32 - max(1, side // 2)))
                spec.update({"variant": "square", "side": side, "cx": cx, "cy": cy})
            elif family == "resolution":
                v = str(rng.choice(["pixelate", "quantize"]))
                if v == "pixelate":
                    spec.update(
                        {"variant": v, "size": max(7, min(30, int(round(30 - 21 * u))))}
                    )
                else:
                    spec.update(
                        {
                            "variant": v,
                            "levels": max(6, min(64, int(round(64 - 56 * u)))),
                        }
                    )
            specs.append(spec)
            cid += 1
    return specs


def _rand_like(x, seed):
    g = torch.Generator(device=x.device)
    g.manual_seed(int(seed))
    return torch.rand(x.shape, generator=g, device=x.device, dtype=x.dtype)


def apply_spec(x, spec, pool_salt=0):
    from torchvision.transforms import functional as TF

    fam = spec["family"]
    seed = int(spec["seed"]) + int(pool_salt)
    if fam == "gaussian_noise":
        g = torch.Generator(device=x.device)
        g.manual_seed(seed)
        return (
            x
            + torch.randn(x.shape, generator=g, device=x.device, dtype=x.dtype)
            * float(spec["sigma"])
        ).clamp(0, 1)
    if fam == "impulse_noise":
        r = _rand_like(x, seed)
        p = float(spec["prob"])
        y = x.clone()
        y[r < p / 2] = 0
        y[(r >= p / 2) & (r < p)] = 1
        return y
    if fam == "blur":
        sigma = float(spec["sigma"])
        if spec["variant"] == "gaussian":
            return TF.gaussian_blur(x, [5, 5], [sigma, sigma]).clamp(0, 1)
        k = int(max(3, min(9, 2 * round(1 + 2.5 * spec["strength"]) + 1)))
        if seed % 2:
            return F.avg_pool2d(
                F.pad(x, (k // 2, k // 2, 0, 0), mode="reflect"), (1, k), stride=1
            )
        return F.avg_pool2d(
            F.pad(x, (0, 0, k // 2, k // 2), mode="reflect"), (k, 1), stride=1
        )
    if fam == "photometric":
        v = float(spec["value"])
        if spec["variant"] == "brightness":
            return TF.adjust_brightness(x, v).clamp(0, 1)
        if spec["variant"] == "contrast":
            return TF.adjust_contrast(x, v).clamp(0, 1)
        return TF.adjust_gamma(x, v).clamp(0, 1)
    if fam == "color":
        if spec["variant"] == "saturation":
            return TF.adjust_saturation(x, float(spec["value"])).clamp(0, 1)
        if spec["variant"] == "channel_gain":
            g = torch.tensor(spec["gains"], device=x.device, dtype=x.dtype).view(
                1, 3, 1, 1
            )
            return (x * g).clamp(0, 1)
        gray = TF.rgb_to_grayscale(x, num_output_channels=3)
        mix = float(spec["mix"])
        return ((1 - mix) * x + mix * gray).clamp(0, 1)
    if fam == "geometry":
        if spec["variant"] == "rotate":
            return TF.rotate(
                x,
                float(spec["degrees"]),
                interpolation=TF.InterpolationMode.BILINEAR,
                fill=0,
            ).clamp(0, 1)
        if spec["variant"] == "translate":
            return TF.affine(
                x,
                0,
                [int(spec["dx"]), int(spec["dy"])],
                1.0,
                [0.0, 0.0],
                interpolation=TF.InterpolationMode.BILINEAR,
                fill=0,
            ).clamp(0, 1)
        return TF.affine(
            x,
            0,
            [0, 0],
            float(spec["scale"]),
            [0.0, 0.0],
            interpolation=TF.InterpolationMode.BILINEAR,
            fill=0,
        ).clamp(0, 1)
    if fam == "occlusion":
        y = x.clone()
        side = int(spec["side"])
        cx = int(spec["cx"])
        cy = int(spec["cy"])
        x0 = max(0, min(32 - side, cx - side // 2))
        y0 = max(0, min(32 - side, cy - side // 2))
        y[:, :, y0 : y0 + side, x0 : x0 + side] = 0
        return y
    if fam == "resolution":
        if spec["variant"] == "pixelate":
            s = int(spec["size"])
            y = F.interpolate(x, (s, s), mode="bilinear", align_corners=False)
            return F.interpolate(y, (32, 32), mode="nearest").clamp(0, 1)
        levels = float(spec["levels"])
        return (torch.round(x * (levels - 1)) / (levels - 1)).clamp(0, 1)
    raise ValueError(fam)


def sample_seed(spec_seed: int, image_id: str, pool_salt: int = 0) -> int:
    value = f"imagenet-synthetic-v1\0{int(spec_seed)}\0{int(pool_salt)}\0{image_id}"
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "little") % (
        2**63 - 1
    )


def _scaled_odd(kernel: int, side: int) -> int:
    # Scale the kernel footprint and round to the nearest odd size (up on ties).
    size = max(1, int(round(kernel * side / 32)))
    return size if size % 2 else size + 1


def _apply_one(
    x: torch.Tensor, spec: dict, image_id: str, pool_salt: int
) -> torch.Tensor:
    if x.ndim != 3 or x.shape[0] != 3 or x.shape[-2] != x.shape[-1]:
        raise ValueError("Synthetic transforms require a square 3-channel CHW tensor")
    side = x.shape[-1]
    scale = side / 32
    seed = sample_seed(spec["seed"], image_id, pool_salt)
    generator = torch.Generator(device=x.device).manual_seed(seed)
    family, variant = spec["family"], spec["variant"]
    if family == "gaussian_noise":
        noise = torch.randn(
            x.shape, generator=generator, device=x.device, dtype=x.dtype
        )
        return (x + noise * float(spec["sigma"])).clamp(0, 1)
    if family == "impulse_noise":
        noise = torch.rand(x.shape, generator=generator, device=x.device, dtype=x.dtype)
        probability = float(spec["prob"])
        output = x.clone()
        output[noise < probability / 2] = 0
        output[(noise >= probability / 2) & (noise < probability)] = 1
        return output
    if family == "blur":
        if variant == "gaussian":
            kernel, sigma = _scaled_odd(5, side), float(spec["sigma"]) * scale
            return TF.gaussian_blur(x, [kernel, kernel], [sigma, sigma]).clamp(0, 1)
        base = int(max(3, min(9, 2 * round(1 + 2.5 * spec["strength"]) + 1)))
        kernel = _scaled_odd(base, side)
        # Orientation is a condition-level choice, as in the original generator.
        if (int(spec["seed"]) + int(pool_salt)) % 2:
            return F.avg_pool2d(
                F.pad(x, (kernel // 2, kernel // 2, 0, 0), mode="reflect"),
                (1, kernel),
                stride=1,
            )
        return F.avg_pool2d(
            F.pad(x, (0, 0, kernel // 2, kernel // 2), mode="reflect"),
            (kernel, 1),
            stride=1,
        )
    if family == "photometric":
        value = float(spec["value"])
        operation = {
            "brightness": TF.adjust_brightness,
            "contrast": TF.adjust_contrast,
            "gamma": TF.adjust_gamma,
        }[variant]
        return operation(x, value).clamp(0, 1)
    if family == "color":
        if variant == "saturation":
            return TF.adjust_saturation(x, float(spec["value"])).clamp(0, 1)
        if variant == "channel_gain":
            gains = torch.as_tensor(spec["gains"], device=x.device, dtype=x.dtype).view(
                3, 1, 1
            )
            return (x * gains).clamp(0, 1)
        gray = TF.rgb_to_grayscale(x, num_output_channels=3)
        return ((1 - float(spec["mix"])) * x + float(spec["mix"]) * gray).clamp(0, 1)
    if family == "geometry":
        if variant == "rotate":
            return TF.rotate(
                x,
                float(spec["degrees"]),
                interpolation=InterpolationMode.BILINEAR,
                fill=0,
            ).clamp(0, 1)
        translation = (
            [int(round(spec["dx"] * scale)), int(round(spec["dy"] * scale))]
            if variant == "translate"
            else [0, 0]
        )
        factor = float(spec["scale"]) if variant == "scale" else 1.0
        return TF.affine(
            x,
            0,
            translation,
            factor,
            [0.0, 0.0],
            interpolation=InterpolationMode.BILINEAR,
            fill=0,
        ).clamp(0, 1)
    if family == "occlusion":
        # Scale the original clipped pixel bounds; preserve relative placement.
        original_side = int(spec["side"])
        x0 = max(0, min(32 - original_side, int(spec["cx"]) - original_side // 2))
        y0 = max(0, min(32 - original_side, int(spec["cy"]) - original_side // 2))
        output = x.clone()
        output[
            :,
            round(y0 * scale) : round((y0 + original_side) * scale),
            round(x0 * scale) : round((x0 + original_side) * scale),
        ] = 0
        return output
    if family == "resolution":
        if variant == "pixelate":
            reduced = max(1, int(round(float(spec["size"]) * scale)))
            output = F.interpolate(
                x.unsqueeze(0), (reduced, reduced), mode="bilinear", align_corners=False
            )
            return (
                F.interpolate(output, (side, side), mode="nearest")
                .squeeze(0)
                .clamp(0, 1)
            )
        levels = float(spec["levels"])
        return (torch.round(x * (levels - 1)) / (levels - 1)).clamp(0, 1)
    raise ValueError(f"Unknown synthetic family: {family}")


def apply_synthetic(
    images: torch.Tensor,
    spec: dict,
    image_ids: str | Sequence[str],
    pool_salt: int = 0,
) -> torch.Tensor:
    """Apply one condition with identity-keyed per-image random generators."""
    if images.ndim == 3:
        if not isinstance(image_ids, str):
            raise ValueError("One CHW image requires one string image ID")
        return _apply_one(images, spec, image_ids, pool_salt)
    if images.ndim != 4 or isinstance(image_ids, str) or len(image_ids) != len(images):
        raise ValueError("A BCHW batch requires one image ID per row")
    return torch.stack(
        [
            _apply_one(x, spec, str(image_id), pool_salt)
            for x, image_id in zip(images, image_ids)
        ]
    )
