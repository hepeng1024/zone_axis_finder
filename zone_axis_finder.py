#!/usr/bin/env python3
"""
Find reachable FCC zone axes from one indexed TEM diffraction pattern.

The script does three jobs:
1. Detect diffraction spots in an experimental image.
2. Match the spots against analytic FCC [100], [110], and [111] zone
   patterns while allowing arbitrary in-plane rotation.
3. Use the indexed pattern plus the current alpha/beta angles to predict the
   holder angles for other symmetry-equivalent <100>, <110>, and <111> axes.

Coordinate conventions used by the default holder model:
- alpha is a right-handed rotation about holder X.
- beta is a right-handed rotation about holder Y.
- rotation order "xy" means R = Rx(alpha) @ Ry(beta).
- image +x is to the right; image +y is up, so pixel rows are inverted.
- --image-to-holder-rotation-deg is the in-plane CCW angle from image +x to
  holder +X. Keep it at 0 only if the camera axes are calibrated that way.

Important: a single centrosymmetric diffraction pattern cannot determine every
instrument sign convention by itself. If a known calibration move predicts the
opposite alpha or beta sign, change --image-to-holder-rotation-deg, use the
other --holder-order, or invert the corresponding microscope sign convention.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Iterable, Sequence


try:
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    from scipy import ndimage as ndi
    from scipy.spatial import cKDTree
except ImportError as exc:  # pragma: no cover - exercised by users' envs.
    raise SystemExit(
        "Missing Python package: {0}\n\n"
        "This script needs numpy, pillow, and scipy. On this machine the "
        "Conda env named 'test' appears to have them, for example:\n"
        "  /home/hepeng/miniconda3/envs/test/bin/python "
        "/home/hepeng/find_zone_axis/zone_axis_finder.py IMAGE --alpha 0 --beta 0"
        .format(exc.name)
    )


Array = "np.ndarray"
SUPPORTED_ZONE_FAMILIES = ("100", "110", "111", "102", "103", "104", "112", "113", "114")
ZONE_FAMILY_COLORS: dict[str, tuple[int, int, int]] = {
    "100": (31, 119, 180),
    "110": (255, 127, 14),
    "111": (44, 160, 44),
    "102": (214, 39, 40),
    "103": (148, 103, 189),
    "104": (140, 86, 75),
    "112": (23, 190, 207),
    "113": (188, 189, 34),
    "114": (227, 119, 194),
}
ZONE_OUTLINE_PALETTE: tuple[tuple[int, int, int], ...] = (
    (12, 63, 180),
    (190, 24, 32),
    (0, 132, 73),
    (120, 37, 179),
    (214, 92, 0),
    (0, 145, 160),
    (170, 0, 110),
    (92, 92, 0),
    (80, 80, 80),
    (18, 105, 60),
    (122, 74, 0),
    (95, 70, 190),
)


@dataclass(frozen=True)
class Peak:
    x: float
    y: float
    value: float


@dataclass(frozen=True)
class Reflection:
    h: int
    k: int
    l: int
    xy: tuple[float, float]
    g_norm: float


@dataclass
class PatternMatch:
    family: str
    zone: tuple[int, int, int]
    basis_x: Array
    basis_y: Array
    reflections: list[Reflection]
    linear_unit: Array
    scale: float
    translation: Array
    matched_indices: dict[int, int]
    visible_indices: list[int]
    rms_px: float
    tolerance_px: float
    score: float

    @property
    def matched_count(self) -> int:
        return len(self.matched_indices)

    @property
    def visible_count(self) -> int:
        return len(self.visible_indices)

    @property
    def quality(self) -> float:
        if not self.visible_indices:
            return 0.0
        return self.matched_count / len(self.visible_indices)

    @property
    def angle_deg(self) -> float:
        return math.degrees(math.atan2(self.linear_unit[1, 0], self.linear_unit[0, 0]))


@dataclass(frozen=True)
class ZoneAxisMapPoint:
    family: str
    zone: tuple[int, int, int]
    alpha_deg: float
    beta_deg: float
    is_current: bool = False
    within_limits: bool = True


def normalize(v: Array) -> Array:
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    if n == 0:
        raise ValueError("Cannot normalize a zero vector.")
    return v / n


def gcd3(values: Iterable[int]) -> int:
    g = 0
    for value in values:
        g = math.gcd(g, abs(int(value)))
    return max(g, 1)


def reduce_miller(v: Sequence[int]) -> tuple[int, int, int]:
    g = gcd3(v)
    out = tuple(int(x // g) for x in v)
    return out  # type: ignore[return-value]


def canonical_line(v: Sequence[int]) -> tuple[int, int, int]:
    """Return one representative for an antipodal zone-axis line."""
    out = reduce_miller(v)
    for value in out:
        if value < 0:
            return tuple(-x for x in out)  # type: ignore[return-value]
        if value > 0:
            return out
    return out


def zone_outline_color(v: Sequence[int]) -> tuple[int, int, int]:
    """Return a stable outline color for a specific zone-axis direction."""
    reduced = reduce_miller(v)
    seed = 0
    for idx, value in enumerate(reduced):
        seed = (seed * 41 + (int(value) + 11) * (idx + 3)) % len(ZONE_OUTLINE_PALETTE)
    return ZONE_OUTLINE_PALETTE[seed]


def parse_miller(text: str) -> tuple[int, int, int]:
    cleaned = (
        text.strip()
        .replace("[", " ")
        .replace("]", " ")
        .replace("(", " ")
        .replace(")", " ")
        .replace(",", " ")
    )
    parts = [p for p in cleaned.split() if p]
    if len(parts) == 1 and len(parts[0]) == 3 and parts[0].lstrip("-").isdigit():
        parts = list(parts[0])
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"Expected a Miller direction like '1,0,0' or '1 1 0', got {text!r}."
        )
    try:
        return reduce_miller(tuple(int(p) for p in parts))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def format_miller(v: Sequence[int], brackets: str = "[]") -> str:
    left, right = brackets[0], brackets[1]
    return left + " ".join(f"{int(x):+d}" if int(x) < 0 else f"{int(x):d}" for x in v) + right


def family_name(v: Sequence[int]) -> str:
    vals = sorted(abs(int(x)) for x in reduce_miller(v))
    for family in SUPPORTED_ZONE_FAMILIES:
        if vals == sorted(parse_family_base(family)):
            return family
    return "".join(str(x) for x in sorted(vals, reverse=True))


def parse_family_base(family: str) -> tuple[int, int, int]:
    cleaned = family.strip().replace("<", "").replace(">", "").replace(",", " ").replace("_", " ")
    parts = [p for p in cleaned.split() if p]
    if len(parts) == 1:
        token = parts[0]
        if len(token) == 3 and token.isdigit():
            parts = list(token)
        else:
            raise ValueError(f"Unsupported FCC family <{family}>. Use values such as 100, 110, 111, 102, or 114.")
    if len(parts) != 3:
        raise ValueError(f"Unsupported FCC family <{family}>. Use values such as 100, 110, 111, 102, or 114.")
    try:
        base = tuple(abs(int(p)) for p in parts)
    except ValueError as exc:
        raise ValueError(f"Unsupported FCC family <{family}>. Use integer Miller indices.") from exc
    if base == (0, 0, 0):
        raise ValueError("A zone-axis family cannot be <000>.")
    return reduce_miller(base)


def family_directions(family: str, include_opposites: bool = False) -> list[tuple[int, int, int]]:
    base = parse_family_base(family)

    dirs: set[tuple[int, int, int]] = set()
    for perm in set(itertools.permutations(base, 3)):
        nonzero = [i for i, value in enumerate(perm) if value != 0]
        for signs in itertools.product((-1, 1), repeat=len(nonzero)):
            vals = list(perm)
            for idx, sign in zip(nonzero, signs):
                vals[idx] *= sign
            d = reduce_miller(vals)
            if include_opposites:
                dirs.add(d)
            else:
                dirs.add(canonical_line(d))
    return sorted(dirs)


def fcc_allowed(h: int, k: int, l: int) -> bool:
    parities = (h & 1, k & 1, l & 1)
    return parities[0] == parities[1] == parities[2]


def perpendicular_basis(zone: Sequence[int]) -> tuple[Array, Array]:
    z = normalize(np.asarray(zone, dtype=float))
    trial = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(trial, z))) > 0.9:
        trial = np.array([0.0, 1.0, 0.0])
    bx = normalize(trial - np.dot(trial, z) * z)
    by = normalize(np.cross(z, bx))
    return bx, by


def make_fcc_reflections(
    zone: Sequence[int],
    max_index: int = 4,
    max_g_norm: float = 5.05,
) -> tuple[list[Reflection], Array, Array]:
    z = np.asarray(reduce_miller(zone), dtype=int)
    bx, by = perpendicular_basis(z)
    reflections: list[Reflection] = [
        Reflection(0, 0, 0, (0.0, 0.0), 0.0)
    ]
    for h in range(-max_index, max_index + 1):
        for k in range(-max_index, max_index + 1):
            for l in range(-max_index, max_index + 1):
                if h == k == l == 0:
                    continue
                if h * z[0] + k * z[1] + l * z[2] != 0:
                    continue
                if not fcc_allowed(h, k, l):
                    continue
                g = np.array([h, k, l], dtype=float)
                g_norm = float(np.linalg.norm(g))
                if g_norm > max_g_norm:
                    continue
                xy = (float(np.dot(g, bx)), float(np.dot(g, by)))
                reflections.append(Reflection(h, k, l, xy, g_norm))

    reflections.sort(key=lambda r: (r.g_norm, r.h, r.k, r.l))
    return reflections, bx, by


def load_grayscale(path: Path, invert: bool = False) -> tuple[Array, Image.Image]:
    image = Image.open(path)
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    if invert:
        gray = float(gray.max()) - gray
    return gray, image.convert("RGB")


def detect_scale_bar_pixels(path: Path) -> dict[str, float] | None:
    gray = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
    height, width = gray.shape
    roi_top = int(height * 0.55)
    roi = gray[roi_top:, :]
    if roi.size == 0:
        return None

    percentile_threshold = float(np.percentile(roi, 99.6))
    thresholds = sorted({250.0, 240.0, 230.0, max(180.0, percentile_threshold)}, reverse=True)
    min_run = max(30, int(width * 0.025))
    best: tuple[int, int, int, int] | None = None

    for threshold in thresholds:
        mask = roi >= threshold
        for y_local in range(mask.shape[0]):
            xs = np.flatnonzero(mask[y_local])
            if xs.size < min_run:
                continue
            start = int(xs[0])
            previous = int(xs[0])
            for x in xs[1:]:
                x_int = int(x)
                if x_int == previous + 1:
                    previous = x_int
                    continue
                length = previous - start + 1
                if length >= min_run and (best is None or length > best[3]):
                    best = (start, previous, roi_top + y_local, length)
                start = previous = x_int
            length = previous - start + 1
            if length >= min_run and (best is None or length > best[3]):
                best = (start, previous, roi_top + y_local, length)
        if best is not None:
            break

    if best is None:
        return None
    x1, x2, y, length = best
    return {
        "length_px": float(length),
        "x1": float(x1),
        "x2": float(x2),
        "y": float(y),
    }


def nms_from_smoothed(
    smoothed: Array,
    percentile: float,
    n_peaks: int,
    min_distance_px: float,
    candidate_multiplier: int = 1200,
) -> list[Peak]:
    flat = smoothed.ravel()
    threshold = float(np.percentile(flat, percentile))
    idx = np.flatnonzero(flat >= threshold)
    if idx.size == 0:
        return []

    candidate_limit = min(idx.size, max(n_peaks * candidate_multiplier, n_peaks))
    if idx.size > candidate_limit:
        local_values = flat[idx]
        keep = np.argpartition(local_values, -candidate_limit)[-candidate_limit:]
        idx = idx[keep]

    values = flat[idx]
    order = np.argsort(values)[::-1]
    idx = idx[order]
    values = values[order]

    h, w = smoothed.shape
    min_d2 = float(min_distance_px * min_distance_px)
    selected: list[Peak] = []
    selected_xy: list[tuple[float, float]] = []
    for flat_idx, value in zip(idx, values):
        y, x = divmod(int(flat_idx), w)
        xf, yf = float(x), float(y)
        if any((xf - sx) ** 2 + (yf - sy) ** 2 < min_d2 for sx, sy in selected_xy):
            continue
        selected.append(Peak(xf, yf, float(value)))
        selected_xy.append((xf, yf))
        if len(selected) >= n_peaks:
            break
    return selected


def detect_spots(
    gray: Array,
    n_peaks: int = 80,
    min_distance_px: float | None = None,
    spot_sigma_px: float | None = None,
    peak_percentile: float = 99.0,
) -> list[Peak]:
    h, w = gray.shape
    min_dim = min(h, w)
    if spot_sigma_px is None:
        spot_sigma_px = max(1.5, min_dim / 300.0)
    if min_distance_px is None:
        min_distance_px = max(10.0, min_dim / 28.0)

    smoothed = ndi.gaussian_filter(gray, spot_sigma_px)
    peaks = nms_from_smoothed(smoothed, peak_percentile, n_peaks, min_distance_px)
    if len(peaks) < 8 and peak_percentile > 95.0:
        peaks = nms_from_smoothed(smoothed, 95.0, n_peaks, min_distance_px)
    return peaks


def as_screen_points(peaks: Sequence[Peak]) -> Array:
    return np.asarray([[p.x, -p.y] for p in peaks], dtype=float)


def rotation2(angle_rad: float) -> Array:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[c, -s], [s, c]], dtype=float)


def apply_transform(model_xy: Array, linear: Array, translation: Array) -> Array:
    return model_xy @ linear.T + translation


def visible_mask(points: Array, width: int, height: int, margin: float = 10.0) -> Array:
    x = points[:, 0]
    y_pixel = -points[:, 1]
    return (
        (x >= -margin)
        & (x <= width + margin)
        & (y_pixel >= -margin)
        & (y_pixel <= height + margin)
    )


def fit_similarity(model_xy: Array, obs_xy: Array) -> tuple[Array, float, Array] | None:
    if len(model_xy) < 2:
        return None
    xm = model_xy.mean(axis=0)
    ym = obs_xy.mean(axis=0)
    x = model_xy - xm
    y = obs_xy - ym
    denom = float(np.sum(x * x))
    if denom <= 0:
        return None
    u, svals, vt = np.linalg.svd(x.T @ y)
    rot = vt.T @ u.T
    if np.linalg.det(rot) < 0:
        vt[-1, :] *= -1
        rot = vt.T @ u.T
    scale = float(np.sum(svals) / denom)
    translation = ym - scale * (rot @ xm)
    return rot, scale, translation


def score_transform(
    reflections: Sequence[Reflection],
    peaks_screen: Array,
    linear_unit: Array,
    scale: float,
    translation: Array,
    image_size: tuple[int, int],
    tolerance_fraction: float,
) -> tuple[float, dict[int, int], list[int], float, float]:
    width, height = image_size
    model_xy = np.asarray([r.xy for r in reflections], dtype=float)
    pred = apply_transform(model_xy, scale * linear_unit, translation)
    vis = visible_mask(pred, width, height)
    visible_indices = [int(i) for i in np.flatnonzero(vis)]
    if not visible_indices:
        return -1e9, {}, [], float("inf"), 0.0

    min_shell = min((r.g_norm for r in reflections if r.g_norm > 0), default=1.0)
    tolerance_px = max(10.0, tolerance_fraction * scale * min_shell)
    tree = cKDTree(peaks_screen)
    dists, obs_indices = tree.query(pred[vis], k=1)

    candidates: list[tuple[float, int, int]] = []
    for ref_idx, dist, obs_idx in zip(visible_indices, dists, obs_indices):
        if float(dist) <= tolerance_px:
            candidates.append((float(dist), ref_idx, int(obs_idx)))

    # Enforce one observed peak per reflection and one reflection per peak.
    candidates.sort(key=lambda item: item[0])
    matched: dict[int, int] = {}
    used_obs: set[int] = set()
    used_ref: set[int] = set()
    distances: list[float] = []
    for dist, ref_idx, obs_idx in candidates:
        if ref_idx in used_ref or obs_idx in used_obs:
            continue
        matched[ref_idx] = obs_idx
        used_ref.add(ref_idx)
        used_obs.add(obs_idx)
        distances.append(dist)

    rms = math.sqrt(float(np.mean(np.square(distances)))) if distances else float("inf")
    matched_count = len(matched)
    visible_count = len(visible_indices)
    quality = matched_count / visible_count
    normalized_rms = rms / tolerance_px if distances else 999.0
    score = matched_count + 4.0 * quality - 0.25 * normalized_rms
    return score, matched, visible_indices, rms, tolerance_px


def refine_match(
    reflections: Sequence[Reflection],
    peaks_screen: Array,
    linear_unit: Array,
    scale: float,
    translation: Array,
    image_size: tuple[int, int],
    tolerance_fraction: float,
    iterations: int = 4,
) -> tuple[Array, float, Array, float, dict[int, int], list[int], float, float]:
    model_xy = np.asarray([r.xy for r in reflections], dtype=float)
    best = score_transform(
        reflections,
        peaks_screen,
        linear_unit,
        scale,
        translation,
        image_size,
        tolerance_fraction,
    )
    best_score, best_matched, best_visible, best_rms, best_tol = best
    best_linear, best_scale, best_translation = linear_unit, scale, translation

    for _ in range(iterations):
        if len(best_matched) < 3:
            break
        ref_indices = sorted(best_matched)
        obs_indices = [best_matched[i] for i in ref_indices]
        fit = fit_similarity(model_xy[ref_indices], peaks_screen[obs_indices])
        if fit is None:
            break
        rot, new_scale, new_translation = fit
        scored = score_transform(
            reflections,
            peaks_screen,
            rot,
            new_scale,
            new_translation,
            image_size,
            tolerance_fraction,
        )
        score, matched, visible, rms, tol = scored
        if score + 1e-9 < best_score:
            break
        best_score = score
        best_matched = matched
        best_visible = visible
        best_rms = rms
        best_tol = tol
        best_linear = rot
        best_scale = new_scale
        best_translation = new_translation

    return (
        best_linear,
        best_scale,
        best_translation,
        best_score,
        best_matched,
        best_visible,
        best_rms,
        best_tol,
    )


def match_pattern(
    family: str,
    zone: Sequence[int],
    peaks: Sequence[Peak],
    image_size: tuple[int, int],
    center_xy: tuple[float, float] | None,
    max_index: int,
    max_g_norm: float,
    tolerance_fraction: float,
) -> PatternMatch | None:
    reflections, bx, by = make_fcc_reflections(zone, max_index=max_index, max_g_norm=max_g_norm)
    model_xy = np.asarray([r.xy for r in reflections], dtype=float)
    peaks_screen = as_screen_points(peaks)
    if len(peaks_screen) < 4:
        return None

    if center_xy is None:
        width, height = image_size
        image_center = np.array([width / 2.0, height / 2.0], dtype=float)
        peak_image = np.asarray([[p.x, p.y] for p in peaks], dtype=float)
        center_idx = int(np.argmin(np.linalg.norm(peak_image - image_center, axis=1)))
        center_screen = peaks_screen[center_idx]
    else:
        center_screen = np.array([center_xy[0], -center_xy[1]], dtype=float)

    nonzero_indices = [i for i, r in enumerate(reflections) if r.g_norm > 0]
    nonzero_indices.sort(key=lambda i: reflections[i].g_norm)
    seed_model_indices = nonzero_indices[: min(16, len(nonzero_indices))]

    obs_seed_indices = [
        i for i in range(len(peaks_screen))
        if np.linalg.norm(peaks_screen[i] - center_screen) > 10.0
    ]
    if not obs_seed_indices:
        return None
    obs_seed_indices = obs_seed_indices[: min(50, len(obs_seed_indices))]

    best: PatternMatch | None = None
    for mi in seed_model_indices:
        m = model_xy[mi]
        m_norm = float(np.linalg.norm(m))
        if m_norm <= 0:
            continue
        for oi in obs_seed_indices:
            obs_vec = peaks_screen[oi] - center_screen
            obs_norm = float(np.linalg.norm(obs_vec))
            if obs_norm <= 0:
                continue
            scale = obs_norm / m_norm
            if scale < 2.0:
                continue
            angle = math.atan2(obs_vec[1], obs_vec[0]) - math.atan2(m[1], m[0])
            linear_unit = rotation2(angle)
            translation = center_screen.copy()
            refined = refine_match(
                reflections,
                peaks_screen,
                linear_unit,
                scale,
                translation,
                image_size,
                tolerance_fraction,
            )
            (
                unit,
                refined_scale,
                refined_translation,
                score,
                matched,
                visible,
                rms,
                tol,
            ) = refined
            result = PatternMatch(
                family=family,
                zone=reduce_miller(zone),
                basis_x=bx,
                basis_y=by,
                reflections=reflections,
                linear_unit=unit,
                scale=refined_scale,
                translation=refined_translation,
                matched_indices=matched,
                visible_indices=visible,
                rms_px=rms,
                tolerance_px=tol,
                score=score,
            )
            if best is None or result.score > best.score:
                best = result

    return best


def choose_best_match(
    peaks: Sequence[Peak],
    image_size: tuple[int, int],
    current_zone: tuple[int, int, int] | None,
    center_xy: tuple[float, float] | None,
    max_index: int,
    max_g_norm: float,
    tolerance_fraction: float,
) -> tuple[PatternMatch, list[PatternMatch]]:
    candidates: list[tuple[str, tuple[int, int, int]]] = []
    if current_zone is not None:
        candidates.append((family_name(current_zone), current_zone))
    else:
        candidates.extend((family, parse_family_base(family)) for family in SUPPORTED_ZONE_FAMILIES)

    results: list[PatternMatch] = []
    for family, zone in candidates:
        result = match_pattern(
            family,
            zone,
            peaks,
            image_size,
            center_xy,
            max_index,
            max_g_norm,
            tolerance_fraction,
        )
        if result is not None:
            results.append(result)

    if not results:
        raise RuntimeError("Could not match the detected spots to any FCC reference pattern.")
    results.sort(key=lambda r: r.score, reverse=True)
    return results[0], results


def rotate_match_in_plane(
    match: PatternMatch,
    angle_deg: float,
    peaks: Sequence[Peak],
    image_size: tuple[int, int],
    tolerance_fraction: float,
) -> PatternMatch:
    """Return a copy of a fitted match with its in-plane indexing rotated."""
    linear_unit = rotation2(math.radians(angle_deg)) @ match.linear_unit
    peaks_screen = as_screen_points(peaks)
    score, matched, visible, rms, tolerance = score_transform(
        match.reflections,
        peaks_screen,
        linear_unit,
        match.scale,
        match.translation,
        image_size,
        tolerance_fraction,
    )
    return replace(
        match,
        linear_unit=linear_unit,
        matched_indices=matched,
        visible_indices=visible,
        rms_px=rms,
        tolerance_px=tolerance,
        score=score,
    )


def holder_rotation(alpha_deg: float, beta_deg: float, order: str = "xy") -> Array:
    a = math.radians(alpha_deg)
    b = math.radians(beta_deg)
    ca, sa = math.cos(a), math.sin(a)
    cb, sb = math.cos(b), math.sin(b)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, ca, -sa], [0.0, sa, ca]], dtype=float)
    ry = np.array([[cb, 0.0, sb], [0.0, 1.0, 0.0], [-sb, 0.0, cb]], dtype=float)
    if order == "yx":
        return ry @ rx
    if order == "xy":
        return rx @ ry
    raise ValueError("holder order must be 'yx' or 'xy'")


def solve_holder_angles(v_zero: Array, order: str = "xy") -> tuple[float, float] | None:
    v = normalize(v_zero)
    if order == "yx":
        beta = math.asin(float(np.clip(-v[0], -1.0, 1.0)))
        cb = math.cos(beta)
        if abs(cb) < 1e-10:
            return None
        alpha = math.atan2(float(v[1]), float(v[2]))
    elif order == "xy":
        alpha = math.asin(float(np.clip(v[1], -1.0, 1.0)))
        ca = math.cos(alpha)
        if abs(ca) < 1e-10:
            return None
        beta = math.atan2(float(-v[0]), float(v[2]))
    else:
        raise ValueError("holder order must be 'yx' or 'xy'")
    return math.degrees(alpha), math.degrees(beta)


def orientation_from_match(
    match: PatternMatch,
    alpha_deg: float,
    beta_deg: float,
    image_to_holder_rotation_deg: float,
    holder_order: str,
) -> tuple[Array, Array, Array]:
    """Return (crystal_to_zero_holder, image_axes_in_crystal, holder_axes_in_crystal)."""
    zone_c = normalize(np.asarray(match.zone, dtype=float))
    q = match.linear_unit

    image_x_c = normalize(q[0, 0] * match.basis_x + q[0, 1] * match.basis_y)
    image_y_c = normalize(q[1, 0] * match.basis_x + q[1, 1] * match.basis_y)
    if np.dot(np.cross(image_x_c, image_y_c), zone_c) < 0:
        image_y_c = -image_y_c

    phi = math.radians(image_to_holder_rotation_deg)
    holder_x_c = normalize(math.cos(phi) * image_x_c + math.sin(phi) * image_y_c)
    holder_y_c = normalize(-math.sin(phi) * image_x_c + math.cos(phi) * image_y_c)
    if np.dot(np.cross(holder_x_c, holder_y_c), zone_c) < 0:
        holder_y_c = -holder_y_c

    lab_axes_in_crystal_current = np.column_stack([holder_x_c, holder_y_c, zone_c])
    crystal_to_lab_current = lab_axes_in_crystal_current.T
    r_current = holder_rotation(alpha_deg, beta_deg, holder_order)
    crystal_to_zero_holder = r_current.T @ crystal_to_lab_current
    image_axes = np.column_stack([image_x_c, image_y_c, zone_c])
    holder_axes = np.column_stack([holder_x_c, holder_y_c, zone_c])
    return crystal_to_zero_holder, image_axes, holder_axes


def target_rows(
    match: PatternMatch,
    crystal_to_zero_holder: Array,
    alpha_deg: float,
    beta_deg: float,
    target_families: Sequence[str],
    include_opposites: bool,
    holder_order: str,
    alpha_limits: tuple[float, float],
    beta_limits: tuple[float, float],
) -> list[dict[str, object]]:
    current = normalize(np.asarray(match.zone, dtype=float))
    rows: list[dict[str, object]] = []
    for family in target_families:
        for line_direction in family_directions(family, include_opposites=include_opposites):
            target_indices = np.asarray(line_direction, dtype=int)
            target = normalize(target_indices.astype(float))
            if not include_opposites and float(np.dot(target, current)) < 0:
                target = -target
                target_indices = -target_indices
            solved = solve_holder_angles(crystal_to_zero_holder @ target, holder_order)
            if solved is None:
                continue
            target_alpha, target_beta = solved
            dot = float(np.clip(np.dot(current, target), -1.0, 1.0))
            angle = math.degrees(math.acos(dot))
            reachable = (
                alpha_limits[0] <= target_alpha <= alpha_limits[1]
                and beta_limits[0] <= target_beta <= beta_limits[1]
            )
            rows.append(
                {
                    "family": f"<{family}>",
                    "zone": format_miller(tuple(int(x) for x in target_indices), "[]"),
                    "angle_from_current_deg": angle,
                    "alpha_deg": target_alpha,
                    "beta_deg": target_beta,
                    "delta_alpha_deg": target_alpha - alpha_deg,
                    "delta_beta_deg": target_beta - beta_deg,
                    "within_limits": "yes" if reachable else "no",
                }
            )

    rows.sort(
        key=lambda r: (
            0 if r["within_limits"] == "yes" else 1,
            str(r["family"]),
            float(r["angle_from_current_deg"]),
            abs(float(r["delta_alpha_deg"])) + abs(float(r["delta_beta_deg"])),
        )
    )
    return rows


def rotate_zero_holder_in_plane(v_zero: Array, ccw_deg: float) -> Array:
    """Rotate a zero-holder vector by sample loading rotation about -Z.

    Positive ccw_deg follows the schematic convention: counterclockwise when
    viewed along the holder -Z axis.
    """
    theta = math.radians(-ccw_deg)
    c, s = math.cos(theta), math.sin(theta)
    rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    return rz @ v_zero


def holder_angles_after_sample_rotation(
    v_zero: Array,
    ccw_deg: float,
    holder_order: str,
) -> tuple[float, float] | None:
    return solve_holder_angles(rotate_zero_holder_in_plane(v_zero, ccw_deg), holder_order)


def sample_rotation_trace_points(
    v_zero: Array,
    ccw_min_deg: float,
    ccw_max_deg: float,
    holder_order: str,
    max_points: int = 72,
) -> list[tuple[float, float, float]]:
    span = max(0.0, ccw_max_deg - ccw_min_deg)
    count = max(8, min(max_points, int(math.ceil(span / 2.0)) + 1))
    trace: list[tuple[float, float, float]] = []
    for ccw_deg in np.linspace(ccw_min_deg, ccw_max_deg, count):
        solved = holder_angles_after_sample_rotation(v_zero, float(ccw_deg), holder_order)
        if solved is None:
            continue
        alpha_deg, beta_deg = solved
        trace.append((float(ccw_deg), alpha_deg, beta_deg))
    return trace


def in_limits(
    alpha_deg: float,
    beta_deg: float,
    alpha_limits: tuple[float, float],
    beta_limits: tuple[float, float],
) -> bool:
    return alpha_limits[0] <= alpha_deg <= alpha_limits[1] and beta_limits[0] <= beta_deg <= beta_limits[1]


def sample_rotation_ranges_for_target(
    v_zero: Array,
    holder_order: str,
    alpha_limits: tuple[float, float],
    beta_limits: tuple[float, float],
    step_deg: float = 0.5,
) -> list[tuple[float, float]]:
    def inside(ccw_deg: float) -> bool:
        solved = holder_angles_after_sample_rotation(v_zero, ccw_deg, holder_order)
        if solved is None:
            return False
        alpha_deg, beta_deg = solved
        return in_limits(alpha_deg, beta_deg, alpha_limits, beta_limits)

    sample_count = max(1, int(round(360.0 / step_deg)))
    angles = np.linspace(-180.0, 180.0, sample_count + 1)
    flags = sample_rotation_limit_mask(v_zero, angles, holder_order, alpha_limits, beta_limits).tolist()
    ranges: list[tuple[float, float]] = []
    idx = 0
    while idx < len(angles):
        if not flags[idx]:
            idx += 1
            continue
        start_idx = idx
        while idx + 1 < len(angles) and flags[idx + 1]:
            idx += 1
        end_idx = idx

        if start_idx == 0:
            start = float(angles[start_idx])
        else:
            start = refine_rotation_boundary(float(angles[start_idx - 1]), float(angles[start_idx]), inside)
        if end_idx == len(angles) - 1:
            end = float(angles[end_idx])
        else:
            end = refine_rotation_boundary(float(angles[end_idx + 1]), float(angles[end_idx]), inside)

        if end >= start:
            ranges.append((start, end))
        idx += 1

    return ranges


def sample_rotation_limit_mask(
    v_zero: Array,
    ccw_angles_deg: Array,
    holder_order: str,
    alpha_limits: tuple[float, float],
    beta_limits: tuple[float, float],
) -> Array:
    v = normalize(v_zero)
    theta = np.radians(-ccw_angles_deg)
    c = np.cos(theta)
    s = np.sin(theta)
    x = c * v[0] - s * v[1]
    y = s * v[0] + c * v[1]
    z = np.full_like(x, v[2], dtype=float)
    if holder_order == "yx":
        beta = np.degrees(np.arcsin(np.clip(-x, -1.0, 1.0)))
        alpha = np.degrees(np.arctan2(y, z))
    elif holder_order == "xy":
        alpha = np.degrees(np.arcsin(np.clip(y, -1.0, 1.0)))
        beta = np.degrees(np.arctan2(-x, z))
    else:
        raise ValueError("holder order must be 'yx' or 'xy'")
    return (
        (alpha >= alpha_limits[0])
        & (alpha <= alpha_limits[1])
        & (beta >= beta_limits[0])
        & (beta <= beta_limits[1])
    )


def refine_rotation_boundary(
    outside_deg: float,
    inside_deg: float,
    inside_fn: Callable[[float], bool],
    iterations: int = 18,
) -> float:
    outside = outside_deg
    inside = inside_deg
    for _ in range(iterations):
        mid = 0.5 * (outside + inside)
        if inside_fn(mid):
            inside = mid
        else:
            outside = mid
    return inside


def in_plane_rotation_rows(
    match: PatternMatch,
    crystal_to_zero_holder: Array,
    target_families: Sequence[str],
    include_opposites: bool,
    holder_order: str,
    alpha_limits: tuple[float, float],
    beta_limits: tuple[float, float],
) -> list[dict[str, object]]:
    current = normalize(np.asarray(match.zone, dtype=float))
    rows: list[dict[str, object]] = []
    for family in target_families:
        for line_direction in family_directions(family, include_opposites=include_opposites):
            target_indices = np.asarray(line_direction, dtype=int)
            target = normalize(target_indices.astype(float))
            if not include_opposites and float(np.dot(target, current)) < 0:
                target = -target
                target_indices = -target_indices
            v_zero = crystal_to_zero_holder @ target
            solved = solve_holder_angles(v_zero, holder_order)
            if solved is None:
                continue
            target_alpha, target_beta = solved
            if in_limits(target_alpha, target_beta, alpha_limits, beta_limits):
                continue
            ranges = sample_rotation_ranges_for_target(v_zero, holder_order, alpha_limits, beta_limits)
            if not ranges:
                continue
            dot = float(np.clip(np.dot(current, target), -1.0, 1.0))
            angle = math.degrees(math.acos(dot))
            for rot_min, rot_max in ranges:
                min_angles = holder_angles_after_sample_rotation(v_zero, rot_min, holder_order)
                max_angles = holder_angles_after_sample_rotation(v_zero, rot_max, holder_order)
                if min_angles is None or max_angles is None:
                    continue
                alpha_ccw_min, beta_ccw_min = min_angles
                alpha_ccw_max, beta_ccw_max = max_angles
                zone_indices = tuple(int(x) for x in target_indices)
                rows.append(
                    {
                        "ok": "yes",
                        "family": f"<{family}>",
                        "zone": format_miller(zone_indices, "[]"),
                        "zone_indices": zone_indices,
                        "angle_from_current_deg": angle,
                        "alpha_deg": target_alpha,
                        "beta_deg": target_beta,
                        "rotation_min_deg": rot_min,
                        "alpha_ccw_min_deg": alpha_ccw_min,
                        "beta_ccw_min_deg": beta_ccw_min,
                        "rotation_max_deg": rot_max,
                        "alpha_ccw_max_deg": alpha_ccw_max,
                        "beta_ccw_max_deg": beta_ccw_max,
                        "trace_points": sample_rotation_trace_points(v_zero, rot_min, rot_max, holder_order),
                    }
                )

    rows.sort(
        key=lambda r: (
            str(r["family"]),
            float(r["angle_from_current_deg"]),
            abs(float(r["alpha_deg"])) + abs(float(r["beta_deg"])),
            float(r["rotation_min_deg"]),
        )
    )
    return rows


def sample_rotation_reachable_rows(
    match: PatternMatch,
    crystal_to_zero_holder: Array,
    ccw_deg: float,
    target_families: Sequence[str],
    include_opposites: bool,
    holder_order: str,
    alpha_limits: tuple[float, float],
    beta_limits: tuple[float, float],
) -> list[dict[str, object]]:
    current = normalize(np.asarray(match.zone, dtype=float))
    rows: list[dict[str, object]] = []
    seen_zones: set[tuple[int, int, int]] = set()
    for family in target_families:
        for line_direction in family_directions(family, include_opposites=include_opposites):
            target_indices = np.asarray(line_direction, dtype=int)
            target = normalize(target_indices.astype(float))
            if not include_opposites and float(np.dot(target, current)) < 0:
                target = -target
                target_indices = -target_indices
            zone_indices = tuple(int(x) for x in target_indices)
            if zone_indices in seen_zones:
                continue
            seen_zones.add(zone_indices)
            v_zero = crystal_to_zero_holder @ target
            solved = holder_angles_after_sample_rotation(v_zero, ccw_deg, holder_order)
            if solved is None:
                continue
            alpha_deg, beta_deg = solved
            if not in_limits(alpha_deg, beta_deg, alpha_limits, beta_limits):
                continue
            dot = float(np.clip(np.dot(current, target), -1.0, 1.0))
            rows.append(
                {
                    "family": f"<{family}>",
                    "zone": format_miller(zone_indices, "[]"),
                    "zone_indices": zone_indices,
                    "angle_from_current_deg": math.degrees(math.acos(dot)),
                    "alpha_deg": alpha_deg,
                    "beta_deg": beta_deg,
                }
            )

    rows.sort(
        key=lambda r: (
            str(r["family"]),
            float(r["angle_from_current_deg"]),
            abs(float(r["alpha_deg"])) + abs(float(r["beta_deg"])),
            str(r["zone"]),
        )
    )
    return rows


def sample_rotation_map_points(
    match: PatternMatch,
    crystal_to_zero_holder: Array,
    ccw_deg: float,
    target_families: Sequence[str],
    include_opposites: bool,
    holder_order: str,
    alpha_limits: tuple[float, float],
    beta_limits: tuple[float, float],
) -> list[ZoneAxisMapPoint]:
    current = normalize(np.asarray(match.zone, dtype=float))
    points: list[ZoneAxisMapPoint] = []
    seen_zones: set[tuple[int, int, int]] = set()
    for family in target_families:
        for line_direction in family_directions(family, include_opposites=include_opposites):
            target_indices = np.asarray(line_direction, dtype=int)
            target = normalize(target_indices.astype(float))
            if not include_opposites and float(np.dot(target, current)) < 0:
                target = -target
                target_indices = -target_indices
            zone_indices = tuple(int(x) for x in target_indices)
            if zone_indices in seen_zones:
                continue
            seen_zones.add(zone_indices)
            solved = holder_angles_after_sample_rotation(crystal_to_zero_holder @ target, ccw_deg, holder_order)
            if solved is None:
                continue
            alpha_deg, beta_deg = solved
            points.append(
                ZoneAxisMapPoint(
                    family=family,
                    zone=zone_indices,
                    alpha_deg=alpha_deg,
                    beta_deg=beta_deg,
                    within_limits=in_limits(alpha_deg, beta_deg, alpha_limits, beta_limits),
                )
            )

    points.sort(
        key=lambda p: (
            p.family,
            0 if p.within_limits else 1,
            abs(p.alpha_deg) + abs(p.beta_deg),
            p.zone,
        )
    )
    return points


def zone_axis_map_points(
    match: PatternMatch,
    crystal_to_zero_holder: Array,
    alpha_deg: float,
    beta_deg: float,
    map_families: Sequence[str],
    holder_order: str,
    alpha_limits: tuple[float, float],
    beta_limits: tuple[float, float],
) -> list[ZoneAxisMapPoint]:
    current_indices = tuple(int(x) for x in match.zone)
    current = normalize(np.asarray(current_indices, dtype=float))
    points = [
        ZoneAxisMapPoint(
            family=family_name(current_indices),
            zone=current_indices,
            alpha_deg=alpha_deg,
            beta_deg=beta_deg,
            is_current=True,
            within_limits=(
                alpha_limits[0] <= alpha_deg <= alpha_limits[1]
                and beta_limits[0] <= beta_deg <= beta_limits[1]
            ),
        )
    ]

    seen_zones = {current_indices}
    for family in map_families:
        for line_direction in family_directions(family, include_opposites=False):
            target_indices = np.asarray(line_direction, dtype=int)
            target = normalize(target_indices.astype(float))
            if float(np.dot(target, current)) < 0:
                target = -target
                target_indices = -target_indices
            target_tuple = tuple(int(x) for x in target_indices)
            if target_tuple in seen_zones:
                continue
            solved = solve_holder_angles(crystal_to_zero_holder @ target, holder_order)
            if solved is None:
                continue
            target_alpha, target_beta = solved
            seen_zones.add(target_tuple)
            points.append(
                ZoneAxisMapPoint(
                    family=family,
                    zone=target_tuple,
                    alpha_deg=target_alpha,
                    beta_deg=target_beta,
                    within_limits=(
                        alpha_limits[0] <= target_alpha <= alpha_limits[1]
                        and beta_limits[0] <= target_beta <= beta_limits[1]
                    ),
                )
            )

    return points


def zone_axis_reachable_with_sample_rotation(
    zone: Sequence[int],
    crystal_to_zero_holder: Array,
    holder_order: str,
    alpha_limits: tuple[float, float],
    beta_limits: tuple[float, float],
) -> bool:
    v_zero = crystal_to_zero_holder @ normalize(np.asarray(zone, dtype=float))
    return bool(sample_rotation_ranges_for_target(v_zero, holder_order, alpha_limits, beta_limits))


def filter_reachable_zone_axis_map_points(
    points: Sequence[ZoneAxisMapPoint],
    crystal_to_zero_holder: Array,
    holder_order: str,
    alpha_limits: tuple[float, float],
    beta_limits: tuple[float, float],
) -> list[ZoneAxisMapPoint]:
    reachable: list[ZoneAxisMapPoint] = []
    reachable_cache: dict[tuple[int, int, int], bool] = {}
    for point in points:
        if point.is_current or point.within_limits:
            reachable.append(point)
            continue
        if point.zone not in reachable_cache:
            reachable_cache[point.zone] = zone_axis_reachable_with_sample_rotation(
                point.zone,
                crystal_to_zero_holder,
                holder_order,
                alpha_limits,
                beta_limits,
            )
        if reachable_cache[point.zone]:
            reachable.append(point)
    return reachable


def write_targets_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    fields = [
        "family",
        "zone",
        "angle_from_current_deg",
        "alpha_deg",
        "beta_deg",
        "delta_alpha_deg",
        "delta_beta_deg",
        "within_limits",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_indexed_spots_csv(path: Path, match: PatternMatch, peaks: Sequence[Peak]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["h", "k", "l", "observed_x_px", "observed_y_px", "predicted_x_px", "predicted_y_px"])
        model_xy = np.asarray([r.xy for r in match.reflections], dtype=float)
        pred = apply_transform(model_xy, match.scale * match.linear_unit, match.translation)
        for ref_idx, obs_idx in sorted(match.matched_indices.items()):
            r = match.reflections[ref_idx]
            peak = peaks[obs_idx]
            writer.writerow([r.h, r.k, r.l, peak.x, peak.y, pred[ref_idx, 0], -pred[ref_idx, 1]])


def draw_circle(draw: ImageDraw.ImageDraw, x: float, y: float, radius: float, color: tuple[int, int, int], width: int) -> None:
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=width)


def load_label_font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font, stroke_width=1)
    return int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1])


def draw_miller_label(
    draw: ImageDraw.ImageDraw,
    position: tuple[float, float],
    indices: Sequence[int],
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int] = (245, 245, 245),
) -> None:
    tokens = [(str(abs(int(value))), int(value) < 0) for value in indices]
    sizes = [text_size(draw, token, font) for token, _negative in tokens]
    total_width = sum(width for width, _height in sizes)
    max_height = max((height for _width, height in sizes), default=0)
    x = float(position[0]) - total_width / 2.0
    y = float(position[1]) - max_height / 2.0
    bar_y = y - max(2, max_height // 8)
    bar_width = max(1, max_height // 10)

    for (token, negative), (width, _height) in zip(tokens, sizes):
        draw.text((x, y), token, font=font, fill=fill, stroke_width=1, stroke_fill=(0, 0, 0))
        if negative:
            draw.line((x + 1, bar_y, x + width - 1, bar_y), fill=fill, width=bar_width)
        x += width


def transformed_pattern_points(
    match: PatternMatch,
    image_size: tuple[int, int],
    rotated: bool,
) -> Array:
    model_xy = np.asarray([r.xy for r in match.reflections], dtype=float)
    if rotated:
        linear = match.scale * match.linear_unit
        translation = match.translation
    else:
        linear = match.scale * np.eye(2)
        translation = np.array([image_size[0] / 2.0, -image_size[1] / 2.0], dtype=float)
    return apply_transform(model_xy, linear, translation)


def draw_predicted_spots_and_labels(
    draw: ImageDraw.ImageDraw,
    match: PatternMatch,
    pred_screen: Array,
    image_size: tuple[int, int],
    show_labels: bool,
    spot_radius: int | None = None,
    spot_fill: tuple[int, int, int] = (245, 145, 190),
    spot_outline: tuple[int, int, int] = (255, 230, 245),
    label_fill: tuple[int, int, int] = (245, 245, 245),
) -> None:
    min_dim = min(image_size)
    if spot_radius is None:
        spot_radius = max(5, int(min_dim / 210))
    font = load_label_font(max(24, int(min_dim / 48)))
    label_offset = max(18, int(min_dim / 55))
    vis = visible_mask(pred_screen, image_size[0], image_size[1], margin=60.0)

    for idx in np.flatnonzero(vis):
        x = float(pred_screen[int(idx), 0])
        y = float(-pred_screen[int(idx), 1])
        draw.ellipse(
            (x - spot_radius, y - spot_radius, x + spot_radius, y + spot_radius),
            fill=spot_fill,
            outline=spot_outline,
            width=max(1, spot_radius // 3),
        )

    if not show_labels:
        return

    for idx in np.flatnonzero(vis):
        r = match.reflections[int(idx)]
        x = float(pred_screen[int(idx), 0])
        y = float(-pred_screen[int(idx), 1]) - label_offset
        draw_miller_label(draw, (x, y), (r.h, r.k, r.l), font, fill=label_fill)


def predicted_pattern_image(
    match: PatternMatch,
    image_size: tuple[int, int],
    rotated: bool,
    title: str | None = None,
    show_labels: bool = True,
    draw_guides: bool = False,
    kikuchi_max_g_norm: float = 0.0,
) -> Image.Image:
    image = Image.new("RGB", image_size, (0, 0, 0))
    draw = ImageDraw.Draw(image)
    min_dim = min(image_size)
    title_font = load_label_font(max(20, int(min_dim / 68)))

    pred = transformed_pattern_points(match, image_size, rotated=rotated)
    if draw_guides:
        draw_kikuchi_guides(
            draw,
            match,
            image_size,
            max_g_norm=kikuchi_max_g_norm,
            width=max(1, int(min_dim / 850)),
            pred_screen=pred,
        )
    draw_predicted_spots_and_labels(draw, match, pred, image_size, show_labels=show_labels)

    if title:
        draw.text((14, 12), title, font=title_font, fill=(235, 235, 235))
    return image


def fitted_diffraction_image(
    image: Image.Image,
    match: PatternMatch,
    show_labels: bool = True,
    draw_guides: bool = True,
    kikuchi_max_g_norm: float = 0.0,
    title: str | None = None,
) -> Image.Image:
    out = image.copy().convert("RGB")
    draw = ImageDraw.Draw(out)
    min_dim = min(out.size)
    title_font = load_label_font(max(20, int(min_dim / 68)))
    pred = transformed_pattern_points(match, out.size, rotated=True)
    if draw_guides:
        draw_kikuchi_guides(
            draw,
            match,
            out.size,
            max_g_norm=kikuchi_max_g_norm,
            width=max(1, int(min_dim / 700)),
            pred_screen=pred,
        )
    draw_predicted_spots_and_labels(
        draw,
        match,
        pred,
        out.size,
        show_labels=show_labels,
        spot_fill=(255, 120, 190),
        spot_outline=(255, 245, 255),
        label_fill=(255, 120, 190),
    )
    if title:
        draw.rectangle((8, 8, 24 + 11 * len(title), 42), fill=(0, 0, 0))
        draw.text((16, 14), title, font=title_font, fill=(255, 255, 255))
    return out


def draw_dashed_line(
    draw: ImageDraw.ImageDraw,
    start: tuple[float, float],
    end: tuple[float, float],
    fill: tuple[int, int, int],
    width: int = 1,
    dash: int = 8,
    gap: int = 5,
) -> None:
    x1, y1 = start
    x2, y2 = end
    length = math.hypot(x2 - x1, y2 - y1)
    if length <= 0:
        return
    ux = (x2 - x1) / length
    uy = (y2 - y1) / length
    distance = 0.0
    while distance < length:
        segment_end = min(distance + dash, length)
        draw.line(
            (
                x1 + ux * distance,
                y1 + uy * distance,
                x1 + ux * segment_end,
                y1 + uy * segment_end,
            ),
            fill=fill,
            width=width,
        )
        distance += dash + gap


def draw_dashed_polyline(
    draw: ImageDraw.ImageDraw,
    points: Sequence[tuple[float, float]],
    fill: tuple[int, int, int],
    width: int = 1,
    dash: int = 8,
    gap: int = 5,
) -> None:
    if len(points) < 2:
        return
    draw_segment = True
    remaining_pattern = float(dash)
    for start, end in zip(points, points[1:]):
        x1, y1 = start
        x2, y2 = end
        length = math.hypot(x2 - x1, y2 - y1)
        if length <= 0:
            continue
        ux = (x2 - x1) / length
        uy = (y2 - y1) / length
        distance = 0.0
        while distance < length:
            step = min(remaining_pattern, length - distance)
            sx = x1 + ux * distance
            sy = y1 + uy * distance
            ex = x1 + ux * (distance + step)
            ey = y1 + uy * (distance + step)
            if draw_segment:
                draw.line((sx, sy, ex, ey), fill=fill, width=width)
            distance += step
            remaining_pattern -= step
            if remaining_pattern <= 1e-9:
                draw_segment = not draw_segment
                remaining_pattern = float(dash if draw_segment else gap)


def padded_range(values: Sequence[float], minimum_span: float = 10.0) -> tuple[float, float]:
    low = min(values)
    high = max(values)
    span = high - low
    if span < minimum_span:
        center = 0.5 * (low + high)
        low = center - minimum_span / 2.0
        high = center + minimum_span / 2.0
        span = minimum_span
    pad = max(4.0, span * 0.08)
    return low - pad, high + pad


def draw_star_marker(
    draw: ImageDraw.ImageDraw,
    center: tuple[float, float],
    radius: float,
    fill: tuple[int, int, int],
    outline: tuple[int, int, int] = (0, 0, 0),
    outline_width: int = 2,
) -> None:
    cx, cy = center
    inner = radius * 0.45
    points: list[tuple[float, float]] = []
    for idx in range(10):
        angle = -math.pi / 2.0 + idx * math.pi / 5.0
        r = radius if idx % 2 == 0 else inner
        points.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
    draw.polygon(points, fill=fill)
    draw.line([*points, points[0]], fill=outline, width=outline_width)


def zone_axis_map_image(
    points: Sequence[ZoneAxisMapPoint],
    alpha_limits: tuple[float, float],
    beta_limits: tuple[float, float],
    rotation_rows: Sequence[dict[str, object]] | None = None,
    map_label_individual_color: bool = True,
    reverse_alpha_axis: bool = False,
    reverse_beta_axis: bool = False,
    compact_axes: bool = False,
    alpha_range: tuple[float, float] | None = None,
    beta_range: tuple[float, float] | None = None,
    image_size: tuple[int, int] = (1100, 1100),
    title: str = "Zone Axis Map",
) -> Image.Image:
    width, height = image_size
    image = Image.new("RGB", image_size, (255, 255, 255))
    draw = ImageDraw.Draw(image)
    min_dim = min(image_size)
    title_font = load_label_font(max(18, int(min_dim / 38)))
    axis_font = load_label_font(max(16, int(min_dim / 44)))
    tick_font = load_label_font(max(14, int(min_dim / 50)))
    label_font = load_label_font(max(12, int(min_dim / (64 if compact_axes else 56))))

    if compact_axes:
        left = max(52, int(width * 0.055))
        right = width - max(26, int(width * 0.025))
        top = max(36, int(height * 0.045))
        bottom = height - max(52, int(height * 0.065))
    else:
        left = max(72, int(width * 0.08))
        right = width - max(42, int(width * 0.04))
        top = max(58, int(height * 0.08))
        bottom = height - max(72, int(height * 0.10))

    def row_zone_indices(row: dict[str, object]) -> tuple[int, int, int]:
        indices = row.get("zone_indices")
        if indices is not None:
            return tuple(int(x) for x in indices)  # type: ignore[arg-type]
        return parse_miller(str(row["zone"]))

    rotation_endpoints: list[tuple[float, float, str, tuple[int, int, int], str, float]] = []
    rotation_traces: list[tuple[list[tuple[float, float]], tuple[int, int, int]]] = []
    for row in rotation_rows or []:
        zone = str(row["zone"])
        zone_indices = row_zone_indices(row)
        trace_points: list[tuple[float, float]] = []
        for trace_item in row.get("trace_points", []):  # type: ignore[union-attr]
            _ccw_deg, alpha_deg, beta_deg = trace_item
            trace_points.append((float(alpha_deg), float(beta_deg)))
        if len(trace_points) > 1:
            rotation_traces.append((trace_points, zone_indices))
        rotation_endpoints.extend(
            [
                (
                    float(row["alpha_ccw_min_deg"]),
                    float(row["beta_ccw_min_deg"]),
                    str(row["family"]).strip("<>"),
                    zone_indices,
                    f"{zone} min={float(row['rotation_min_deg']):.1f}",
                    float(row["rotation_min_deg"]),
                ),
                (
                    float(row["alpha_ccw_max_deg"]),
                    float(row["beta_ccw_max_deg"]),
                    str(row["family"]).strip("<>"),
                    zone_indices,
                    f"{zone} max={float(row['rotation_max_deg']):.1f}",
                    float(row["rotation_max_deg"]),
                ),
            ]
        )

    trace_alpha_values = [alpha for trace, _zone in rotation_traces for alpha, _beta in trace]
    trace_beta_values = [beta for trace, _zone in rotation_traces for _alpha, beta in trace]
    alpha_values = (
        [p.alpha_deg for p in points]
        + [p[0] for p in rotation_endpoints]
        + trace_alpha_values
        + [alpha_limits[0], alpha_limits[1], 0.0]
    )
    beta_values = (
        [p.beta_deg for p in points]
        + [p[1] for p in rotation_endpoints]
        + trace_beta_values
        + [beta_limits[0], beta_limits[1], 0.0]
    )
    if alpha_range is None:
        alpha_min, alpha_max = padded_range(alpha_values)
    else:
        alpha_min, alpha_max = alpha_range
    if beta_range is None:
        beta_min, beta_max = padded_range(beta_values)
    else:
        beta_min, beta_max = beta_range

    def map_x(alpha: float) -> float:
        fraction = (alpha - alpha_min) / (alpha_max - alpha_min)
        if reverse_alpha_axis:
            return right - fraction * (right - left)
        return left + fraction * (right - left)

    def map_y(beta: float) -> float:
        fraction = (beta - beta_min) / (beta_max - beta_min)
        if reverse_beta_axis:
            return top + fraction * (bottom - top)
        return bottom - fraction * (bottom - top)

    axis_color = (25, 25, 25)
    limit_color = (90, 90, 90)
    draw.line((left, bottom, right, bottom), fill=axis_color, width=2)
    draw.line((left, top, left, bottom), fill=axis_color, width=2)
    arrow_len = max(12, int(min_dim / 45))
    arrow_half_width = max(5, int(min_dim / 90))
    if reverse_alpha_axis:
        draw.polygon(
            (
                (left, bottom),
                (left + arrow_len, bottom - arrow_half_width),
                (left + arrow_len, bottom + arrow_half_width),
            ),
            fill=axis_color,
        )
    else:
        draw.polygon(
            (
                (right, bottom),
                (right - arrow_len, bottom - arrow_half_width),
                (right - arrow_len, bottom + arrow_half_width),
            ),
            fill=axis_color,
        )
    if reverse_beta_axis:
        draw.polygon(
            (
                (left, bottom),
                (left - arrow_half_width, bottom - arrow_len),
                (left + arrow_half_width, bottom - arrow_len),
            ),
            fill=axis_color,
        )
    else:
        draw.polygon(
            (
                (left, top),
                (left - arrow_half_width, top + arrow_len),
                (left + arrow_half_width, top + arrow_len),
            ),
            fill=axis_color,
        )
    if title:
        title_w, _title_h = text_size(draw, title, title_font)
        draw.text(((left + right - title_w) / 2.0, 16), title, font=title_font, fill=axis_color)

    x_title = "Alpha (deg)"
    y_title = "Beta (deg)"
    x_title_w, x_title_h = text_size(draw, x_title, axis_font)
    _, y_title_h = text_size(draw, y_title, axis_font)
    x_title_y = height - x_title_h - (8 if compact_axes else 12)
    y_title_y = top - y_title_h - (5 if compact_axes else 10)
    draw.text(((left + right - x_title_w) / 2.0, x_title_y), x_title, font=axis_font, fill=axis_color)
    draw.text((left, y_title_y), y_title, font=axis_font, fill=axis_color)

    for value in sorted({round(alpha_min), 0, round(alpha_max), round(alpha_limits[0]), round(alpha_limits[1])}):
        if alpha_min <= value <= alpha_max:
            x = map_x(float(value))
            draw.line((x, bottom - 5, x, bottom + 5), fill=axis_color, width=1)
            text = f"{value:g}"
            tw, _th = text_size(draw, text, tick_font)
            draw.text((x - tw / 2.0, bottom + 9), text, font=tick_font, fill=axis_color)

    for value in sorted({round(beta_min), 0, round(beta_max), round(beta_limits[0]), round(beta_limits[1])}):
        if beta_min <= value <= beta_max:
            y = map_y(float(value))
            draw.line((left - 5, y, left + 5, y), fill=axis_color, width=1)
            text = f"{value:g}"
            tw, th = text_size(draw, text, tick_font)
            draw.text((left - tw - 10, y - th / 2.0), text, font=tick_font, fill=axis_color)

    x1, x2 = map_x(alpha_limits[0]), map_x(alpha_limits[1])
    y1, y2 = map_y(beta_limits[0]), map_y(beta_limits[1])
    dash_width = max(1, int(min_dim / 420))
    draw_dashed_line(draw, (x1, y1), (x2, y1), fill=limit_color, width=dash_width)
    draw_dashed_line(draw, (x2, y1), (x2, y2), fill=limit_color, width=dash_width)
    draw_dashed_line(draw, (x2, y2), (x1, y2), fill=limit_color, width=dash_width)
    draw_dashed_line(draw, (x1, y2), (x1, y1), fill=limit_color, width=dash_width)

    trace_width = max(2, int(min_dim / 360))
    trace_dash = max(8, int(min_dim / 85))
    trace_gap = max(7, int(min_dim / 95))
    for trace, zone_indices in rotation_traces:
        outline = zone_outline_color(zone_indices)
        screen_trace = [(map_x(alpha), map_y(beta)) for alpha, beta in trace]
        draw_dashed_polyline(draw, screen_trace, fill=outline, width=trace_width, dash=trace_dash, gap=trace_gap)

    for point in points:
        x = map_x(point.alpha_deg)
        y = map_y(point.beta_deg)
        label = format_miller(point.zone)
        if point.is_current:
            radius = max(8, int(min_dim / 70))
            draw.line((x - radius, y - radius, x + radius, y + radius), fill=(0, 0, 0), width=3)
            draw.line((x - radius, y + radius, x + radius, y - radius), fill=(0, 0, 0), width=3)
            fill = (0, 0, 0)
            label_fill = fill
            label_text = f"{label}"
        else:
            fill = ZONE_FAMILY_COLORS.get(point.family, (80, 80, 80))
            radius = max(5, int(min_dim / 105))
            outline = zone_outline_color(point.zone)
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill, outline=outline, width=3)
            label_fill = outline if map_label_individual_color else fill
            label_stroke = (255, 255, 255) if map_label_individual_color else outline
            label_text = label
        if point.is_current:
            label_stroke = (255, 255, 255)
        draw.text(
            (x + radius + 4, y - radius - 10),
            label_text,
            font=label_font,
            fill=label_fill,
            stroke_width=2,
            stroke_fill=label_stroke,
        )

    for alpha_deg, beta_deg, family, zone_indices, label_text, _rotation_deg in rotation_endpoints:
        x = map_x(alpha_deg)
        y = map_y(beta_deg)
        fill = ZONE_FAMILY_COLORS.get(family, (80, 80, 80))
        outline = zone_outline_color(zone_indices)
        radius = max(8, int(min_dim / 88))
        draw_star_marker(draw, (x, y), radius, fill=fill, outline=outline, outline_width=3)
        label_fill = outline if map_label_individual_color else fill
        label_stroke = (255, 255, 255) if map_label_individual_color else outline
        draw.text(
            (x + radius + 5, y - radius - 12),
            label_text,
            font=label_font,
            fill=label_fill,
            stroke_width=2,
            stroke_fill=label_stroke,
        )

    return image


def write_predicted_patterns(
    prefix: Path,
    match: PatternMatch,
    image: Image.Image,
    show_labels: bool = True,
    draw_guides: bool = True,
    kikuchi_max_g_norm: float = 0.0,
) -> tuple[Path, Path]:
    predicted_path = Path(f"{prefix}_predicted_zone_pattern.png")
    fitted_path = Path(f"{prefix}_fitted_diffraction_pattern.png")
    predicted_pattern_image(
        match,
        image.size,
        rotated=False,
        title=f"Predicted FCC {format_miller(match.zone)} before rotation",
        show_labels=show_labels,
        draw_guides=draw_guides,
        kikuchi_max_g_norm=kikuchi_max_g_norm,
    ).save(predicted_path)
    fitted_diffraction_image(
        image,
        match,
        show_labels=show_labels,
        draw_guides=draw_guides,
        kikuchi_max_g_norm=kikuchi_max_g_norm,
        title=f"Fitted FCC {format_miller(match.zone)}",
    ).save(fitted_path)
    return predicted_path, fitted_path


def line_box_intersections(
    point: tuple[float, float],
    direction: tuple[float, float],
    width: int,
    height: int,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    px, py = point
    dx, dy = direction
    if abs(dx) < 1e-12 and abs(dy) < 1e-12:
        return None

    hits: list[tuple[float, float, float]] = []
    if abs(dx) > 1e-12:
        for x in (0.0, float(width)):
            t = (x - px) / dx
            y = py + t * dy
            if 0.0 <= y <= height:
                hits.append((t, x, y))
    if abs(dy) > 1e-12:
        for y in (0.0, float(height)):
            t = (y - py) / dy
            x = px + t * dx
            if 0.0 <= x <= width:
                hits.append((t, x, y))

    if len(hits) < 2:
        return None
    hits.sort(key=lambda item: item[0])
    return (hits[0][1], hits[0][2]), (hits[-1][1], hits[-1][2])


def draw_kikuchi_guides(
    draw: ImageDraw.ImageDraw,
    match: PatternMatch,
    image_size: tuple[int, int],
    max_g_norm: float,
    width: int,
    pred_screen: Array | None = None,
) -> None:
    """Draw approximate Kikuchi line pairs for indexed low-order spots.

    In the small-angle SAED geometry used here, the pair associated with a
    reciprocal vector g is drawn perpendicular to the center-to-spot vector.
    The two guide lines pass through the midpoint between O and +g and the
    midpoint between O and -g.
    The physical band width still depends on voltage/lattice parameter, so
    these are navigation guides rather than full dynamical simulations.
    """
    if pred_screen is None:
        pred_screen = transformed_pattern_points(match, image_size, rotated=True)
    center_idx = next(
        (i for i, r in enumerate(match.reflections) if r.h == 0 and r.k == 0 and r.l == 0),
        0,
    )
    center_px = (float(pred_screen[center_idx, 0]), float(-pred_screen[center_idx, 1]))
    drawn: set[tuple[int, int, int]] = set()
    if max_g_norm <= 0:
        first_shell = min((r.g_norm for r in match.reflections if r.g_norm > 0), default=0.0)
        max_g_norm = first_shell * 1.05

    vis = visible_mask(pred_screen, image_size[0], image_size[1], margin=80.0)
    candidate_indices = sorted(int(i) for i in np.flatnonzero(vis))
    for ref_idx in candidate_indices:
        r = match.reflections[ref_idx]
        if r.g_norm == 0 or r.g_norm > max_g_norm:
            continue
        key = canonical_line((r.h, r.k, r.l))
        if key in drawn:
            continue
        drawn.add(key)

        spot_px = (float(pred_screen[ref_idx, 0]), float(-pred_screen[ref_idx, 1]))
        vx = spot_px[0] - center_px[0]
        vy = spot_px[1] - center_px[1]
        direction = (-vy, vx)
        for point in (
            (center_px[0] + 0.5 * vx, center_px[1] + 0.5 * vy),
            (center_px[0] - 0.5 * vx, center_px[1] - 0.5 * vy),
        ):
            segment = line_box_intersections(point, direction, image_size[0], image_size[1])
            if segment is not None:
                draw.line((segment[0][0], segment[0][1], segment[1][0], segment[1][1]), fill=(0, 180, 255), width=width)


def write_overlay(
    path: Path,
    image: Image.Image,
    peaks: Sequence[Peak],
    match: PatternMatch,
    draw_guides: bool = True,
    kikuchi_max_g_norm: float = 0.0,
    show_labels: bool = True,
) -> None:
    out = fitted_diffraction_image(
        image,
        match,
        show_labels=show_labels,
        draw_guides=draw_guides,
        kikuchi_max_g_norm=kikuchi_max_g_norm,
        title=(
            f"Fitted FCC {format_miller(match.zone)}, "
            f"rot {match.angle_deg:.2f} deg, "
            f"{match.matched_count}/{match.visible_count} spots"
        ),
    )
    out.save(path)


def print_match_summary(best: PatternMatch, all_results: Sequence[PatternMatch]) -> None:
    print("\nBest present-zone match")
    print("-----------------------")
    print(f"zone family       : <{best.family}>")
    print(f"indexed zone      : {format_miller(best.zone)}")
    print(f"in-plane rotation : {best.angle_deg:.3f} deg")
    print(f"spot match        : {best.matched_count}/{best.visible_count} visible spots")
    print(f"RMS fit error     : {best.rms_px:.2f} px (tolerance {best.tolerance_px:.2f} px)")
    print(f"pixel scale       : {best.scale:.3f} px per reciprocal-index unit")

    if len(all_results) > 1:
        print("\nAlternative reference scores")
        print("----------------------------")
        for result in all_results:
            print(
                f"<{result.family}> {format_miller(result.zone):>10s}  "
                f"{result.matched_count:2d}/{result.visible_count:<2d} spots  "
                f"rms {result.rms_px:7.2f} px  score {result.score:7.2f}"
            )


def print_target_table(rows: Sequence[dict[str, object]], limit: int = 40) -> None:
    print("\nPredicted target holder angles")
    print("------------------------------")
    header = f"{'ok':<3} {'family':<6} {'zone':<12} {'angle':>8} {'alpha':>9} {'beta':>9} {'d_alpha':>9} {'d_beta':>9}"
    print(header)
    print("-" * len(header))
    for row in rows[:limit]:
        ok = "yes" if row["within_limits"] == "yes" else "no"
        print(
            f"{ok:<3} {row['family']:<6} {row['zone']:<12} "
            f"{float(row['angle_from_current_deg']):8.3f} "
            f"{float(row['alpha_deg']):9.3f} "
            f"{float(row['beta_deg']):9.3f} "
            f"{float(row['delta_alpha_deg']):9.3f} "
            f"{float(row['delta_beta_deg']):9.3f}"
        )


def print_in_plane_rotation_table(rows: Sequence[dict[str, object]], limit: int = 40) -> None:
    if not rows:
        return
    print("\nOut-of-limit targets reachable by sample in-plane rotation")
    print("----------------------------------------------------------")
    header = (
        f"{'ok':<3} {'family':<6} {'zone':<12} {'angle':>8} "
        f"{'alpha':>9} {'beta':>9} {'ccw_min':>9} "
        f"{'alpha_ccw_min':>13} {'beta_ccw_min':>12} {'ccw_max':>9} "
        f"{'alpha_ccw_max':>13} {'beta_ccw_max':>12}"
    )
    print(header)
    print("-" * len(header))
    for row in rows[:limit]:
        print(
            f"{row['ok']:<3} {row['family']:<6} {row['zone']:<12} "
            f"{float(row['angle_from_current_deg']):8.3f} "
            f"{float(row['alpha_deg']):9.3f} "
            f"{float(row['beta_deg']):9.3f} "
            f"{float(row['rotation_min_deg']):9.2f} "
            f"{float(row['alpha_ccw_min_deg']):13.3f} "
            f"{float(row['beta_ccw_min_deg']):12.3f} "
            f"{float(row['rotation_max_deg']):9.2f} "
            f"{float(row['alpha_ccw_max_deg']):13.3f} "
            f"{float(row['beta_ccw_max_deg']):12.3f}"
        )


def parse_limits(values: Sequence[float], name: str) -> tuple[float, float]:
    if len(values) != 2:
        raise argparse.ArgumentTypeError(f"{name} needs two values: min max")
    low, high = float(values[0]), float(values[1])
    if low > high:
        low, high = high, low
    return low, high


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Index an FCC zone-axis diffraction image and predict reachable double-tilt holder angles.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("image", type=Path, help="Experimental diffraction image.")
    parser.add_argument("--alpha", type=float, required=True, help="Current alpha tilt angle in degrees.")
    parser.add_argument("--beta", type=float, required=True, help="Current beta tilt angle in degrees.")
    parser.add_argument(
        "--current-zone",
        type=parse_miller,
        default=None,
        help="Optional known current zone, e.g. '1,0,0'. If omitted, the script chooses among supported FCC zones.",
    )
    parser.add_argument(
        "--target-families",
        nargs="+",
        default=list(SUPPORTED_ZONE_FAMILIES),
        help="Target zone-axis families to enumerate.",
    )
    parser.add_argument(
        "--include-opposites",
        action="store_true",
        help="Report both [uvw] and [-u-v-w]. Default reports one physical line, using the sign nearest the current zone.",
    )
    parser.add_argument("--alpha-limits", nargs=2, type=float, default=(-35.0, 35.0), metavar=("MIN", "MAX"))
    parser.add_argument("--beta-limits", nargs=2, type=float, default=(-30.0, 30.0), metavar=("MIN", "MAX"))
    parser.add_argument(
        "--holder-order",
        choices=["xy", "yx"],
        default="xy",
        help="'xy' uses R=Rx(alpha)@Ry(beta) and is the default; 'yx' uses R=Ry(beta)@Rx(alpha).",
    )
    parser.add_argument(
        "--image-to-holder-rotation-deg",
        type=float,
        default=90.0,
        help="CCW angle from image +x to holder +X in the diffraction image plane.",
    )
    parser.add_argument(
        "--rotate-pattern-180",
        action="store_true",
        help=(
            "Re-index the fitted diffraction pattern after applying a 180 deg in-plane rotation. "
            "Use this when the centrosymmetric 2D pattern fit has the opposite real-space direction."
        ),
    )
    parser.add_argument(
        "--show-in-plane-rotation-predictions",
        action="store_true",
        help=(
            "Report out-of-limit target axes that become reachable after sample in-plane loading rotation, "
            "and include their CCW rotation ranges."
        ),
    )
    parser.add_argument("--center", nargs=2, type=float, metavar=("X", "Y"), help="Direct-beam center in image pixels.")
    parser.add_argument("--n-peaks", type=int, default=120, help="Maximum detected bright peaks to keep.")
    parser.add_argument("--min-distance-px", type=float, default=None, help="Minimum spacing between detected peaks.")
    parser.add_argument("--spot-sigma-px", type=float, default=None, help="Gaussian smoothing sigma for spot detection.")
    parser.add_argument("--peak-percentile", type=float, default=99.0, help="Brightness percentile used for peak candidates.")
    parser.add_argument("--invert", action="store_true", help="Use this if diffraction spots are dark on a bright background.")
    parser.add_argument("--max-index", type=int, default=8, help="Maximum absolute h,k,l used in FCC reference spots.")
    parser.add_argument("--max-g-norm", type=float, default=9.0, help="Maximum reciprocal-vector length in reference spots.")
    parser.add_argument(
        "--tolerance-fraction",
        type=float,
        default=0.18,
        help="Match tolerance as a fraction of first-shell spot spacing.",
    )
    parser.add_argument("--output-prefix", type=Path, default=None, help="Prefix for selected output files.")
    parser.add_argument("--no-overlay", action="store_true", help="Do not write the fitted diffraction pattern PNG.")
    parser.add_argument("--no-labels", action="store_true", help="Do not draw Miller-index labels on output pattern images.")
    parser.add_argument(
        "--no-kikuchi-guides",
        action="store_true",
        help="Do not draw approximate low-order Kikuchi guide-line pairs on the overlay.",
    )
    parser.add_argument(
        "--kikuchi-max-g-norm",
        type=float,
        default=0.0,
        help="Largest indexed |g| used when drawing Kikuchi guides; 0 means first shell only.",
    )
    parser.add_argument(
        "--export-files",
        nargs="*",
        choices=["target_csv", "indexed_spots", "predicted_pattern", "fitted_pattern"],
        default=None,
        help="Files to export. Omit this option to export all; pass the option with no values to export none.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.image.exists():
        raise SystemExit(f"Image not found: {args.image}")

    alpha_limits = parse_limits(args.alpha_limits, "alpha limits")
    beta_limits = parse_limits(args.beta_limits, "beta limits")

    gray, rgb = load_grayscale(args.image, invert=args.invert)
    image_size = rgb.size
    peaks = detect_spots(
        gray,
        n_peaks=args.n_peaks,
        min_distance_px=args.min_distance_px,
        spot_sigma_px=args.spot_sigma_px,
        peak_percentile=args.peak_percentile,
    )
    if len(peaks) < 4:
        raise SystemExit("Too few diffraction spots were detected. Try lowering --peak-percentile.")

    center = tuple(args.center) if args.center is not None else None
    best, all_results = choose_best_match(
        peaks,
        image_size,
        current_zone=args.current_zone,
        center_xy=center,
        max_index=args.max_index,
        max_g_norm=args.max_g_norm,
        tolerance_fraction=args.tolerance_fraction,
    )
    if args.rotate_pattern_180:
        rotated_best = rotate_match_in_plane(best, 180.0, peaks, image_size, args.tolerance_fraction)
        all_results = [rotated_best if result is best else result for result in all_results]
        best = rotated_best

    crystal_to_zero_holder, _image_axes, _holder_axes = orientation_from_match(
        best,
        alpha_deg=args.alpha,
        beta_deg=args.beta,
        image_to_holder_rotation_deg=args.image_to_holder_rotation_deg,
        holder_order=args.holder_order,
    )

    rows = target_rows(
        best,
        crystal_to_zero_holder,
        alpha_deg=args.alpha,
        beta_deg=args.beta,
        target_families=args.target_families,
        include_opposites=args.include_opposites,
        holder_order=args.holder_order,
        alpha_limits=alpha_limits,
        beta_limits=beta_limits,
    )
    sample_rotation_rows = []
    if args.show_in_plane_rotation_predictions:
        sample_rotation_rows = in_plane_rotation_rows(
            best,
            crystal_to_zero_holder,
            target_families=args.target_families,
            include_opposites=args.include_opposites,
            holder_order=args.holder_order,
            alpha_limits=alpha_limits,
            beta_limits=beta_limits,
        )

    prefix = args.output_prefix
    if prefix is None:
        prefix = args.image.with_suffix("")

    if args.export_files is None:
        export_files = {"target_csv", "indexed_spots", "predicted_pattern", "fitted_pattern"}
    else:
        export_files = set(args.export_files)
    if args.no_overlay:
        export_files.discard("fitted_pattern")

    written: list[tuple[str, Path]] = []
    target_csv = Path(f"{prefix}_target_zone_axes.csv")
    indexed_csv = Path(f"{prefix}_indexed_spots.csv")
    predicted_path = Path(f"{prefix}_predicted_zone_pattern.png")
    fitted_path = Path(f"{prefix}_fitted_diffraction_pattern.png")

    if "target_csv" in export_files:
        write_targets_csv(target_csv, rows)
        written.append(("target angles", target_csv))
    if "indexed_spots" in export_files:
        write_indexed_spots_csv(indexed_csv, best, peaks)
        written.append(("indexed spots", indexed_csv))
    if "predicted_pattern" in export_files:
        predicted_pattern_image(
            best,
            image_size,
            rotated=False,
            title=f"Predicted FCC {format_miller(best.zone)} before rotation",
            show_labels=not args.no_labels,
            draw_guides=not args.no_kikuchi_guides,
            kikuchi_max_g_norm=args.kikuchi_max_g_norm,
        ).save(predicted_path)
        written.append(("predicted", predicted_path))
    if "fitted_pattern" in export_files:
        fitted_diffraction_image(
            rgb,
            best,
            draw_guides=not args.no_kikuchi_guides,
            kikuchi_max_g_norm=args.kikuchi_max_g_norm,
            show_labels=not args.no_labels,
            title=f"Fitted FCC {format_miller(best.zone)}",
        ).save(fitted_path)
        written.append(("fitted pattern", fitted_path))

    print_match_summary(best, all_results)
    if args.rotate_pattern_180:
        print("\nApplied correction: fitted pattern indexing rotated by 180 deg.")
    print_target_table(rows)
    if args.show_in_plane_rotation_predictions:
        print_in_plane_rotation_table(sample_rotation_rows)
    if written:
        print("\nWrote")
        for label, path in written:
            print(f"  {label:<15}: {path}")
    else:
        print("\nNo files were exported.")

    if args.current_zone is None:
        print(
            "\nNote: the auto-indexed [uvw] is a conventional cubic assignment. "
            "Use --current-zone if you need to force a specific equivalent index."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
