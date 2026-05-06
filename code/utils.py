"""
Shared utilities for the directional faithfulness experiment suite.

The experiments intentionally depend only on NumPy and the Python standard
library.  This keeps the numerical evidence reproducible in lightweight
environments and avoids binary-wheel issues in SciPy/sklearn/matplotlib.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np


EPS = 1e-12


def make_rng(seed: int | None = None) -> np.random.Generator:
    return np.random.default_rng(seed)


def normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, EPS)


def fibonacci_sphere(samples: int) -> np.ndarray:
    """Approximately uniform directions on S^2."""
    if samples <= 1:
        return np.array([[1.0, 0.0, 0.0]])
    points = []
    phi = math.pi * (3.0 - math.sqrt(5.0))
    for i in range(samples):
        y = 1.0 - (i / float(samples - 1)) * 2.0
        radius = math.sqrt(max(0.0, 1.0 - y * y))
        theta = phi * i
        points.append([math.cos(theta) * radius, y, math.sin(theta) * radius])
    return np.asarray(points, dtype=float)


def random_directions(n: int, dim: int = 3, rng: np.random.Generator | None = None) -> np.ndarray:
    rng = make_rng() if rng is None else rng
    return normalize_rows(rng.normal(size=(n, dim)))


def planar_directions(n: int) -> np.ndarray:
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.column_stack([np.cos(theta), np.sin(theta), np.zeros(n)])


def generate_tubes(
    directions: np.ndarray,
    delta: float,
    rng: np.random.Generator,
    a: float = 0.5,
    center_region: tuple[float, float] = (0.25, 0.75),
    plane_z: float | None = None,
) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """Generate unit-length tubes in R^d with radius a*delta."""
    low, high = center_region
    dim = directions.shape[1]
    centers = rng.uniform(low, high, size=(len(directions), dim))
    if plane_z is not None and dim >= 3:
        centers[:, 2] = plane_z
    tubes = []
    for center, direction in zip(centers, directions):
        direction = direction / max(np.linalg.norm(direction), EPS)
        tubes.append((center - 0.5 * direction, center + 0.5 * direction, a * delta))
    return tubes


def embed_tubes(
    tubes: Iterable[tuple[np.ndarray, np.ndarray, float]],
    dim: int,
    offset: np.ndarray | None = None,
) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """Embed lower-dimensional tubes into R^dim with an optional offset."""
    offset = np.zeros(dim) if offset is None else np.asarray(offset, dtype=float)
    embedded = []
    for p1, p2, r in tubes:
        q1 = np.zeros(dim)
        q2 = np.zeros(dim)
        q1[: len(p1)] = p1
        q2[: len(p2)] = p2
        embedded.append((q1 + offset, q2 + offset, r))
    return embedded


def point_to_tube_union_mask(
    points: np.ndarray,
    tubes: list[tuple[np.ndarray, np.ndarray, float]],
    chunk_size: int = 64,
) -> np.ndarray:
    """Return a boolean mask for points inside at least one tube."""
    points = np.asarray(points, dtype=float)
    occupied = np.zeros(len(points), dtype=bool)
    if len(points) == 0 or len(tubes) == 0:
        return occupied

    for start in range(0, len(tubes), chunk_size):
        batch = tubes[start : start + chunk_size]
        p1 = np.stack([t[0] for t in batch])
        p2 = np.stack([t[1] for t in batch])
        radii = np.asarray([t[2] for t in batch])
        v = p2 - p1
        length_sq = np.sum(v * v, axis=1)

        diff = points[:, None, :] - p1[None, :, :]
        proj = np.sum(diff * v[None, :, :], axis=2) / np.maximum(length_sq[None, :], EPS)
        proj = np.clip(proj, 0.0, 1.0)
        closest = p1[None, :, :] + proj[:, :, None] * v[None, :, :]
        dist_sq = np.sum((points[:, None, :] - closest) ** 2, axis=2)
        occupied |= np.any(dist_sq <= (radii[None, :] ** 2 + 1e-15), axis=1)
    return occupied


def grid_points(dim: int, spacing: float, domain: tuple[float, float] = (0.0, 1.0)) -> np.ndarray:
    """Cell-center grid over [domain[0], domain[1]]^dim."""
    low, high = domain
    n = int(math.ceil((high - low) / spacing))
    coords = low + (np.arange(n) + 0.5) * ((high - low) / n)
    mesh = np.meshgrid(*([coords] * dim), indexing="ij")
    return np.stack([m.ravel() for m in mesh], axis=1)


def grid_tube_cover_stats(
    tubes: list[tuple[np.ndarray, np.ndarray, float]],
    delta: float,
    dim: int,
    cover_spacing: float | None = None,
    volume_spacing: float | None = None,
    domain: tuple[float, float] = (0.0, 1.0),
) -> dict[str, float]:
    """
    Estimate union volume and delta-cover counts using deterministic grids.

    The cover count is the number of occupied cover cells.  This is a stable
    proxy for the number of radius-O(delta) balls required to cover the target.
    """
    cover_spacing = delta if cover_spacing is None else cover_spacing
    volume_spacing = delta / 2.0 if volume_spacing is None else volume_spacing

    cover_pts = grid_points(dim, cover_spacing, domain)
    cover_mask = point_to_tube_union_mask(cover_pts, tubes)
    cover_count = int(np.sum(cover_mask))

    volume_pts = grid_points(dim, volume_spacing, domain)
    volume_mask = point_to_tube_union_mask(volume_pts, tubes)
    volume_fraction = float(np.mean(volume_mask))
    volume = volume_fraction * ((domain[1] - domain[0]) ** dim)

    return {
        "cover_count": float(max(cover_count, 1)),
        "volume": float(max(volume, EPS)),
        "volume_fraction": float(volume_fraction),
        "cover_grid_points": float(len(cover_pts)),
        "volume_grid_points": float(len(volume_pts)),
    }


def sample_points_from_tube_nd(
    tube: tuple[np.ndarray, np.ndarray, float],
    rng: np.random.Generator,
    num_points: int,
) -> np.ndarray:
    """Uniform-ish samples from a tube in arbitrary ambient dimension."""
    p1, p2, radius = tube
    dim = len(p1)
    direction = p2 - p1
    length = np.linalg.norm(direction)
    if length <= EPS:
        return np.empty((0, dim))
    direction = direction / length

    samples = []
    for _ in range(num_points):
        t = rng.uniform(0.0, length)
        base = p1 + t * direction
        normal = rng.normal(size=dim)
        normal -= np.dot(normal, direction) * direction
        normal_norm = np.linalg.norm(normal)
        if normal_norm <= EPS:
            normal = np.zeros(dim)
        else:
            normal /= normal_norm
        radial = radius * (rng.uniform() ** (1.0 / max(dim - 1, 1)))
        samples.append(base + radial * normal)
    return np.asarray(samples)


def sample_points_from_tubes_nd(
    tubes: list[tuple[np.ndarray, np.ndarray, float]],
    rng: np.random.Generator,
    points_per_tube: int = 8,
) -> np.ndarray:
    if not tubes:
        return np.empty((0, 0))
    pts = [sample_points_from_tube_nd(tube, rng, points_per_tube) for tube in tubes]
    pts = [p for p in pts if len(p) > 0]
    if not pts:
        return np.empty((0, len(tubes[0][0])))
    return np.vstack(pts)


def greedy_cover_points(points: np.ndarray, radius: float) -> int:
    """
    Greedy set cover for sampled points.  This is used only in the additivity
    diagnostic, where sample sizes are intentionally modest.
    """
    if len(points) == 0:
        return 0
    dist_sq = np.sum((points[:, None, :] - points[None, :, :]) ** 2, axis=2)
    neighborhoods = dist_sq <= radius * radius
    uncovered = np.ones(len(points), dtype=bool)
    count = 0
    while np.any(uncovered):
        gains = neighborhoods[:, uncovered].sum(axis=1)
        center = int(np.argmax(gains))
        uncovered &= ~neighborhoods[center]
        count += 1
    return count


def fit_power_law(x_values: Iterable[float], y_values: Iterable[float]) -> tuple[float, float]:
    """Fit y ~= exp(intercept) * x^slope on log-log axes."""
    x = np.asarray(list(x_values), dtype=float)
    y = np.asarray(list(y_values), dtype=float)
    mask = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    if np.sum(mask) < 2:
        return float("nan"), float("nan")
    slope, intercept = np.polyfit(np.log(x[mask]), np.log(y[mask]), 1)
    return float(slope), float(intercept)


def mean_ci(values: Iterable[float]) -> tuple[float, float]:
    vals = np.asarray(list(values), dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return float("nan"), float("nan")
    if len(vals) == 1:
        return float(vals[0]), 0.0
    return float(np.mean(vals)), float(1.96 * np.std(vals, ddof=1) / math.sqrt(len(vals)))


def pairwise_sq_dists(a: np.ndarray, b: np.ndarray, chunk_size: int = 4096) -> np.ndarray:
    """Squared Euclidean distances from rows of a to rows of b."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    out = np.empty((len(a), len(b)), dtype=float)
    for start in range(0, len(a), chunk_size):
        chunk = a[start : start + chunk_size]
        diff = chunk[:, None, :] - b[None, :, :]
        out[start : start + chunk_size] = np.sum(diff * diff, axis=2)
    return np.maximum(np.nan_to_num(out, nan=1e30, posinf=1e30, neginf=1e30), 0.0)


def kmeans_fit(
    x: np.ndarray,
    k: int,
    rng: np.random.Generator,
    n_iter: int = 40,
    n_init: int = 3,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Small NumPy k-means with k-means++ initialization."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    k = int(min(max(1, k), n))
    best_centers = None
    best_assignments = None
    best_loss = float("inf")

    for _ in range(n_init):
        centers = np.empty((k, x.shape[1]), dtype=float)
        centers[0] = x[rng.integers(0, n)]
        min_dist = np.sum((x - centers[0]) ** 2, axis=1)
        for j in range(1, k):
            probs = min_dist / max(np.sum(min_dist), EPS)
            idx = rng.choice(n, p=probs)
            centers[j] = x[idx]
            min_dist = np.minimum(min_dist, np.sum((x - centers[j]) ** 2, axis=1))

        assignments = np.zeros(n, dtype=int)
        for _iter in range(n_iter):
            dist = pairwise_sq_dists(x, centers)
            new_assignments = np.argmin(dist, axis=1)
            if np.array_equal(new_assignments, assignments) and _iter > 0:
                break
            assignments = new_assignments
            for j in range(k):
                mask = assignments == j
                if np.any(mask):
                    centers[j] = np.mean(x[mask], axis=0)
                else:
                    centers[j] = x[rng.integers(0, n)]

        loss = float(np.mean(np.min(pairwise_sq_dists(x, centers), axis=1)))
        if loss < best_loss:
            best_loss = loss
            best_centers = centers.copy()
            best_assignments = assignments.copy()

    return best_centers, best_assignments, best_loss


def kmeans_predict(x: np.ndarray, centers: np.ndarray) -> np.ndarray:
    return np.argmin(pairwise_sq_dists(np.asarray(x, dtype=float), centers), axis=1)


@dataclass
class Standardizer:
    mean: np.ndarray
    scale: np.ndarray

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.scale

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return x * self.scale + self.mean


def fit_standardizer(x: np.ndarray) -> Standardizer:
    mean = np.mean(x, axis=0)
    scale = np.std(x, axis=0)
    scale = np.maximum(scale, 1e-6)
    return Standardizer(mean, scale)


def swiss_roll_from_params(params: np.ndarray) -> np.ndarray:
    """Smooth 2D Swiss-roll chart embedded in R^3."""
    t = params[:, 0]
    y = params[:, 1]
    x = t * np.cos(t)
    z = t * np.sin(t)
    return np.column_stack([x, z, y])


def sample_swiss_roll_params(
    n: int,
    rng: np.random.Generator,
    t_range: tuple[float, float] = (1.5 * np.pi, 4.5 * np.pi),
    y_range: tuple[float, float] = (-1.0, 1.0),
) -> np.ndarray:
    t = rng.uniform(t_range[0], t_range[1], size=n)
    y = rng.uniform(y_range[0], y_range[1], size=n)
    return np.column_stack([t, y])


def torus_from_params(params: np.ndarray, major_radius: float = 1.0, minor_radius: float = 0.35) -> np.ndarray:
    """Smooth 2D torus chart embedded in R^3."""
    u = params[:, 0]
    v = params[:, 1]
    radial = major_radius + minor_radius * np.cos(v)
    x = radial * np.cos(u)
    y = radial * np.sin(u)
    z = minor_radius * np.sin(v)
    return np.column_stack([x, y, z])


def sample_torus_params(n: int, rng: np.random.Generator) -> np.ndarray:
    u = rng.uniform(0.0, 2.0 * np.pi, size=n)
    v = rng.uniform(0.0, 2.0 * np.pi, size=n)
    return np.column_stack([u, v])


def swiss_roll_tangent_basis(params: np.ndarray, standardizer: Standardizer | None = None) -> np.ndarray:
    """Analytic tangent basis dX/dt and dX/dy for each parameter point."""
    t = params[:, 0]
    dt = np.column_stack([np.cos(t) - t * np.sin(t), np.sin(t) + t * np.cos(t), np.zeros_like(t)])
    dy = np.column_stack([np.zeros_like(t), np.zeros_like(t), np.ones_like(t)])
    basis = np.stack([dt, dy], axis=1)
    if standardizer is not None:
        basis = basis / standardizer.scale[None, None, :]
    return basis


def torus_tangent_basis(
    params: np.ndarray,
    standardizer: Standardizer | None = None,
    major_radius: float = 1.0,
    minor_radius: float = 0.35,
) -> np.ndarray:
    """Analytic tangent basis dX/du and dX/dv for a torus."""
    u = params[:, 0]
    v = params[:, 1]
    radial = major_radius + minor_radius * np.cos(v)
    du = np.column_stack([-radial * np.sin(u), radial * np.cos(u), np.zeros_like(u)])
    dv = np.column_stack(
        [
            -minor_radius * np.sin(v) * np.cos(u),
            -minor_radius * np.sin(v) * np.sin(u),
            minor_radius * np.cos(v),
        ]
    )
    basis = np.stack([du, dv], axis=1)
    if standardizer is not None:
        basis = basis / standardizer.scale[None, None, :]
    return basis


def sample_manifold_params(
    manifold: str,
    n: int,
    rng: np.random.Generator,
    base_region: bool = False,
) -> np.ndarray:
    """Sample parameters for supported 2D manifolds."""
    if manifold == "swiss_roll":
        if base_region:
            return sample_swiss_roll_params(
                n,
                rng,
                t_range=(1.8 * math.pi, 4.2 * math.pi),
                y_range=(-0.75, 0.75),
            )
        return sample_swiss_roll_params(n, rng)
    if manifold == "torus":
        return sample_torus_params(n, rng)
    raise ValueError(f"unknown manifold: {manifold}")


def manifold_from_params(manifold: str, params: np.ndarray) -> np.ndarray:
    if manifold == "swiss_roll":
        return swiss_roll_from_params(params)
    if manifold == "torus":
        return torus_from_params(params)
    raise ValueError(f"unknown manifold: {manifold}")


def manifold_tangent_basis(
    manifold: str,
    params: np.ndarray,
    standardizer: Standardizer | None = None,
) -> np.ndarray:
    if manifold == "swiss_roll":
        return swiss_roll_tangent_basis(params, standardizer=standardizer)
    if manifold == "torus":
        return torus_tangent_basis(params, standardizer=standardizer)
    raise ValueError(f"unknown manifold: {manifold}")


def make_directional_trajectories(
    base_params: np.ndarray,
    angles: np.ndarray,
    step_radius: float,
    num_steps: int,
    manifold: str = "swiss_roll",
    point_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    standardizer: Standardizer | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return trajectories with shape (B, A, T, 3) and analytic directions at t=0.
    """
    times = np.linspace(-0.5, 0.5, num_steps)
    all_traj = []
    all_dirs = []
    for base in base_params:
        traj_for_base = []
        dirs_for_base = []
        basis = manifold_tangent_basis(manifold, base[None, :], standardizer=standardizer)[0]
        for angle in angles:
            param_dir = np.array([math.cos(angle), math.sin(angle)])
            params = base[None, :] + times[:, None] * step_radius * param_dir[None, :]
            if manifold == "torus":
                params[:, 0] = np.mod(params[:, 0], 2.0 * np.pi)
                params[:, 1] = np.mod(params[:, 1], 2.0 * np.pi)
            points = manifold_from_params(manifold, params) if point_fn is None else point_fn(params)
            if standardizer is not None:
                points = standardizer.transform(points)
            tangent = param_dir @ basis
            tangent = tangent / max(np.linalg.norm(tangent), EPS)
            traj_for_base.append(points)
            dirs_for_base.append(tangent)
        all_traj.append(traj_for_base)
        all_dirs.append(dirs_for_base)
    return np.asarray(all_traj), np.asarray(all_dirs)


def principal_direction(points: np.ndarray) -> np.ndarray | None:
    centered = points - np.mean(points, axis=0, keepdims=True)
    if np.linalg.norm(centered) <= EPS:
        return None
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return vt[0]


def hamming_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.asarray(a) != np.asarray(b)))


def write_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def write_pdf_line_plot(
    path: str,
    series: list[dict],
    xlabel: str,
    ylabel: str,
    title: str,
    subtitle: str | None = None,
    width: int = 612,
    height: int = 420,
    logx: bool = False,
    logy: bool = False,
    y_limits: tuple[float, float] | None = None,
    max_ticks: int = 5,
) -> None:
    """Publication-style Matplotlib/Seaborn line plot saved as PDF and PNG."""
    try:
        plt = _load_matplotlib()
    except Exception as exc:
        _write_plot_skip_notice(path, exc)
        return
    fig, ax = plt.subplots(figsize=(width / 72.0, height / 72.0), constrained_layout=True)
    _plot_series_on_axis(ax, series, logx=logx, logy=logy)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=14)
    if subtitle:
        ax.text(0.5, 1.01, subtitle, transform=ax.transAxes, ha="center", va="bottom", fontsize=8.5)
    if y_limits is not None:
        ax.set_ylim(*y_limits)
    ax.grid(True, which="major", color="#d9d9d9", linewidth=0.8)
    ax.grid(True, which="minor", color="#eeeeee", linewidth=0.5, alpha=0.7)
    ax.legend(frameon=True, framealpha=0.95, edgecolor="#d0d0d0", fontsize=8)
    _save_matplotlib_figure(fig, path)


def _load_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        import seaborn as sns

        sns.set_theme(style="whitegrid", context="paper", font_scale=1.08)
    except Exception:
        plt.style.use("default")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 11,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    return plt


def _plot_series_on_axis(ax, series: list[dict], logx: bool = False, logy: bool = False) -> None:
    palette = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000"]
    markers = ["o", "s", "^", "D", "v", "P", "X"]
    for idx, s in enumerate(series):
        color = s.get("color", palette[idx % len(palette)])
        marker = s.get("marker", markers[idx % len(markers)])
        x = np.asarray(s["x"], dtype=float)
        y = np.asarray(s["y"], dtype=float)
        yerr = np.asarray(s["yerr"], dtype=float) if "yerr" in s else None
        if yerr is not None and logy:
            yerr = np.minimum(yerr, np.maximum(y * 0.95, EPS))
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            label=s["label"],
            color=color,
            marker=marker,
            linewidth=1.8,
            markersize=4.8,
            capsize=2.5 if yerr is not None else 0,
            elinewidth=1.0,
        )
    if logx:
        ax.set_xscale("log", base=2)
    if logy:
        ax.set_yscale("log")


def _save_matplotlib_figure(fig, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    root, _ext = os.path.splitext(path)
    fig.savefig(root + ".pdf", bbox_inches="tight")
    fig.savefig(root + ".png", bbox_inches="tight", dpi=300)
    import matplotlib.pyplot as plt

    plt.close(fig)


def write_pdf_facet_line_plot(
    path: str,
    series: list[dict],
    facet_order: list[str],
    xlabel: str,
    ylabel: str,
    title: str,
    subtitle: str | None = None,
    width: int = 760,
    height: int = 360,
    logx: bool = False,
    logy: bool = False,
    y_limits: tuple[float, float] | None = None,
    sharey: bool = True,
) -> None:
    """Two-or-more-panel Matplotlib/Seaborn line plot, saved as PDF and PNG."""
    try:
        plt = _load_matplotlib()
    except Exception as exc:
        _write_plot_skip_notice(path, exc)
        return
    n = len(facet_order)
    fig, axes = plt.subplots(
        1,
        n,
        figsize=(width / 72.0, height / 72.0),
        sharey=sharey,
        constrained_layout=True,
    )
    if n == 1:
        axes = [axes]

    handles = []
    labels = []
    for ax, facet in zip(axes, facet_order):
        facet_series = [s for s in series if s.get("facet") == facet]
        _plot_series_on_axis(ax, facet_series, logx=logx, logy=logy)
        ax.set_title(facet.replace("_", " ").title(), pad=8)
        ax.set_xlabel(xlabel)
        if ax is axes[0]:
            ax.set_ylabel(ylabel)
        else:
            ax.set_ylabel("")
        if y_limits is not None:
            ax.set_ylim(*y_limits)
        ax.grid(True, which="major", color="#d9d9d9", linewidth=0.8)
        ax.grid(True, which="minor", color="#eeeeee", linewidth=0.5, alpha=0.7)
        h, l = ax.get_legend_handles_labels()
        for handle, label in zip(h, l):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()

    fig.suptitle(title, y=1.05, fontsize=12)
    if subtitle:
        fig.text(0.5, 1.005, subtitle, ha="center", va="bottom", fontsize=8.5)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=min(4, len(labels)), frameon=False)
    _save_matplotlib_figure(fig, path)


def _write_plot_skip_notice(path: str, exc: Exception) -> None:
    root, _ext = os.path.splitext(path)
    notice_path = root + "_PLOT_SKIPPED.txt"
    write_text(
        notice_path,
        "Plot generation was skipped because Matplotlib/Seaborn could not be imported in this Python environment.\n"
        "The numerical experiment still completed and wrote its summary table.\n\n"
        f"Original plotting target: {path}\n"
        f"Import error: {type(exc).__name__}: {exc}\n",
    )
    print(f"Plot skipped for {path}: {type(exc).__name__}: {exc}")
