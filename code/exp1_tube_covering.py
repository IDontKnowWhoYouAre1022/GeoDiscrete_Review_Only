"""
Experiment 1: 3D tube-union covering and directional capacity scaling.

This experiment is a controlled geometric stress test for the base mechanism
behind Theorem 1/2: a direction-rich 3D tube union has enough volume that an
O(delta)-cover needs about delta^{-3} representatives.  A planar control keeps
the same local tube radius but removes 3D richness, producing about delta^{-2}
scaling.
"""

from __future__ import annotations

import math
import os

import numpy as np

from utils import (
    fibonacci_sphere,
    fit_power_law,
    generate_tubes,
    grid_tube_cover_stats,
    make_rng,
    mean_ci,
    planar_directions,
    random_directions,
    write_pdf_line_plot,
    write_text,
)


def _directions(kind: str, n: int, rng: np.random.Generator) -> np.ndarray:
    if kind == "isotropic":
        return fibonacci_sphere(n)
    if kind == "generic":
        return random_directions(n, dim=3, rng=rng)
    if kind == "planar":
        return planar_directions(n)
    raise ValueError(kind)


def run_experiment(output_dir: str = "./results/exp1", seeds: list[int] | None = None):
    os.makedirs(output_dir, exist_ok=True)
    seeds = [0, 11, 2024, 31337] if seeds is None else seeds

    deltas = [1 / 6, 1 / 8, 1 / 10, 1 / 12, 1 / 16]
    eta = 1.0
    tube_radius_coeff = 0.45
    geometries = {
        "3D isotropic directions": "isotropic",
        "3D random directions": "generic",
        "Planar control": "planar",
    }

    print("=" * 88)
    print("Experiment 1: 3D tube-union covering")
    print("=" * 88)
    print(f"deltas: {deltas}")
    print(f"seeds: {seeds}")

    results = {
        name: {
            "cover_counts": [],
            "volumes": [],
            "volume_lower_bounds": [],
            "cover_slopes": [],
            "volume_slopes": [],
        }
        for name in geometries
    }

    for seed in seeds:
        rng = make_rng(seed)
        print(f"\nseed={seed}")
        for name, kind in geometries.items():
            cover_counts = []
            volumes = []
            lower_bounds = []
            for delta in deltas:
                n_tubes = int(max(36, round((1.0 / delta) ** 2)))
                directions = _directions(kind, n_tubes, rng)
                plane_z = 0.5 if kind == "planar" else None
                tubes = generate_tubes(
                    directions,
                    delta,
                    rng=rng,
                    a=tube_radius_coeff,
                    center_region=(0.22, 0.78),
                    plane_z=plane_z,
                )
                stats = grid_tube_cover_stats(tubes, delta, dim=3)
                volumes.append(stats["volume"])
                ball_volume = 4.0 * math.pi * (eta * delta) ** 3 / 3.0
                lower_bound = stats["volume"] / ball_volume
                lower_bounds.append(lower_bound)
                # Use the theorem's volume/ball conversion as the primary
                # cover proxy.  Raw occupied-cell counts are sensitive to grid
                # phase for codimension-one controls such as the planar case.
                cover_counts.append(lower_bound)

            inv_delta = [1.0 / d for d in deltas]
            cover_slope, _ = fit_power_law(inv_delta, cover_counts)
            volume_slope_raw, _ = fit_power_law(deltas, volumes)
            effective_s = 3.0 - volume_slope_raw

            results[name]["cover_counts"].append(cover_counts)
            results[name]["volumes"].append(volumes)
            results[name]["volume_lower_bounds"].append(lower_bounds)
            results[name]["cover_slopes"].append(cover_slope)
            results[name]["volume_slopes"].append(effective_s)
            print(f"  {name:24s} cover slope={cover_slope:5.2f} volume-derived s={effective_s:5.2f}")

    summary_lines = [
        "Experiment 1: 3D tube-union covering",
        "=" * 88,
        "The cover proxy is the theorem's volume divided by the volume of an eta*delta ball.",
        "The volume-derived exponent is s = 3 - slope_delta(volume).",
        "",
        f"{'Geometry':28s} {'cover exponent':>18s} {'volume-derived s':>20s}",
        "-" * 88,
    ]
    series = []
    colors = ["#0072B2", "#D55E00", "#009E73"]
    for idx, name in enumerate(geometries):
        cover_mean, cover_ci = mean_ci(results[name]["cover_slopes"])
        vol_mean, vol_ci = mean_ci(results[name]["volume_slopes"])
        summary_lines.append(
            f"{name:28s} {cover_mean:8.3f} +/- {cover_ci:<7.3f} {vol_mean:8.3f} +/- {vol_ci:<7.3f}"
        )

        mean_counts = np.mean(np.asarray(results[name]["cover_counts"]), axis=0)
        ci_counts = [
            mean_ci(np.asarray(results[name]["cover_counts"])[:, delta_idx])[1]
            for delta_idx in range(len(deltas))
        ]
        series.append(
            {
                "label": f"{name} (s={cover_mean:.2f})",
                "x": [1.0 / d for d in deltas],
                "y": mean_counts,
                "yerr": ci_counts,
                "color": colors[idx],
            }
        )

    summary_lines.extend(
        [
            "",
            "Interpretation:",
            "- The two 3D controls should be close to exponent 3 at this finite scale.",
            "- The planar control should be close to exponent 2, showing that direction richness matters.",
            "- These are numerical stress tests of the cover-volume mechanism, not proofs of Kakeya estimates.",
        ]
    )

    write_text(os.path.join(output_dir, "exp1_summary.txt"), "\n".join(summary_lines) + "\n")
    write_pdf_line_plot(
        os.path.join(output_dir, "exp1_cover_scaling.pdf"),
        series,
        xlabel="1 / delta",
        ylabel="volume / ball-volume lower bound",
        title="3D Tube-Union Cover Scaling",
        subtitle="Volume-to-ball lower bound with 95% confidence intervals over seeds",
        logx=True,
        logy=True,
    )
    print(f"\nSaved {os.path.join(output_dir, 'exp1_summary.txt')}")
    print(f"Saved {os.path.join(output_dir, 'exp1_cover_scaling.pdf')}")
    return results


if __name__ == "__main__":
    run_experiment()
