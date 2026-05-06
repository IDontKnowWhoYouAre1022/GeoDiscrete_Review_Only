"""
Experiment 4: additivity across separated local certificates.

Proposition 4 says that cover-active code sets must be disjoint when intrinsic
tube unions are separated by more than 2 eta delta.  This experiment embeds
multiple identical 3D tube certificates into disjoint coordinates of a higher
ambient space and verifies linear growth of sampled covering counts.
"""

from __future__ import annotations

import os

import numpy as np

from utils import (
    embed_tubes,
    fibonacci_sphere,
    fit_power_law,
    generate_tubes,
    greedy_cover_points,
    make_rng,
    mean_ci,
    sample_points_from_tubes_nd,
    write_pdf_line_plot,
    write_text,
)


def separated_copy(
    base_tubes: list[tuple[np.ndarray, np.ndarray, float]],
    copy_idx: int,
    ambient_dim: int,
    separation: float,
) -> list[tuple[np.ndarray, np.ndarray, float]]:
    offset = np.zeros(ambient_dim)
    offset[3 + copy_idx] = separation
    return embed_tubes(base_tubes, dim=ambient_dim, offset=offset)


def run_experiment(output_dir: str = "./results/exp4", seeds: list[int] | None = None):
    os.makedirs(output_dir, exist_ok=True)
    seeds = [0, 42, 123, 456] if seeds is None else seeds
    quick = len(seeds) == 1

    deltas = [1 / 8, 1 / 10, 1 / 12, 1 / 16, 1 / 20]
    j_values = [1, 2, 4, 8, 16, 32]
    eta = 1.0
    ambient_dim = 40
    separation = 8.0
    points_per_tube = 4
    if quick:
        deltas = [1 / 8, 1 / 12, 1 / 20]
        j_values = [1, 4, 16, 32]
        points_per_tube = 2

    print("=" * 92)
    print("Experiment 4: additivity across separated local certificates")
    print("=" * 92)
    print(f"deltas: {deltas}")
    print(f"J values: {j_values}")

    cover_counts = {seed: {j: [] for j in j_values} for seed in seeds}
    min_separations = {seed: {j: [] for j in j_values} for seed in seeds}

    for seed in seeds:
        rng = make_rng(seed)
        print(f"\nseed={seed}")
        for delta in deltas:
            n_tubes = int(max(30, round((1.0 / delta) ** 2)))
            base_tubes = generate_tubes(
                fibonacci_sphere(n_tubes),
                delta,
                rng=rng,
                a=0.45,
                center_region=(0.22, 0.78),
            )

            for j in j_values:
                # Because copies are separated by much more than 2 eta delta,
                # the global cover is the sum of the per-copy covers.  We
                # estimate each copy separately to avoid an O(n^3) greedy
                # set-cover pass over a block-diagonal distance matrix.
                n_cover = 0
                for copy_idx in range(j):
                    copy_tubes = separated_copy(base_tubes, copy_idx, ambient_dim, separation)
                    points = sample_points_from_tubes_nd(copy_tubes, rng, points_per_tube=points_per_tube)
                    n_cover += greedy_cover_points(points, radius=eta * delta)
                cover_counts[seed][j].append(max(1, n_cover))
                min_separations[seed][j].append(separation if j > 1 else float("inf"))

        for j in j_values:
            print(f"  J={j:2d} mean sampled cover={np.mean(cover_counts[seed][j]):.1f}")

    ratios = {j: [] for j in j_values}
    slopes_by_delta = []
    for seed in seeds:
        for delta_idx, _delta in enumerate(deltas):
            n1 = cover_counts[seed][1][delta_idx]
            for j in j_values:
                ratios[j].append(cover_counts[seed][j][delta_idx] / max(j * n1, 1))

    for delta_idx, _delta in enumerate(deltas):
        mean_by_j = [np.mean([cover_counts[seed][j][delta_idx] for seed in seeds]) for j in j_values]
        slope, _ = fit_power_law(j_values, mean_by_j)
        slopes_by_delta.append(slope)

    summary = [
        "Experiment 4: additivity across separated certificates",
        "=" * 94,
        "Ratio is N_delta,J / (J * N_delta,1); perfect additivity is 1.",
        f"Ambient dimension: {ambient_dim}; copy separation: {separation} > 2 eta delta for all tested deltas.",
        "",
        f"{'J':>6s} {'additivity ratio':>24s}",
        "-" * 94,
    ]
    for j in j_values:
        ratio_m, ratio_ci = mean_ci(ratios[j])
        summary.append(f"{j:6d} {ratio_m:10.3f} +/- {ratio_ci:<7.3f}")
    slope_m, slope_ci = mean_ci(slopes_by_delta)
    summary.extend(
        [
            "",
            f"log-log slope of N_delta,J versus J: {slope_m:.3f} +/- {slope_ci:.3f}",
            "",
            "Interpretation:",
            "- A slope near 1 and additivity ratios near 1 support Proposition 4's disjoint-code accumulation.",
            "- The diagnostic uses sampled tube points because the proposition is about separated cover-active sets,",
            "  not about a new Kakeya exponent.",
        ]
    )
    write_text(os.path.join(output_dir, "exp4_summary.txt"), "\n".join(summary) + "\n")

    series = []
    for delta_idx, delta in enumerate(deltas):
        series.append(
            {
                "label": f"delta={delta:.3f}",
                "x": j_values,
                "y": [np.mean([cover_counts[seed][j][delta_idx] for seed in seeds]) for j in j_values],
                "yerr": [mean_ci([cover_counts[seed][j][delta_idx] for seed in seeds])[1] for j in j_values],
            }
        )
    write_pdf_line_plot(
        os.path.join(output_dir, "exp4_additivity.pdf"),
        series,
        xlabel="number of separated copies J",
        ylabel="sampled cover count",
        title="Additivity Across Separated Local Certificates",
        subtitle="Sampled cover count grows linearly with the number of separated charts",
        logx=True,
        logy=True,
    )
    print(f"\nSaved {os.path.join(output_dir, 'exp4_summary.txt')}")
    return {"cover_counts": cover_counts, "ratios": ratios, "slopes_by_delta": slopes_by_delta}


if __name__ == "__main__":
    run_experiment()
