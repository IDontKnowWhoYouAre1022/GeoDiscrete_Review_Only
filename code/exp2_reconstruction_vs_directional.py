"""
Experiment 2: pointwise reconstruction versus directional faithfulness.

This experiment matches the strong-faithfulness diagnostic in the theory more
closely than a generic trajectory comparison: for each base point, we sample
multiple tangent directions, tokenize equal-length trajectories, and measure
whether angularly separated directions produce separated symbolic sequences.
"""

from __future__ import annotations

import math
import os

import numpy as np

from utils import (
    fit_standardizer,
    hamming_distance,
    kmeans_fit,
    kmeans_predict,
    make_directional_trajectories,
    make_rng,
    manifold_from_params,
    mean_ci,
    principal_direction,
    sample_manifold_params,
    write_pdf_line_plot,
    write_text,
)


def _angular_separation(a: float, b: float) -> float:
    raw = abs(a - b) % (2.0 * math.pi)
    return min(raw, 2.0 * math.pi - raw)


def compute_directional_metrics(
    trajectories: np.ndarray,
    true_dirs: np.ndarray,
    codebook: np.ndarray,
    assignment_fn,
    angles: np.ndarray,
    theta: float = math.pi / 3.0,
    alpha: float = 0.35,
) -> dict[str, float]:
    """Strong-faithfulness metrics at fixed base points."""
    num_bases, num_angles, _, _ = trajectories.shape
    hamming_values = []
    strong_passes = []
    angular_errors = []
    diversity_values = []
    crossing_rates = []

    for b in range(num_bases):
        codes_by_angle = []
        for a_idx in range(num_angles):
            codes = assignment_fn(trajectories[b, a_idx])
            codes_by_angle.append(codes)

            recon = codebook[codes]
            decoded_dir = principal_direction(recon)
            if decoded_dir is None:
                angular_errors.append(math.pi / 2.0)
            else:
                if np.dot(decoded_dir, true_dirs[b, a_idx]) < 0:
                    decoded_dir = -decoded_dir
                cos_angle = float(np.clip(np.dot(decoded_dir, true_dirs[b, a_idx]), -1.0, 1.0))
                angular_errors.append(math.acos(cos_angle))

            diversity_values.append(len(np.unique(codes)) / len(codes))
            crossing_rates.append(np.mean(codes[1:] != codes[:-1]))

        for i in range(num_angles):
            for j in range(i + 1, num_angles):
                if _angular_separation(float(angles[i]), float(angles[j])) >= theta:
                    d_h = hamming_distance(codes_by_angle[i], codes_by_angle[j])
                    hamming_values.append(d_h)
                    strong_passes.append(float(d_h >= alpha))

    return {
        "mean_hamming": float(np.mean(hamming_values)),
        "strong_pass_rate": float(np.mean(strong_passes)),
        "directional_failure": float(1.0 - np.mean(strong_passes)),
        "angular_error": float(np.mean(angular_errors)),
        "code_diversity": float(np.mean(diversity_values)),
        "boundary_crossing_rate": float(np.mean(crossing_rates)),
    }


def run_experiment(output_dir: str = "./results/exp2", seeds: list[int] | None = None):
    os.makedirs(output_dir, exist_ok=True)
    seeds = [42, 123, 456] if seeds is None else seeds
    quick = len(seeds) == 1

    k_values = [16, 32, 64, 128, 256, 512, 1024]
    train_n = 10000
    test_n = 2400
    base_n = 160
    datasets = ["swiss_roll", "torus"]
    step_radius_by_dataset = {"swiss_roll": 0.32, "torus": 0.22}
    angles = np.linspace(0.0, 2.0 * math.pi, 12, endpoint=False)
    num_steps = 20
    if quick:
        k_values = [16, 64, 256, 1024]
        train_n = 3500
        test_n = 900
        base_n = 60
        angles = np.linspace(0.0, 2.0 * math.pi, 8, endpoint=False)
        num_steps = 14

    print("=" * 88)
    print("Experiment 2: reconstruction versus directional faithfulness")
    print("=" * 88)
    print(f"K values: {k_values}")
    print(f"datasets: {datasets}")
    print(f"train_n={train_n}, test_n={test_n}, base_n={base_n}")
    print(f"seeds: {seeds}")

    metric_names = [
        "reconstruction",
        "mean_hamming",
        "strong_pass_rate",
        "directional_failure",
        "angular_error",
        "code_diversity",
        "boundary_crossing_rate",
    ]
    metrics = {dataset: {name: {k: [] for k in k_values} for name in metric_names} for dataset in datasets}

    for seed in seeds:
        print(f"\nseed={seed}")
        for dataset_idx, dataset in enumerate(datasets):
            rng = make_rng(seed + 1009 * dataset_idx)
            print(f" dataset={dataset}")
            train_params = sample_manifold_params(dataset, train_n, rng)
            test_params = sample_manifold_params(dataset, test_n, rng)
            train_raw = manifold_from_params(dataset, train_params)
            test_raw = manifold_from_params(dataset, test_params)
            standardizer = fit_standardizer(train_raw)
            train_x = standardizer.transform(train_raw)
            test_x = standardizer.transform(test_raw)

            base_params = sample_manifold_params(dataset, base_n, rng, base_region=True)
            trajectories, true_dirs = make_directional_trajectories(
                base_params,
                angles,
                step_radius=step_radius_by_dataset[dataset],
                num_steps=num_steps,
                manifold=dataset,
                standardizer=standardizer,
            )

            for k in k_values:
                centers, _, _ = kmeans_fit(train_x, k, rng, n_iter=30, n_init=2)
                test_codes = kmeans_predict(test_x, centers)
                recon = centers[test_codes]
                recon_mse = float(np.mean(np.sum((test_x - recon) ** 2, axis=1)))

                def assign_fn(points, centers=centers):
                    return kmeans_predict(points, centers)

                d_metrics = compute_directional_metrics(
                    trajectories,
                    true_dirs,
                    centers,
                    assign_fn,
                    angles,
                    theta=math.pi / 3.0,
                    alpha=0.35,
                )

                metrics[dataset]["reconstruction"][k].append(recon_mse)
                for name, value in d_metrics.items():
                    metrics[dataset][name][k].append(value)
                print(
                    f"  K={k:4d} recon={recon_mse:.4f} "
                    f"strong={d_metrics['strong_pass_rate']:.3f} "
                    f"angular={d_metrics['angular_error']:.3f}"
                )

    summary = [
        "Experiment 2: reconstruction versus directional faithfulness",
        "=" * 98,
        "Strong pass rate is the fraction of same-base, angle-separated trajectory pairs",
        "whose Hamming distance is at least alpha=0.35.",
        "",
        f"{'dataset':>12s} {'K':>6s} {'recon MSE':>18s} {'strong pass':>18s} {'mean ham':>18s} {'angular err':>18s}",
        f"{'':>12s} {'':>6s} {'failure':>18s} {'diversity':>18s} {'crossing':>18s} {'':>18s}",
        "-" * 98,
    ]
    for dataset in datasets:
        for k in k_values:
            recon_m, recon_ci = mean_ci(metrics[dataset]["reconstruction"][k])
            pass_m, pass_ci = mean_ci(metrics[dataset]["strong_pass_rate"][k])
            ham_m, ham_ci = mean_ci(metrics[dataset]["mean_hamming"][k])
            angular_m, angular_ci = mean_ci(metrics[dataset]["angular_error"][k])
            fail_m, fail_ci = mean_ci(metrics[dataset]["directional_failure"][k])
            div_m, div_ci = mean_ci(metrics[dataset]["code_diversity"][k])
            cross_m, cross_ci = mean_ci(metrics[dataset]["boundary_crossing_rate"][k])
            summary.append(
                f"{dataset:>12s} {k:6d} {recon_m:8.4f} +/- {recon_ci:<7.4f} "
                f"{pass_m:8.4f} +/- {pass_ci:<7.4f} "
                f"{ham_m:8.4f} +/- {ham_ci:<7.4f} "
                f"{angular_m:8.4f} +/- {angular_ci:<7.4f} "
            )
            summary.append(
                f"{'':>12s} {'':6s} {fail_m:8.4f} +/- {fail_ci:<7.4f} "
                f"{div_m:8.4f} +/- {div_ci:<7.4f} "
                f"{cross_m:8.4f} +/- {cross_ci:<7.4f}"
            )
    summary.extend(
        [
            "",
            "Interpretation:",
            "- Reconstruction MSE measures pointwise quantization quality.",
            "- Strong pass rate directly tests symbolic separation of angularly separated directions.",
            "- A gap between improving MSE and delayed strong-pass improvement is the empirical alignment tax.",
        ]
    )

    write_text(os.path.join(output_dir, "exp2_summary.txt"), "\n".join(summary) + "\n")
    colors = {"swiss_roll": "#0072B2", "torus": "#D55E00"}

    def dataset_series(metric: str, label_suffix: str = ""):
        return [
            {
                "label": f"{dataset}{label_suffix}",
                "x": k_values,
                "y": [mean_ci(metrics[dataset][metric][k])[0] for k in k_values],
                "yerr": [mean_ci(metrics[dataset][metric][k])[1] for k in k_values],
                "color": colors[dataset],
            }
            for dataset in datasets
        ]

    write_pdf_line_plot(
        os.path.join(output_dir, "exp2_reconstruction.pdf"),
        dataset_series("reconstruction"),
        xlabel="code count K",
        ylabel="MSE",
        title="Pointwise Reconstruction Across Manifolds",
        subtitle="Larger datasets: 10k train, 2.4k test, 160 base points per manifold",
        logx=True,
        logy=True,
    )
    write_pdf_line_plot(
        os.path.join(output_dir, "exp2_directional.pdf"),
        dataset_series("strong_pass_rate"),
        xlabel="code count K",
        ylabel="strong pass rate",
        title="Same-Base Symbolic Directional Separation",
        subtitle="Fraction of angularly separated trajectories with Hamming distance >= 0.35",
        logx=True,
        y_limits=(0.0, 1.0),
    )
    write_pdf_line_plot(
        os.path.join(output_dir, "exp2_hamming.pdf"),
        dataset_series("mean_hamming"),
        xlabel="code count K",
        ylabel="mean Hamming distance",
        title="Symbolic Trajectory Hamming Separation",
        subtitle="Same-base angularly separated trajectory pairs",
        logx=True,
        y_limits=(0.0, 1.0),
    )
    write_pdf_line_plot(
        os.path.join(output_dir, "exp2_angular.pdf"),
        dataset_series("angular_error"),
        xlabel="code count K",
        ylabel="radians",
        title="Decoded Trajectory Angular Error",
        subtitle="Lower is better; error bars are 95% confidence intervals over seeds",
        logx=True,
    )
    write_pdf_line_plot(
        os.path.join(output_dir, "exp2_diversity.pdf"),
        dataset_series("code_diversity"),
        xlabel="code count K",
        ylabel="unique-token fraction",
        title="Trajectory Code Diversity",
        subtitle="Fraction of distinct tokens along local tangent trajectories",
        logx=True,
        y_limits=(0.0, 1.0),
    )
    write_pdf_line_plot(
        os.path.join(output_dir, "exp2_crossing.pdf"),
        dataset_series("boundary_crossing_rate"),
        xlabel="code count K",
        ylabel="boundary crossing rate",
        title="Symbolic Trajectory Boundary Crossings",
        subtitle="Fraction of consecutive trajectory steps that change token",
        logx=True,
        y_limits=(0.0, 1.0),
    )
    print(f"\nSaved {os.path.join(output_dir, 'exp2_summary.txt')}")
    return metrics


if __name__ == "__main__":
    run_experiment()
