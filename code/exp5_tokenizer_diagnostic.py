"""
Experiment 5: learned-tokenizer directional diagnostic.

This is the ML relevance experiment.  We train a small continuous autoencoder
on a Swiss-roll data manifold, then compare three practical tokenizer styles
on learned latent trajectories:

1. VQ-style nearest-neighbor k-means codebooks.
2. FSQ-style finite scalar grids.
3. LGQ-style learned-metric k-means after whitening the latent geometry.

The diagnostic measures both directional collapse and trajectory instability:
same-base tangent directions should produce distinguishable symbolic sequences
without destroying decoded trajectory directions.
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
    pairwise_sq_dists,
    principal_direction,
    sample_manifold_params,
    write_pdf_facet_line_plot,
    write_text,
)


class NumpyAutoencoder:
    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int, rng: np.random.Generator):
        scale1 = math.sqrt(2.0 / (input_dim + hidden_dim))
        scale2 = math.sqrt(2.0 / (hidden_dim + latent_dim))
        scale3 = math.sqrt(2.0 / (latent_dim + hidden_dim))
        scale4 = math.sqrt(2.0 / (hidden_dim + input_dim))
        self.w1 = rng.normal(scale=scale1, size=(input_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim)
        self.wz = rng.normal(scale=scale2, size=(hidden_dim, latent_dim))
        self.bz = np.zeros(latent_dim)
        self.w2 = rng.normal(scale=scale3, size=(latent_dim, hidden_dim))
        self.b2 = np.zeros(hidden_dim)
        self.wout = rng.normal(scale=scale4, size=(hidden_dim, input_dim))
        self.bout = np.zeros(input_dim)

        self._adam_m = {name: np.zeros_like(value) for name, value in self.params().items()}
        self._adam_v = {name: np.zeros_like(value) for name, value in self.params().items()}
        self._adam_t = 0

    def params(self) -> dict[str, np.ndarray]:
        return {
            "w1": self.w1,
            "b1": self.b1,
            "wz": self.wz,
            "bz": self.bz,
            "w2": self.w2,
            "b2": self.b2,
            "wout": self.wout,
            "bout": self.bout,
        }

    @staticmethod
    def _linear(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
        with np.errstate(all="ignore"):
            y = x @ w + b
        return np.nan_to_num(y, nan=0.0, posinf=20.0, neginf=-20.0)

    def encode(self, x: np.ndarray) -> np.ndarray:
        h1 = np.tanh(np.clip(self._linear(x, self.w1, self.b1), -20.0, 20.0))
        return self._linear(h1, self.wz, self.bz)

    def decode(self, z: np.ndarray) -> np.ndarray:
        h2 = np.tanh(np.clip(self._linear(z, self.w2, self.b2), -20.0, 20.0))
        return self._linear(h2, self.wout, self.bout)

    def reconstruct(self, x: np.ndarray) -> np.ndarray:
        return self.decode(self.encode(x))

    def train(
        self,
        x: np.ndarray,
        rng: np.random.Generator,
        epochs: int = 120,
        batch_size: int = 256,
        lr: float = 2e-3,
    ) -> list[float]:
        losses = []
        n = len(x)
        for epoch in range(epochs):
            order = rng.permutation(n)
            for start in range(0, n, batch_size):
                batch = x[order[start : start + batch_size]]
                h1 = np.tanh(np.clip(self._linear(batch, self.w1, self.b1), -20.0, 20.0))
                z = self._linear(h1, self.wz, self.bz)
                h2 = np.tanh(np.clip(self._linear(z, self.w2, self.b2), -20.0, 20.0))
                out = self._linear(h2, self.wout, self.bout)

                diff = out - batch
                dout = (2.0 / (len(batch) * batch.shape[1])) * diff
                grads = {}
                with np.errstate(all="ignore"):
                    grads["wout"] = h2.T @ dout
                    grads["bout"] = np.sum(dout, axis=0)
                    dh2 = (dout @ self.wout.T) * (1.0 - h2 * h2)
                    grads["w2"] = z.T @ dh2
                    grads["b2"] = np.sum(dh2, axis=0)
                    dz = dh2 @ self.w2.T
                    grads["wz"] = h1.T @ dz
                    grads["bz"] = np.sum(dz, axis=0)
                    dh1 = (dz @ self.wz.T) * (1.0 - h1 * h1)
                    grads["w1"] = batch.T @ dh1
                    grads["b1"] = np.sum(dh1, axis=0)
                self._adam_step(grads, lr=lr)

            if epoch % 10 == 0 or epoch == epochs - 1:
                recon = self.reconstruct(x)
                losses.append(float(np.mean(np.sum((recon - x) ** 2, axis=1))))
        return losses

    def _adam_step(self, grads: dict[str, np.ndarray], lr: float) -> None:
        self._adam_t += 1
        beta1, beta2 = 0.9, 0.999
        for name, param in self.params().items():
            grad = np.nan_to_num(grads[name], nan=0.0, posinf=0.0, neginf=0.0)
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 5.0:
                grad = grad * (5.0 / grad_norm)
            grad = np.clip(grad, -1.0, 1.0)
            self._adam_m[name] = beta1 * self._adam_m[name] + (1.0 - beta1) * grad
            self._adam_v[name] = beta2 * self._adam_v[name] + (1.0 - beta2) * (grad * grad)
            m_hat = self._adam_m[name] / (1.0 - beta1**self._adam_t)
            v_hat = self._adam_v[name] / (1.0 - beta2**self._adam_t)
            param -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
            np.clip(param, -20.0, 20.0, out=param)


class VQStyleTokenizer:
    def __init__(self, num_codes: int, rng: np.random.Generator):
        self.num_codes = num_codes
        self.rng = rng
        self.centers = None

    def fit(self, z: np.ndarray) -> None:
        self.centers, _, _ = kmeans_fit(z, self.num_codes, self.rng, n_iter=40, n_init=2)

    def tokenize(self, z: np.ndarray) -> np.ndarray:
        return kmeans_predict(z, self.centers)

    def reconstruct(self, codes: np.ndarray) -> np.ndarray:
        return self.centers[codes]


class FSQStyleTokenizer:
    def __init__(self, levels: int):
        levels_array = np.asarray(levels if np.ndim(levels) > 0 else [levels], dtype=int)
        self.levels = levels_array
        self.num_codes = int(np.prod(levels_array))
        self.low = None
        self.high = None

    def fit(self, z: np.ndarray) -> None:
        if len(self.levels) == 1:
            self.levels = np.repeat(self.levels[0], z.shape[1])
        if len(self.levels) != z.shape[1]:
            raise ValueError(f"FSQ levels length {len(self.levels)} does not match latent dim {z.shape[1]}")
        self.num_codes = int(np.prod(self.levels))
        self.low = np.percentile(z, 1.0, axis=0)
        self.high = np.percentile(z, 99.0, axis=0)
        self.high = np.maximum(self.high, self.low + 1e-5)

    def _indices(self, z: np.ndarray) -> np.ndarray:
        scaled = (z - self.low) / (self.high - self.low)
        scaled = np.clip(scaled, 0.0, 1.0)
        return np.rint(scaled * (self.levels[None, :] - 1)).astype(int)

    def tokenize(self, z: np.ndarray) -> np.ndarray:
        idx = self._indices(z)
        codes = np.zeros(len(z), dtype=int)
        for d in range(idx.shape[1]):
            codes = codes * self.levels[d] + idx[:, d]
        return codes

    def reconstruct(self, codes: np.ndarray) -> np.ndarray:
        latent_dim = len(self.low)
        idx = np.zeros((len(codes), latent_dim), dtype=int)
        c = codes.copy()
        for d in range(latent_dim - 1, -1, -1):
            idx[:, d] = c % self.levels[d]
            c //= self.levels[d]
        scaled = idx / np.maximum(self.levels[None, :] - 1, 1)
        return self.low + scaled * (self.high - self.low)


def fsq_mixed_levels_for_code_count(latent_dim: int, code_count: int) -> list[int]:
    """Mixed-radix FSQ levels with product equal to code_count for powers of two."""
    exponent = int(round(math.log2(code_count)))
    if 2**exponent != code_count:
        raise ValueError("mixed FSQ helper expects power-of-two code counts")
    if exponent < latent_dim:
        raise ValueError("code_count too small for at least 2 levels per latent dimension")
    levels = [2] * latent_dim
    remaining = exponent - latent_dim
    dim = latent_dim - 1
    while remaining > 0:
        levels[dim] *= 2
        remaining -= 1
        dim = (dim - 1) % latent_dim
    return levels


class LGQStyleTokenizer:
    """Learned-metric quantizer: whiten latent geometry, then learn k-means cells."""

    def __init__(self, num_codes: int, rng: np.random.Generator):
        self.num_codes = num_codes
        self.rng = rng
        self.mean = None
        self.whitener = None
        self.dewhitener = None
        self.centers_white = None

    def fit(self, z: np.ndarray) -> None:
        self.mean = np.mean(z, axis=0)
        centered = z - self.mean
        cov = centered.T @ centered / max(len(z) - 1, 1)
        vals, vecs = np.linalg.eigh(cov + 1e-5 * np.eye(z.shape[1]))
        vals = np.maximum(vals, 1e-5)
        self.whitener = vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T
        self.dewhitener = np.linalg.pinv(self.whitener)
        z_white = self._to_white(z)
        self.centers_white, _, _ = kmeans_fit(z_white, self.num_codes, self.rng, n_iter=40, n_init=2)

    def _to_white(self, z: np.ndarray) -> np.ndarray:
        with np.errstate(all="ignore"):
            y = (z - self.mean) @ self.whitener
        return np.nan_to_num(y, nan=0.0, posinf=20.0, neginf=-20.0)

    def _from_white(self, y: np.ndarray) -> np.ndarray:
        with np.errstate(all="ignore"):
            z = y @ self.dewhitener + self.mean
        return np.nan_to_num(z, nan=0.0, posinf=20.0, neginf=-20.0)

    def tokenize(self, z: np.ndarray) -> np.ndarray:
        return kmeans_predict(self._to_white(z), self.centers_white)

    def reconstruct(self, codes: np.ndarray) -> np.ndarray:
        return self._from_white(self.centers_white[codes])


def tokenizer_reconstruction_error(tokenizer, z_test: np.ndarray) -> float:
    codes = tokenizer.tokenize(z_test)
    recon = tokenizer.reconstruct(codes)
    return float(np.mean(np.sum((z_test - recon) ** 2, axis=1)))


def evaluate_tokenizer(
    tokenizer,
    latent_trajectories: np.ndarray,
    latent_dirs: np.ndarray,
    angles: np.ndarray,
    theta: float = math.pi / 3.0,
    alpha: float = 0.35,
) -> dict[str, float]:
    num_bases, num_angles, _, _ = latent_trajectories.shape
    collapse_values = []
    instability_values = []
    diversity_values = []
    angular_errors = []
    hamming_values = []
    strong_passes = []

    for b in range(num_bases):
        codes_by_angle = []
        for a_idx in range(num_angles):
            traj = latent_trajectories[b, a_idx]
            codes = tokenizer.tokenize(traj)
            recon = tokenizer.reconstruct(codes)
            codes_by_angle.append(codes)

            collapse_values.append(float(np.mean(codes == codes[len(codes) // 2])))
            instability_values.append(float(np.mean(codes[1:] != codes[:-1])))
            diversity_values.append(float(len(np.unique(codes)) / len(codes)))

            decoded_dir = principal_direction(recon)
            if decoded_dir is None:
                angular_errors.append(math.pi / 2.0)
            else:
                target_dir = latent_dirs[b, a_idx]
                if np.dot(decoded_dir, target_dir) < 0:
                    decoded_dir = -decoded_dir
                angular_errors.append(math.acos(float(np.clip(np.dot(decoded_dir, target_dir), -1.0, 1.0))))

        for i in range(num_angles):
            for j in range(i + 1, num_angles):
                angle_sep = abs(float(angles[i] - angles[j])) % (2.0 * math.pi)
                angle_sep = min(angle_sep, 2.0 * math.pi - angle_sep)
                if angle_sep >= theta:
                    d_h = hamming_distance(codes_by_angle[i], codes_by_angle[j])
                    hamming_values.append(d_h)
                    strong_passes.append(float(d_h >= alpha))

    return {
        "collapse_index": float(np.mean(collapse_values)),
        "instability_index": float(np.mean(instability_values)),
        "code_diversity": float(np.mean(diversity_values)),
        "angular_error": float(np.mean(angular_errors)),
        "mean_hamming": float(np.mean(hamming_values)),
        "strong_pass_rate": float(np.mean(strong_passes)),
    }


def run_experiment(
    output_dir: str = "./results/exp5",
    latent_dim: int = 5,
    seeds: list[int] | None = None,
):
    os.makedirs(output_dir, exist_ok=True)
    seeds = [0, 42, 123] if seeds is None else seeds
    quick = len(seeds) == 1

    datasets = ["swiss_roll", "torus"]
    train_n = 8000
    test_n = 2400
    base_n = 140
    angles = np.linspace(0.0, 2.0 * math.pi, 12, endpoint=False)
    num_steps = 25
    step_radius_by_dataset = {"swiss_roll": 0.34, "torus": 0.22}

    code_sweep = [32, 64, 128, 256, 512]
    if quick:
        train_n = 2500
        test_n = 800
        base_n = 50
        angles = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
        num_steps = 15
        code_sweep = [32, 128, 512]

    configs = []
    for k in code_sweep:
        configs.append(("VQ", k, lambda rng, k=k: VQStyleTokenizer(k, rng)))
        configs.append(("LGQ-style", k, lambda rng, k=k: LGQStyleTokenizer(k, rng)))
        fsq_levels = fsq_mixed_levels_for_code_count(latent_dim, k)
        configs.append(("FSQ", k, lambda rng, levels=fsq_levels: FSQStyleTokenizer(levels)))

    print("=" * 98)
    print("Experiment 5: learned-tokenizer directional diagnostic")
    print("=" * 98)
    print(f"latent_dim={latent_dim}, seeds={seeds}")
    print(f"datasets={datasets}")
    print(f"train_n={train_n}, test_n={test_n}, base_n={base_n}")

    results = {}
    ae_losses = []
    for seed in seeds:
        print(f"\nseed={seed}")
        for dataset_idx, dataset in enumerate(datasets):
            rng = make_rng(seed + 1729 * dataset_idx)
            print(f" dataset={dataset}")
            train_params = sample_manifold_params(dataset, train_n, rng)
            test_params = sample_manifold_params(dataset, test_n, rng)
            train_raw = manifold_from_params(dataset, train_params)
            test_raw = manifold_from_params(dataset, test_params)
            standardizer = fit_standardizer(train_raw)
            train_x = standardizer.transform(train_raw)
            test_x = standardizer.transform(test_raw)

            ae = NumpyAutoencoder(input_dim=3, latent_dim=latent_dim, hidden_dim=96, rng=rng)
            losses = ae.train(train_x, rng, epochs=160, batch_size=256, lr=8e-4)
            ae_losses.append(losses[-1])
            z_train = ae.encode(train_x)
            z_test = ae.encode(test_x)
            print(f"  autoencoder train recon={losses[-1]:.4f}")

            base_params = sample_manifold_params(dataset, base_n, rng, base_region=True)
            input_traj, _ = make_directional_trajectories(
                base_params,
                angles,
                step_radius=step_radius_by_dataset[dataset],
                num_steps=num_steps,
                manifold=dataset,
                standardizer=standardizer,
            )
            flat_input = input_traj.reshape(-1, input_traj.shape[-1])
            flat_latent = ae.encode(flat_input)
            latent_traj = flat_latent.reshape(input_traj.shape[0], input_traj.shape[1], input_traj.shape[2], latent_dim)
            latent_dirs = np.zeros((input_traj.shape[0], input_traj.shape[1], latent_dim))
            for b in range(input_traj.shape[0]):
                for a_idx in range(input_traj.shape[1]):
                    direction = principal_direction(latent_traj[b, a_idx])
                    if direction is None:
                        direction = np.zeros(latent_dim)
                        direction[0] = 1.0
                    latent_dirs[b, a_idx] = direction / max(np.linalg.norm(direction), 1e-12)

            for family, effective_codes, factory in configs:
                tokenizer_rng = make_rng(seed + effective_codes + 1009 * dataset_idx + (17 if family == "LGQ-style" else 0))
                tokenizer = factory(tokenizer_rng)
                tokenizer.fit(z_train)
                rec_err = tokenizer_reconstruction_error(tokenizer, z_test)
                diag = evaluate_tokenizer(tokenizer, latent_traj, latent_dirs, angles)
                key = (dataset, family, effective_codes)
                results.setdefault(key, {"latent_reconstruction": []})
                results[key]["latent_reconstruction"].append(rec_err)
                for metric, value in diag.items():
                    results[key].setdefault(metric, []).append(value)
                print(
                    f"  {family:9s} codes={effective_codes:5d} "
                    f"latent_rec={rec_err:.4f} strong={diag['strong_pass_rate']:.3f} "
                    f"collapse={diag['collapse_index']:.3f}"
                )

    summary = [
        "Experiment 5: learned-tokenizer directional diagnostic",
        "=" * 108,
        f"Autoencoder final train reconstruction: {mean_ci(ae_losses)[0]:.4f} +/- {mean_ci(ae_losses)[1]:.4f}",
        "Collapse index is the fraction of a latent trajectory assigned to its midpoint code.",
        "Instability index is the fraction of consecutive trajectory steps with a code change.",
        "Strong pass rate is same-base symbolic separation for angularly separated latent directions.",
        "",
        f"{'dataset':12s} {'tokenizer':12s} {'codes':>7s} {'latent rec':>16s} {'strong pass':>16s} {'collapse':>16s} {'angular err':>16s}",
        f"{'':12s} {'':12s} {'':>7s} {'mean ham':>16s} {'instability':>16s} {'diversity':>16s} {'':>16s}",
        "-" * 108,
    ]
    for dataset, family, codes in sorted(results.keys(), key=lambda x: (x[0], x[1], x[2])):
        rec_m, rec_ci = mean_ci(results[(dataset, family, codes)]["latent_reconstruction"])
        strong_m, strong_ci = mean_ci(results[(dataset, family, codes)]["strong_pass_rate"])
        collapse_m, collapse_ci = mean_ci(results[(dataset, family, codes)]["collapse_index"])
        angular_m, angular_ci = mean_ci(results[(dataset, family, codes)]["angular_error"])
        hamming_m, hamming_ci = mean_ci(results[(dataset, family, codes)]["mean_hamming"])
        instability_m, instability_ci = mean_ci(results[(dataset, family, codes)]["instability_index"])
        diversity_m, diversity_ci = mean_ci(results[(dataset, family, codes)]["code_diversity"])
        summary.append(
            f"{dataset:12s} {family:12s} {codes:7d} {rec_m:7.4f} +/- {rec_ci:<6.4f} "
            f"{strong_m:7.4f} +/- {strong_ci:<6.4f} "
            f"{collapse_m:7.4f} +/- {collapse_ci:<6.4f} "
            f"{angular_m:7.4f} +/- {angular_ci:<6.4f}"
        )
        summary.append(
            f"{'':12s} {'':12s} {'':7s} {hamming_m:7.4f} +/- {hamming_ci:<6.4f} "
            f"{instability_m:7.4f} +/- {instability_ci:<6.4f} "
            f"{diversity_m:7.4f} +/- {diversity_ci:<6.4f}"
        )
    summary.extend(
        [
            "",
            "Interpretation:",
            "- This is a real learned-latent tokenizer diagnostic: the latent map and codebooks are fitted.",
            "- VQ-style and LGQ-style use learned nearest-neighbor cells; FSQ uses structured scalar cells.",
            "- Directional faithfulness is evaluated on latent trajectories, not only on pointwise reconstruction.",
        ]
    )
    write_text(os.path.join(output_dir, "exp5_summary.txt"), "\n".join(summary) + "\n")

    def family_series(metric: str):
        series = []
        colors = {"VQ": "#0072B2", "FSQ": "#D55E00", "LGQ-style": "#009E73"}
        markers = {"VQ": "o", "FSQ": "s", "LGQ-style": "^"}
        for dataset in datasets:
            for family in ["VQ", "FSQ", "LGQ-style"]:
                keys = sorted([key for key in results if key[0] == dataset and key[1] == family], key=lambda x: x[2])
                series.append(
                    {
                        "facet": dataset,
                        "label": family,
                        "x": [key[2] for key in keys],
                        "y": [mean_ci(results[key][metric])[0] for key in keys],
                        "yerr": [mean_ci(results[key][metric])[1] for key in keys],
                        "color": colors[family],
                        "marker": markers[family],
                    }
                )
        return series

    write_pdf_facet_line_plot(
        os.path.join(output_dir, "exp5_strong_pass.pdf"),
        family_series("strong_pass_rate"),
        facet_order=datasets,
        xlabel="effective code count",
        ylabel="strong pass rate",
        title="Learned-Tokenizer Directional Separation",
        subtitle="Same effective code counts for VQ, FSQ, and LGQ-style tokenizers",
        logx=True,
        y_limits=(0.0, 1.0),
    )
    write_pdf_facet_line_plot(
        os.path.join(output_dir, "exp5_collapse.pdf"),
        family_series("collapse_index"),
        facet_order=datasets,
        xlabel="effective code count",
        ylabel="collapse index",
        title="Learned-Tokenizer Trajectory Collapse",
        subtitle="Lower collapse means trajectories use more than their midpoint code",
        logx=True,
        y_limits=(0.0, 1.0),
    )
    write_pdf_facet_line_plot(
        os.path.join(output_dir, "exp5_angular.pdf"),
        family_series("angular_error"),
        facet_order=datasets,
        xlabel="effective code count",
        ylabel="radians",
        title="Decoded Latent-Trajectory Angular Error",
        subtitle="Error bars are 95% confidence intervals over seeds",
        logx=True,
    )
    write_pdf_facet_line_plot(
        os.path.join(output_dir, "exp5_instability.pdf"),
        family_series("instability_index"),
        facet_order=datasets,
        xlabel="effective code count",
        ylabel="boundary crossing rate",
        title="Symbolic Trajectory Instability",
        subtitle="Fraction of consecutive latent-trajectory steps that change token",
        logx=True,
        y_limits=(0.0, 1.0),
    )
    write_pdf_facet_line_plot(
        os.path.join(output_dir, "exp5_diversity.pdf"),
        family_series("code_diversity"),
        facet_order=datasets,
        xlabel="effective code count",
        ylabel="unique-token fraction",
        title="Trajectory Code Diversity",
        subtitle="Higher diversity means a local path realizes more distinct symbols",
        logx=True,
        y_limits=(0.0, 1.0),
    )
    write_pdf_facet_line_plot(
        os.path.join(output_dir, "exp5_latent_reconstruction.pdf"),
        family_series("latent_reconstruction"),
        facet_order=datasets,
        xlabel="effective code count",
        ylabel="latent reconstruction MSE",
        title="Tokenizer Pointwise Latent Reconstruction",
        subtitle="Pointwise quality is shown beside trajectory diagnostics",
        logx=True,
        logy=True,
    )
    print(f"\nSaved {os.path.join(output_dir, 'exp5_summary.txt')}")
    return results


if __name__ == "__main__":
    run_experiment()
