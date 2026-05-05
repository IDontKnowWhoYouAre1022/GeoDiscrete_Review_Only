# GeoDiscrete_Review_Only
Anonymous Repo for Peer Review: Directional Faithfulness in Learned Discretization for Geometric Representation Learning

The suite is designed as controlled evidence for the theory, not as a large
tokenizer benchmark.  Each experiment measures a quantity that appears in the
paper: tube-union covering, same-base symbolic trajectory separation,
volume-to-cover transfer, additivity across separated charts, or learned-latent
tokenizer collapse.

## Files

| File | Role |
| --- | --- |
| `utils.py` | Shared geometry, k-means, trajectory, and Matplotlib/Seaborn plotting utilities |
| `exp1_tube_covering.py` | 3D tube-union cover scaling plus planar negative control |
| `exp2_reconstruction_vs_directional.py` | Reconstruction vs same-base directional faithfulness |
| `exp3_transfer_theorem_4d.py` | Controlled 4D volume-to-cover transfer |
| `exp4_additivity.py` | Additivity across separated local certificates |
| `exp5_tokenizer_diagnostic.py` | Learned-latent VQ/FSQ/LGQ-style tokenizer diagnostic |
| `run_all_experiments.py` | Master runner |
| `run_experiments.sh` | Shell wrapper |

## Requirements

The experiments use NumPy for computation and Matplotlib/Seaborn for figures:

```bash
pip install numpy matplotlib seaborn
```

The suite intentionally avoids SciPy, scikit-learn, and PyTorch.  Figures are
written as publication-ready PDF and PNG files.

## Running

From this directory:

```bash
python run_all_experiments.py --output-dir ./results
python run_all_experiments.py --quick --output-dir ./results_quick
python run_all_experiments.py --exp 5 --output-dir ./results_exp5
```

or:

```bash
./run_experiments.sh --quick
```

## Output

Each experiment writes:

- `exp*_summary.txt`: numerical table and interpretation.
- `*.pdf` and `*.png`: Matplotlib/Seaborn figure files.

Running all experiments also writes `SUMMARY_REPORT.md`.

## Experiment Framing

### Experiment 1

Tests the base covering mechanism in 3D.  Direction-rich tube families should
produce cover exponents near 3, while a planar control should produce an
exponent closer to 2.  This supports the necessity of 3D directional richness.

### Experiment 2

Uses equal-length tangent trajectories at the same base point on two manifolds:
Swiss roll and torus.  It reports pointwise reconstruction MSE alongside strong
directional pass rate, Hamming separation, decoded angular error, code
diversity, and boundary crossing.  This is the empirical version of the
reconstruction/directional-faithfulness gap.

### Experiment 3

Constructs layered 4D tube families with controlled effective exponents
`s = 3, 3.25, 3.5, 3.75, 4`.  It verifies the conversion in Theorem 3: if volume
scales like `delta^(4-s)`, the cover count scales like `delta^(-s)`.  It does
not claim to prove or numerically certify unresolved 4D Kakeya estimates.

### Experiment 4

Embeds identical local tube certificates into separated coordinates of a
higher-dimensional ambient space.  The sampled cover count should grow linearly
with the number of separated charts, matching Proposition 4.

### Experiment 5

Trains a small NumPy autoencoder on Swiss-roll and torus datasets to obtain
learned continuous latents, then fits:

- VQ-style k-means codebooks,
- FSQ-style scalar grids,
- LGQ-style learned-metric k-means after whitening.

It evaluates symbolic trajectories in the learned latent space.  FSQ now uses
five level settings, matching the number of VQ/LGQ code-count sweeps.  Metrics
include collapse index, instability index, strong pass rate, Hamming separation,
angular error, and latent reconstruction error.

