#!/usr/bin/env python
"""Run the directional faithfulness experiment suite."""

from __future__ import annotations

import argparse
import os
import time
from datetime import datetime

from exp1_tube_covering import run_experiment as run_exp1
from exp2_reconstruction_vs_directional import run_experiment as run_exp2
from exp3_transfer_theorem_4d import run_experiment as run_exp3
from exp4_additivity import run_experiment as run_exp4
from exp5_tokenizer_diagnostic import run_experiment as run_exp5
from utils import write_text


def create_summary_report(output_dir: str, completed: list[int]) -> str:
    report = [
        "# Directional Faithfulness Experimental Summary",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "This suite contains five theory-facing experiments.  The wording is deliberately conservative:",
        "the synthetic studies are controlled geometric stress tests, and the tokenizer study is a",
        "learned-latent diagnostic rather than a large-scale benchmark.",
        "",
        "## Experiments",
        "",
        "1. Exp 1: 3D tube-union cover scaling.  Tests the volume-to-cover mechanism and a planar negative control.",
        "2. Exp 2: Reconstruction versus same-base directional faithfulness on Swiss-roll trajectories.",
        "3. Exp 3: Controlled 4D volume-to-cover transfer for target exponents s=3, 3.5, and 4.",
        "4. Exp 4: Additivity across separated local tube certificates.",
        "5. Exp 5: Learned-latent tokenizer diagnostic for VQ-style, FSQ-style, and LGQ-style quantizers.",
        "",
        "## Completed",
        "",
    ]
    for idx in completed:
        report.append(f"- Experiment {idx}: see `exp{idx}/exp{idx}_summary.txt` and PDF/PNG figures in that directory.")
    report.extend(
        [
            "",
            "## Suggested paper framing",
            "",
            "Use Exp 1-2 in the main text if space permits: they connect the theorems to visible finite-scale",
            "geometry and show the reconstruction/directional gap.  Put Exp 3-4 in the appendix as controlled",
            "mechanism checks.  Use Exp 5 as the ML relevance diagnostic, with careful wording: it shows",
            "directional collapse and symbolic trajectory distortion in learned latent tokenizers.",
            "",
            "The experiments use NumPy for computation and Matplotlib/Seaborn for publication-ready",
            "PDF/PNG figures.",
        ]
    )
    path = os.path.join(output_dir, "SUMMARY_REPORT.md")
    write_text(path, "\n".join(report) + "\n")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run directional faithfulness experiments")
    parser.add_argument("--output-dir", default="./results", help="Directory for experiment outputs")
    parser.add_argument("--quick", action="store_true", help="Use one seed for a faster smoke run")
    parser.add_argument("--exp", type=int, choices=[1, 2, 3, 4, 5], help="Run only one experiment")
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    seeds = [0] if args.quick else None

    runners = {
        1: run_exp1,
        2: run_exp2,
        3: run_exp3,
        4: run_exp4,
        5: run_exp5,
    }
    selected = [args.exp] if args.exp is not None else [1, 2, 3, 4, 5]

    print("=" * 96)
    print("DIRECTIONAL FAITHFULNESS EXPERIMENTS")
    print("=" * 96)
    print(f"output_dir={output_dir}")
    print(f"quick={args.quick}")

    start = time.time()
    completed = []
    for idx in selected:
        print("\n" + "=" * 96)
        print(f"RUNNING EXPERIMENT {idx}")
        print("=" * 96)
        exp_dir = os.path.join(output_dir, f"exp{idx}")
        runners[idx](output_dir=exp_dir, seeds=seeds)
        completed.append(idx)

    if args.exp is None:
        summary_path = create_summary_report(output_dir, completed)
        print(f"\nSaved {summary_path}")

    elapsed = time.time() - start
    print("\n" + "=" * 96)
    print("DONE")
    print("=" * 96)
    print(f"elapsed={elapsed:.1f}s")
    print(f"outputs={output_dir}")


if __name__ == "__main__":
    main()
