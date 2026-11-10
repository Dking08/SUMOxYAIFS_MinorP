"""CLI tool to train offline offloading models (Ridge, Random Forest, Markov, Decision Tree)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.algorithms.offline_trainer import (
    load_dataset_from_csv,
    load_dataset_from_run_dir,
    train_offline_models,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train offline models from simulation logs or dataset CSV")
    parser.add_argument("--csv", help="Path to offline_dataset.csv file")
    parser.add_argument("--run-dir", help="Path to run directory (e.g. sumoXyafs/results/exp_baseline_03)")
    parser.add_argument(
        "--out-dir",
        default=str(ROOT / "trained_models"),
        help="Directory to save trained model parameters (default: sumoXyafs/trained_models)",
    )
    args = parser.parse_args()

    if args.csv:
        records = load_dataset_from_csv(args.csv)
    elif args.run_dir:
        records = load_dataset_from_run_dir(args.run_dir)
    else:
        # Default: try to train on existing baseline run
        baseline = ROOT / "results" / "exp_baseline_03"
        if baseline.exists():
            print(f"[INFO] No input specified; using existing baseline run at {baseline}")
            records = load_dataset_from_run_dir(baseline)
        else:
            raise ValueError("Please provide either --csv <path> or --run-dir <path>")

    print(f"[INFO] Loaded {len(records)} records for offline training.")
    train_offline_models(records, output_dir=args.out_dir)
    print(f"[SUCCESS] Trained models saved to {args.out_dir}")


if __name__ == "__main__":
    main()
