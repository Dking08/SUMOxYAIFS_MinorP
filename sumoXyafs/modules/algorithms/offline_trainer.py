"""Offline training orchestrator for supervised and statistical algorithms (3, 4, 5, 6)."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from modules.algorithms.decision_tree import DecisionTreeSelector
from modules.algorithms.markov_chain import MarkovChainSelector
from modules.algorithms.random_forest import RandomForestSelector
from modules.algorithms.ridge_regression import RidgeRegressionSelector


def load_dataset_from_csv(csv_path: Path | str) -> list[dict]:
    """Load records from a DatasetLogger CSV file."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        return list(reader)


def load_dataset_from_run_dir(run_dir: Path | str) -> list[dict]:
    """Parse historical simulation run directory (decisions.csv + telemetry.csv)."""
    dir_path = Path(run_dir)
    analysis_dir = dir_path / "analysis" if (dir_path / "analysis").exists() else dir_path

    decisions_csv = analysis_dir / "decisions.csv"
    telemetry_csv = analysis_dir / "telemetry.csv"

    if not decisions_csv.exists():
        raise FileNotFoundError(f"Could not find decisions.csv in {run_dir}")

    with decisions_csv.open("r", encoding="utf-8", newline="") as fh:
        decisions = list(csv.DictReader(fh))

    telemetry_by_key: dict[tuple[int, str], dict] = {}
    if telemetry_csv.exists():
        with telemetry_csv.open("r", encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                try:
                    s = int(row.get("step", 0))
                    v = str(row.get("vehicle_id", ""))
                    telemetry_by_key[(s, v)] = row
                except Exception:
                    continue

    records: list[dict] = []
    for d in decisions:
        try:
            step = int(d.get("step", 0))
            vid = str(d.get("vehicle_id", ""))
            obs = str(d.get("observed_rsu", ""))
            owner = str(d.get("owner_rsu", ""))
            exec_rsu = str(d.get("execution_rsu", ""))
            action = str(d.get("action", ""))
            est_finish = int(d.get("estimated_finish_step", step + 2))
            deadline = int(d.get("deadline_step", step + 2))
            dist = float(d.get("distance_to_exec_rsu_m", 500.0))
            slack = deadline - est_finish
            deadline_met = int(est_finish <= deadline)
            latency_s = float(max(1, est_finish - step))

            telem = telemetry_by_key.get((step, vid), {})
            speed = float(telem.get("speed_mps", 10.0))

            records.append(
                {
                    "step": step,
                    "vehicle_id": vid,
                    "observed_rsu": obs,
                    "owner_rsu": owner,
                    "execution_rsu": exec_rsu,
                    "action_rsu": exec_rsu,
                    "action": action,
                    "norm_distance_to_action": min(1.0, dist / 1500.0),
                    "queue_depth_action": 1,
                    "speed": speed,
                    "remaining_compute_s": 2.0,
                    "deadline_step": deadline,
                    "deadline_slack_steps": slack,
                    "latency_s": latency_s,
                    "deadline_met": deadline_met,
                    "reward": 1.0 if deadline_met else -1.0,
                }
            )
        except Exception:
            continue

    return records


def train_offline_models(
    records: list[dict],
    output_dir: Path | str | None = None,
) -> dict[str, object]:
    """Fit Ridge regression, Random forest, Markov chain, and Decision tree on dataset.

    Optionally saves the trained artifacts to output_dir.
    """
    out = {}

    print(f"[TRAIN] Fitting offline models on {len(records)} records...", flush=True)

    # 1. Ridge Regression (Algorithm 3)
    ridge = RidgeRegressionSelector()
    ridge.train_offline(records)
    out["ridge"] = ridge

    # 2. Random Forest Classifier (Algorithm 4)
    rf = RandomForestSelector()
    rf.train_offline(records)
    out["random_forest"] = rf

    # 3. Markov Chain (Algorithm 5)
    markov = MarkovChainSelector()
    markov.train_offline(records)
    out["markov"] = markov

    # 4. Decision Tree (Algorithm 6)
    dt = DecisionTreeSelector()
    dt.train_offline(records)
    out["decision_tree"] = dt

    if output_dir is not None:
        save_path = Path(output_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        ridge.save(save_path / "ridge_weights.json")
        rf.save(save_path / "random_forest_model.json")
        markov.save(save_path / "markov_transitions.json")
        dt.save(save_path / "decision_tree_rules.json")
        print(f"[TRAIN] Saved trained models to {save_path.resolve()}", flush=True)

    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train offline models 3, 4, 5, 6 from simulation logs")
    parser.add_argument("--csv", help="Path to DatasetLogger CSV file")
    parser.add_argument("--run-dir", help="Path to simulation run directory (e.g. results/exp_baseline_03)")
    parser.add_argument("--out-dir", default="trained_models", help="Output directory to save model weights")
    args = parser.parse_args()

    if args.csv:
        data = load_dataset_from_csv(args.csv)
    elif args.run_dir:
        data = load_dataset_from_run_dir(args.run_dir)
    else:
        # Default fallback to existing baseline run if available
        base = Path(__file__).resolve().parents[2] / "results" / "exp_baseline_03"
        if base.exists():
            print(f"[TRAIN] Using existing baseline run: {base}")
            data = load_dataset_from_run_dir(base)
        else:
            raise ValueError("Must provide either --csv or --run-dir")

    train_offline_models(data, output_dir=args.out_dir)
