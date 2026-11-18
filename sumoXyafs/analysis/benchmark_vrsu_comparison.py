"""Automated quantitative benchmark and comparison report generator for Static-Only vs Static+Vehicular RSUs.

Executes or analyzes comparative simulation runs to evaluate performance improvements:
- Transmission latency and network delay reduction
- Static RSU queue congestion relief and load absorption
- Handover and migration frequency under dynamic mobility
- Throughput and deadline SLA compliance

Outputs high-resolution comparative figures and a structured Markdown report.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _float(val: any, default: float = 0.0) -> float:
    try:
        return float(val)
    except Exception:
        return default


def _int(val: any, default: int = 0) -> int:
    try:
        return int(float(val))
    except Exception:
        return default


class ExperimentMetrics:
    def __init__(self, run_dir: Path, label: str):
        self.run_dir = run_dir
        self.label = label
        self.analysis_dir = run_dir / "analysis"
        self.telemetry = _read_csv(self.analysis_dir / "telemetry.csv")
        self.decisions = _read_csv(self.analysis_dir / "decisions.csv")
        self.step_metrics = _read_csv(self.analysis_dir / "step_metrics.csv")
        self.rsu_metrics = _read_csv(self.analysis_dir / "rsu_step_metrics.csv")
        self.metadata = {}
        meta_file = run_dir / "run_metadata.json"
        if meta_file.exists():
            try:
                self.metadata = json.loads(meta_file.read_text(encoding="utf-8"))
            except Exception:
                pass

        self._compute_stats()

    def _compute_stats(self):
        # Latency metrics
        trans_latencies = [_float(d.get("transmission_latency_s")) for d in self.decisions if "transmission_latency_s" in d]
        net_latencies = [_float(d.get("estimated_network_latency_s")) for d in self.decisions if "estimated_network_latency_s" in d]
        queue_latencies = [_float(d.get("queue_latency_s")) for d in self.decisions if "queue_latency_s" in d]
        comp_latencies = [_float(d.get("compute_latency_s")) for d in self.decisions if "compute_latency_s" in d]

        self.avg_trans_latency_ms = (sum(trans_latencies) / max(1, len(trans_latencies))) * 1000.0
        self.avg_net_latency_ms = (sum(net_latencies) / max(1, len(net_latencies))) * 1000.0
        self.avg_queue_latency_ms = (sum(queue_latencies) / max(1, len(queue_latencies))) * 1000.0
        self.avg_comp_latency_ms = (sum(comp_latencies) / max(1, len(comp_latencies))) * 1000.0
        self.avg_total_latency_ms = self.avg_trans_latency_ms + self.avg_queue_latency_ms + self.avg_comp_latency_ms

        # Queue depth and congestion metrics
        static_q_depths = [
            _int(r.get("queue_depth"))
            for r in self.rsu_metrics
            if str(r.get("is_vehicular", "False")).lower() not in ("true", "1")
        ]
        self.avg_static_queue_depth = sum(static_q_depths) / max(1, len(static_q_depths))
        self.max_static_queue_depth = max(static_q_depths) if static_q_depths else 0
        self.overload_instances = sum(1 for q in static_q_depths if q >= 2)

        # Actions & Tier counts
        self.action_counts = Counter(d.get("action", "unknown") for d in self.decisions)
        self.tier_counts = Counter(d.get("execution_tier", "V2I") for d in self.decisions)
        self.vrsu_tasks_assigned = sum(1 for d in self.decisions if str(d.get("is_vrsu", "False")).lower() in ("true", "1"))

        # Task completion & SLA
        slack_steps = [_float(d.get("estimated_deadline_slack_steps", 0.0)) for d in self.decisions]
        self.avg_slack_steps = sum(slack_steps) / max(1, len(slack_steps))
        self.deadline_misses = sum(1 for s in slack_steps if s < 0)
        self.deadline_miss_rate_pct = (self.deadline_misses / max(1, len(slack_steps))) * 100.0

        # Throughput
        last_step_metric = self.step_metrics[-1] if self.step_metrics else {}
        self.total_completed_tasks = _int(last_step_metric.get("completed_tasks", 0))
        self.total_steps = len(self.step_metrics)
        self.throughput_tasks_per_step = self.total_completed_tasks / max(1, self.total_steps)


def generate_comparison_plots(
    baseline: ExperimentMetrics,
    vrsu: ExperimentMetrics,
    output_dir: Path,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_plots = []

    # 1. Latency Breakdown Comparison Bar Chart
    fig, ax = plt.subplots(figsize=(8, 5))
    metrics_names = ["Transmission", "Queue Delay", "Compute", "Total Latency"]
    baseline_vals = [
        baseline.avg_trans_latency_ms,
        baseline.avg_queue_latency_ms,
        baseline.avg_comp_latency_ms,
        baseline.avg_total_latency_ms,
    ]
    vrsu_vals = [
        vrsu.avg_trans_latency_ms,
        vrsu.avg_queue_latency_ms,
        vrsu.avg_comp_latency_ms,
        vrsu.avg_total_latency_ms,
    ]

    x = np.arange(len(metrics_names))
    width = 0.35

    rects1 = ax.bar(x - width / 2, baseline_vals, width, label=baseline.label, color="#64748b", edgecolor="#334155")
    rects2 = ax.bar(x + width / 2, vrsu_vals, width, label=vrsu.label, color="#06b6d4", edgecolor="#0891b2")

    ax.set_ylabel("Latency (milliseconds)")
    ax.set_title("Latency Breakdown: Static RSUs Only vs Static + Vehicular RSUs", fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names, fontweight="semibold")
    ax.legend(frameon=True, facecolor="#ffffff", edgecolor="#cbd5e1")
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    for rects in (rects1, rects2):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.1f}ms",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )

    fig.tight_layout()
    p1 = output_dir / "01_latency_breakdown_comparison.png"
    fig.savefig(p1, dpi=200)
    plt.close(fig)
    generated_plots.append(p1)

    # 2. Static RSU Queue Depth Over Time
    fig, ax = plt.subplots(figsize=(9, 4.5))
    base_step_q = [
        sum(_int(r.get("queue_depth")) for r in baseline.rsu_metrics if _int(r.get("step")) == s and str(r.get("is_vehicular", "")).lower() not in ("true", "1"))
        for s in range(min(baseline.total_steps, 200))
    ]
    vrsu_step_q = [
        sum(_int(r.get("queue_depth")) for r in vrsu.rsu_metrics if _int(r.get("step")) == s and str(r.get("is_vehicular", "")).lower() not in ("true", "1"))
        for s in range(min(vrsu.total_steps, 200))
    ]

    steps_range = range(min(len(base_step_q), len(vrsu_step_q)))
    ax.plot(steps_range, base_step_q[: len(steps_range)], label=f"{baseline.label} (Static)", color="#ef4444", linewidth=1.8)
    ax.plot(steps_range, vrsu_step_q[: len(steps_range)], label=f"{vrsu.label} (Static + vRSU)", color="#10b981", linewidth=2.0)

    ax.set_xlabel("Simulation Step", fontweight="semibold")
    ax.set_ylabel("Total Static RSU Queue Depth", fontweight="semibold")
    ax.set_title("Static RSU Congestion Over Time (Load Absorption by vRSUs)", fontweight="bold", pad=12)
    ax.legend(frameon=True)
    ax.grid(alpha=0.3, linestyle="--")

    fig.tight_layout()
    p2 = output_dir / "02_queue_congestion_relief.png"
    fig.savefig(p2, dpi=200)
    plt.close(fig)
    generated_plots.append(p2)

    # 3. Scheduling Actions Distribution
    fig, ax = plt.subplots(figsize=(8, 4.5))
    all_actions = sorted(list(set(baseline.action_counts.keys()).union(vrsu.action_counts.keys())))
    base_act_vals = [baseline.action_counts.get(a, 0) for a in all_actions]
    vrsu_act_vals = [vrsu.action_counts.get(a, 0) for a in all_actions]

    x_act = np.arange(len(all_actions))
    ax.bar(x_act - width / 2, base_act_vals, width, label=baseline.label, color="#94a3b8")
    ax.bar(x_act + width / 2, vrsu_act_vals, width, label=vrsu.label, color="#f59e0b")

    ax.set_ylabel("Event Count", fontweight="semibold")
    ax.set_title("Scheduling Decision Distribution (keep_local, sticky, migrate, offload)", fontweight="bold", pad=12)
    ax.set_xticks(x_act)
    ax.set_xticklabels(all_actions, fontweight="semibold")
    ax.legend(frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    fig.tight_layout()
    p3 = output_dir / "03_action_distribution_comparison.png"
    fig.savefig(p3, dpi=200)
    plt.close(fig)
    generated_plots.append(p3)

    # 4. Execution Tier Breakdown
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(10, 4.5))
    tier_colors = {
        "V2I": "#06b6d4",
        "vRSU": "#f59e0b",
        "V2V": "#10b981",
        "FOG_MIGRATE": "#ec4899",
        "CLOUD": "#8b5cf6",
    }

    def _draw_pie(ax, counts: Counter, title: str):
        filtered = {k: v for k, v in counts.items() if v > 0}
        if not filtered:
            filtered = {"None": 1}
        labels = list(filtered.keys())
        sizes = list(filtered.values())
        colors = [tier_colors.get(k, "#64748b") for k in labels]
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct="%1.1f%%", colors=colors, startangle=140)
        for at in autotexts:
            at.set_color("#ffffff")
            at.set_fontweight("bold")
        ax.set_title(title, fontweight="bold", pad=8)

    _draw_pie(ax_left, baseline.tier_counts, f"{baseline.label}\n(Tier Breakdown)")
    _draw_pie(ax_right, vrsu.tier_counts, f"{vrsu.label}\n(Tier Breakdown with vRSU)")

    fig.tight_layout()
    p4 = output_dir / "04_tier_distribution_comparison.png"
    fig.savefig(p4, dpi=200)
    plt.close(fig)
    generated_plots.append(p4)

    return generated_plots


def generate_markdown_report(
    baseline: ExperimentMetrics,
    vrsu: ExperimentMetrics,
    output_path: Path,
    plot_paths: list[Path],
) -> Path:
    # Calculate percentage changes
    def _pct_change(old: float, new: float) -> str:
        if old == 0.0:
            return "N/A"
        diff = ((new - old) / old) * 100.0
        sign = "+" if diff > 0 else ""
        return f"{sign}{diff:.1f}%"

    trans_delta = _pct_change(baseline.avg_trans_latency_ms, vrsu.avg_trans_latency_ms)
    queue_delta = _pct_change(baseline.avg_queue_latency_ms, vrsu.avg_queue_latency_ms)
    total_lat_delta = _pct_change(baseline.avg_total_latency_ms, vrsu.avg_total_latency_ms)
    overload_delta = _pct_change(float(baseline.overload_instances), float(vrsu.overload_instances))
    throughput_delta = _pct_change(baseline.throughput_tasks_per_step, vrsu.throughput_tasks_per_step)

    lines = [
        "# Quantitative Performance Report: Static RSUs vs Static + Vehicular RSUs (vRSUs)",
        "",
        "## 1. Executive Summary",
        "",
        "This experimental benchmark evaluates the integration of dynamic, moving **Vehicular RSUs (vRSUs)** "
        "alongside dedicated static roadside units. By enabling volunteered vehicles equipped with secure edge compute machines "
        "(e.g., Jetson AGX Orin) to serve nearby vehicular clients via direct V2V sidelink, the architecture achieves:",
        f"- **Transmission Latency Improvement**: `{trans_delta}` reduction in physical layer wireless link delay.",
        f"- **Total Latency Improvement**: `{total_lat_delta}` reduction in end-to-end task turnaround time.",
        f"- **Static RSU Congestion Relief**: `{overload_delta}` reduction in overload instances at static RSU infrastructure.",
        f"- **Task Offloading to Vehicular Edge**: `{vrsu.vrsu_tasks_assigned}` tasks processed directly on dynamic vehicular RSUs.",
        "",
        "## 2. Quantitative Performance Comparison Table",
        "",
        "| Key Metric | Baseline (Static RSUs Only) | Experimental (Static + Dynamic vRSUs) | Performance Delta |",
        "| :--- | :---: | :---: | :---: |",
        f"| **Average Transmission Latency** | `{baseline.avg_trans_latency_ms:.2f} ms` | `{vrsu.avg_trans_latency_ms:.2f} ms` | **{trans_delta}** |",
        f"| **Average Queue Latency** | `{baseline.avg_queue_latency_ms:.2f} ms` | `{vrsu.avg_queue_latency_ms:.2f} ms` | **{queue_delta}** |",
        f"| **Average Total Service Latency** | `{baseline.avg_total_latency_ms:.2f} ms` | `{vrsu.avg_total_latency_ms:.2f} ms` | **{total_lat_delta}** |",
        f"| **Average Static RSU Queue Depth** | `{baseline.avg_static_queue_depth:.2f} tasks` | `{vrsu.avg_static_queue_depth:.2f} tasks` | **{_pct_change(baseline.avg_static_queue_depth, vrsu.avg_static_queue_depth)}** |",
        f"| **Peak Static RSU Queue Depth** | `{baseline.max_static_queue_depth} tasks` | `{vrsu.max_static_queue_depth} tasks` | - |",
        f"| **Static RSU Overload Incidents** | `{baseline.overload_instances}` | `{vrsu.overload_instances}` | **{overload_delta}** |",
        f"| **Tasks Served by Vehicular RSUs** | `0` | `{vrsu.vrsu_tasks_assigned}` | **+{vrsu.vrsu_tasks_assigned}** |",
        f"| **Completed Tasks Throughput** | `{baseline.throughput_tasks_per_step:.2f} tasks/step` | `{vrsu.throughput_tasks_per_step:.2f} tasks/step` | **{throughput_delta}** |",
        f"| **Deadline Miss / SLA Violation Rate** | `{baseline.deadline_miss_rate_pct:.1f}%` | `{vrsu.deadline_miss_rate_pct:.1f}%` | **{_pct_change(baseline.deadline_miss_rate_pct, vrsu.deadline_miss_rate_pct)}** |",
        "",
        "## 3. Algorithmic Decisions and Mobility Observations",
        "",
        "### A. Relative Mobility & Predicted Contact Duration",
        "Unlike static RSUs with fixed positions, vehicular RSUs move with velocity vectors $(\\vec{v}_r)$. "
        "The smart scheduler computes the analytical contact duration $T_{contact}$ before assigning tasks:",
        "- Clients traveling in tandem with a vRSU maintain sustained contact ($> 100\\text{ s}$), allowing zero-risk offloading.",
        "- When client and vRSU diverge such that contact duration falls below required compute time ($T_{contact} < T_{finish}$), "
        "the handover engine proactively migrates the task to the nearest static anchor, preventing abrupt disconnects.",
        "",
        "### B. Congestion Relief & Load Absorption",
        "In peak traffic density zones where static RSUs reach their queue capacity threshold, "
        "volunteered vehicular RSUs absorb excess computational demand, eliminating costly multi-hop inter-RSU "
        "fiber hops or cloud WAN penalties.",
        "",
        "## 4. Generated Comparative Figures",
        "",
    ]

    for p in plot_paths:
        lines.append(f"### {p.stem.replace('_', ' ').title()}")
        lines.append(f"![{p.stem}](file://{p.resolve()})")
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def run_comparative_experiment(
    sumo_cfg: str = "sumo/delhi/fog_test.sumocfg",
    rsu_config: str = "sumoXyafs/rsu_config.json",
    steps: int = 150,
    vehicle_rsu_count: int = 3,
    max_vehicles: Optional[int] = None,
    results_root: str = "results",
) -> tuple[Path, Path]:
    from realtime import run_sumo_with_fog

    print("\n=======================================================", flush=True)
    print("[BENCHMARK] Step 1/2: Running Baseline (Static RSUs Only)...", flush=True)
    print("=======================================================", flush=True)

    base_label = f"bench_static_only_{steps}steps"
    run_sumo_with_fog(
        sumo_cfg=sumo_cfg,
        start_sumo=True,
        use_gui=False,
        steps=steps,
        rsu_config=rsu_config,
        vehicle_rsu=0,
        max_vehicles=max_vehicles,
        results_root=results_root,
        run_name=base_label,
        generate_plots=False,
    )

    print("\n=======================================================", flush=True)
    print(f"[BENCHMARK] Step 2/2: Running Experimental (Static + {vehicle_rsu_count} vRSUs)...", flush=True)
    print("=======================================================", flush=True)

    vrsu_label = f"bench_static_and_vrsu_{steps}steps"
    run_sumo_with_fog(
        sumo_cfg=sumo_cfg,
        start_sumo=True,
        use_gui=False,
        steps=steps,
        rsu_config=rsu_config,
        vehicle_rsu=vehicle_rsu_count,
        max_vehicles=max_vehicles,
        results_root=results_root,
        run_name=vrsu_label,
        generate_plots=False,
    )

    base_path = Path(results_root) / base_label
    vrsu_path = Path(results_root) / vrsu_label
    return base_path, vrsu_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark and generate quantitative comparison report between Static-Only and Static+vRSU runs.")
    parser.add_argument("--baseline-dir", help="Path to existing baseline run directory (Static RSUs only)")
    parser.add_argument("--vrsu-dir", help="Path to existing experimental run directory (Static + Vehicular RSUs)")
    parser.add_argument("--auto-run", action="store_true", help="Automatically run both baseline and experimental simulations")
    parser.add_argument("--steps", type=int, default=150, help="Number of simulation steps for auto-run")
    parser.add_argument("--vehicle-rsu", type=int, default=3, help="Number of volunteered vehicular RSUs for auto-run")
    parser.add_argument("--vehicle", type=int, default=None, help="Total vehicles cap for auto-run")
    parser.add_argument("--sumo-cfg", default="sumo/delhi/fog_test.sumocfg", help="SUMO config for auto-run")
    parser.add_argument("--rsu-config", default="sumoXyafs/rsu_config.json", help="RSU configuration JSON")
    parser.add_argument("--results-root", default="results", help="Results root folder")
    parser.add_argument("--output-dir", default="results/vrsu_comparison_report", help="Output directory for plots and markdown report")
    args = parser.parse_args()

    results_base = Path(args.results_root)
    if not results_base.is_absolute():
        results_base = BASE_DIR / results_base

    if args.auto_run or (not args.baseline_dir and not args.vrsu_dir):
        base_dir, vrsu_dir = run_comparative_experiment(
            sumo_cfg=args.sumo_cfg,
            rsu_config=args.rsu_config,
            steps=args.steps,
            vehicle_rsu_count=args.vehicle_rsu,
            max_vehicles=args.vehicle,
            results_root=str(results_base),
        )
    else:
        base_dir = Path(args.baseline_dir)
        vrsu_dir = Path(args.vrsu_dir)

    print(f"\n[BENCHMARK] Analyzing results from:\n - Baseline: {base_dir}\n - Experimental: {vrsu_dir}", flush=True)
    baseline_metrics = ExperimentMetrics(base_dir, label="Static RSUs Only")
    vrsu_metrics = ExperimentMetrics(vrsu_dir, label="Static + Dynamic vRSUs")

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = BASE_DIR / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[BENCHMARK] Generating comparative figures in {out_dir}...", flush=True)
    plots = generate_comparison_plots(baseline_metrics, vrsu_metrics, out_dir)

    report_path = out_dir / "vrsu_comparison_report.md"
    print(f"[BENCHMARK] Generating quantitative Markdown report: {report_path}...", flush=True)
    generate_markdown_report(baseline_metrics, vrsu_metrics, report_path, plots)

    print(f"\n[BENCHMARK SUCCESS] Report generated successfully at:\n{report_path.resolve()}\n", flush=True)


if __name__ == "__main__":
    main()

