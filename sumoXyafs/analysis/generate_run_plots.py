from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _f(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except Exception:
        return float(default)


def _i(row: dict[str, str], key: str, default: int = 0) -> int:
    try:
        return int(float(row.get(key, default)))
    except Exception:
        return int(default)


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _binned_stats(x_values: list[float], y_values: list[float], bins: int = 16) -> tuple[list[float], list[float], list[float], list[float]]:
    if not x_values or not y_values:
        return [], [], [], []

    x_min = min(x_values)
    x_max = max(x_values)
    if x_max <= x_min:
        mean_y = sum(y_values) / max(1, len(y_values))
        return [x_min], [mean_y], [mean_y], [mean_y]

    bin_width = (x_max - x_min) / max(1, bins)
    buckets: dict[int, list[float]] = defaultdict(list)
    for xv, yv in zip(x_values, y_values):
        idx = int((xv - x_min) / bin_width) if bin_width > 0 else 0
        idx = min(max(idx, 0), bins - 1)
        buckets[idx].append(yv)

    centers: list[float] = []
    means: list[float] = []
    lows: list[float] = []
    highs: list[float] = []
    for idx in sorted(buckets.keys()):
        ys = sorted(buckets[idx])
        n = len(ys)
        if n == 0:
            continue
        center = x_min + ((idx + 0.5) * bin_width)
        mean_val = sum(ys) / n
        low = ys[max(0, int(0.25 * (n - 1)))]
        high = ys[min(n - 1, int(0.75 * (n - 1)))]
        centers.append(center)
        means.append(mean_val)
        lows.append(low)
        highs.append(high)

    return centers, means, lows, highs


def _plot_vehicle_position_vs_time(telemetry_rows: list[dict[str, str]], plots_dir: Path) -> Path | None:
    if not telemetry_rows:
        return None

    by_vehicle: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in telemetry_rows:
        vid = str(row.get("vehicle_id", "")).strip()
        if not vid:
            continue
        by_vehicle[vid].append(row)

    if not by_vehicle:
        return None

    top_vehicles = [vid for vid, _ in Counter({k: len(v) for k, v in by_vehicle.items()}).most_common(6)]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for vid in top_vehicles:
        rows = sorted(by_vehicle[vid], key=lambda r: _i(r, "step"))
        steps = [_i(r, "step") for r in rows]
        xs = [_f(r, "position_x") for r in rows]
        ys = [_f(r, "position_y") for r in rows]
        axes[0].plot(steps, xs, label=vid, linewidth=1.6)
        axes[1].plot(steps, ys, label=vid, linewidth=1.6)

    axes[0].set_title("Vehicle X Position vs Time")
    axes[0].set_ylabel("x (m)")
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="upper right", fontsize=8)

    axes[1].set_title("Vehicle Y Position vs Time")
    axes[1].set_ylabel("y (m)")
    axes[1].set_xlabel("step")
    axes[1].grid(alpha=0.3)

    return _save(fig, plots_dir / "01_mobility_vehicle_position_vs_time.png")


def _plot_speed_density_vs_region(telemetry_rows: list[dict[str, str]], plots_dir: Path) -> Path | None:
    if not telemetry_rows:
        return None

    region_step: dict[str, dict[int, dict[str, float]]] = defaultdict(lambda: defaultdict(lambda: {"sum_speed": 0.0, "count": 0.0}))
    region_count = Counter()
    for row in telemetry_rows:
        region = str(row.get("observed_rsu", "")).strip() or "unknown"
        step = _i(row, "step")
        speed = _f(row, "speed_mps")
        bucket = region_step[region][step]
        bucket["sum_speed"] += speed
        bucket["count"] += 1.0
        region_count[region] += 1

    top_regions = [r for r, _ in region_count.most_common(6)]
    if not top_regions:
        return None

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

    for region in top_regions:
        step_map = region_step[region]
        steps = sorted(step_map.keys())
        avg_speed = [step_map[s]["sum_speed"] / max(1.0, step_map[s]["count"]) for s in steps]
        density = [step_map[s]["count"] for s in steps]
        axes[0].plot(steps, avg_speed, linewidth=1.4, label=region)
        axes[1].plot(steps, density, linewidth=1.4, label=region)

    axes[0].set_title("Average Speed vs Region")
    axes[0].set_ylabel("speed (m/s)")
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="upper right", fontsize=8)

    axes[1].set_title("Vehicle Density vs Region")
    axes[1].set_ylabel("vehicle count")
    axes[1].set_xlabel("step")
    axes[1].grid(alpha=0.3)

    return _save(fig, plots_dir / "02_mobility_speed_density_vs_region.png")


def _plot_v2i_distance_vs_link_latency(
    decision_rows: list[dict[str, str]],
    step_rows: list[dict[str, str]],
    plots_dir: Path,
) -> Path | None:
    if not decision_rows or not step_rows:
        return None

    link_by_step: dict[int, float] = {}
    for row in step_rows:
        link_by_step[_i(row, "step")] = _f(row, "avg_link_latency_s")

    distances: list[float] = []
    latencies: list[float] = []
    for row in decision_rows:
        step = _i(row, "step")
        link = link_by_step.get(step)
        if link is None:
            continue
        distances.append(_f(row, "distance_to_exec_rsu_m"))
        latencies.append(link)

    if not distances:
        return None

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.scatter(distances, latencies, alpha=0.35, s=14, c="#0077b6")
    ax.set_title("V2I Proxy: Distance to Execution RSU vs Link Latency")
    ax.set_xlabel("distance to execution RSU (m)")
    ax.set_ylabel("YAFS avg link latency (s)")
    ax.grid(alpha=0.3)
    return _save(fig, plots_dir / "03_network_v2i_distance_vs_link_latency.png")


def _plot_latency_vs_load(step_rows: list[dict[str, str]], plots_dir: Path) -> Path | None:
    if not step_rows:
        return None

    load = [_f(r, "total_queue_depth") for r in step_rows]
    latency = [_f(r, "avg_end_to_end_s") for r in step_rows]

    if not load:
        return None

    fig, ax = plt.subplots(figsize=(10, 6))

    hexbin = ax.hexbin(load, latency, gridsize=22, mincnt=1, cmap="magma", alpha=0.9)
    fig.colorbar(hexbin, ax=ax, label="samples per bin")

    centers, means, lows, highs = _binned_stats(load, latency, bins=14)
    if centers:
        ax.plot(centers, means, color="#00d4ff", linewidth=2.4, label="binned mean trend")
        ax.fill_between(centers, lows, highs, color="#00d4ff", alpha=0.18, label="IQR band")

    ax.scatter(load, latency, alpha=0.15, s=12, c="#ffffff", edgecolors="none")
    ax.set_title("Latency vs Load")
    ax.set_xlabel("total queue depth")
    ax.set_ylabel("avg end-to-end latency (s)")
    ax.grid(alpha=0.3)
    if centers:
        ax.legend(loc="upper left", fontsize=8)
    return _save(fig, plots_dir / "04_perf_latency_vs_load.png")


def _plot_throughput_vs_density(step_rows: list[dict[str, str]], plots_dir: Path) -> Path | None:
    if not step_rows:
        return None

    density = [_f(r, "vehicle_count") for r in step_rows]
    throughput = [_f(r, "messages") for r in step_rows]

    fig, ax = plt.subplots(figsize=(10, 6))

    hexbin = ax.hexbin(density, throughput, gridsize=22, mincnt=1, cmap="viridis", alpha=0.9)
    fig.colorbar(hexbin, ax=ax, label="samples per bin")

    centers, means, lows, highs = _binned_stats(density, throughput, bins=14)
    if centers:
        ax.plot(centers, means, color="#ffb703", linewidth=2.4, label="binned mean trend")
        ax.fill_between(centers, lows, highs, color="#ffb703", alpha=0.18, label="IQR band")

    ax.scatter(density, throughput, alpha=0.15, s=12, c="#ffffff", edgecolors="none")
    ax.set_title("Throughput vs Density")
    ax.set_xlabel("vehicle count per step")
    ax.set_ylabel("messages processed per step")
    ax.grid(alpha=0.3)
    if centers:
        ax.legend(loc="upper left", fontsize=8)
    return _save(fig, plots_dir / "05_perf_throughput_vs_density.png")


def _plot_utilization_vs_allocation(rsu_rows: list[dict[str, str]], plots_dir: Path) -> Path | None:
    if not rsu_rows:
        return None

    agg: dict[str, dict[str, float]] = defaultdict(lambda: {"util_sum": 0.0, "alloc_sum": 0.0, "n": 0.0})
    for row in rsu_rows:
        rsu = str(row.get("rsu_id", "")).strip()
        if not rsu:
            continue
        bucket = agg[rsu]
        bucket["util_sum"] += _f(row, "utilization")
        bucket["alloc_sum"] += _f(row, "assigned_count")
        bucket["n"] += 1.0

    if not agg:
        return None

    rsus = sorted(agg.keys())
    mean_util = [agg[r]["util_sum"] / max(1.0, agg[r]["n"]) for r in rsus]
    mean_alloc = [agg[r]["alloc_sum"] / max(1.0, agg[r]["n"]) for r in rsus]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].bar(rsus, mean_util, color="#8ecae6")
    axes[0].set_title("Mean RSU Utilization")
    axes[0].set_ylabel("queue/capacity")
    axes[0].grid(alpha=0.25, axis="y")

    axes[1].bar(rsus, mean_alloc, color="#ffb703")
    axes[1].set_title("Mean Allocation Count per RSU")
    axes[1].set_ylabel("assignments per step")
    axes[1].set_xlabel("RSU")
    axes[1].grid(alpha=0.25, axis="y")

    return _save(fig, plots_dir / "06_perf_utilization_vs_allocation.png")


def generate_plots_for_run(run_dir: str | Path) -> list[Path]:
    run_path = Path(run_dir)
    analysis_dir = run_path / "analysis"
    plots_dir = run_path / "plots"

    telemetry_rows = _read_csv(analysis_dir / "telemetry.csv")
    decision_rows = _read_csv(analysis_dir / "decisions.csv")
    step_rows = _read_csv(analysis_dir / "step_metrics.csv")
    rsu_rows = _read_csv(analysis_dir / "rsu_step_metrics.csv")

    outputs: list[Path] = []
    for candidate in [
        _plot_vehicle_position_vs_time(telemetry_rows, plots_dir),
        _plot_speed_density_vs_region(telemetry_rows, plots_dir),
        _plot_v2i_distance_vs_link_latency(decision_rows, step_rows, plots_dir),
        _plot_latency_vs_load(step_rows, plots_dir),
        _plot_throughput_vs_density(step_rows, plots_dir),
        _plot_utilization_vs_allocation(rsu_rows, plots_dir),
    ]:
        if candidate is not None:
            outputs.append(candidate)

    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate analysis plots for a SUMO+YAFS run directory")
    parser.add_argument("--run-dir", required=True, help="Path to results/<run_name> directory")
    args = parser.parse_args()

    outputs = generate_plots_for_run(args.run_dir)
    if not outputs:
        print("[PLOT] No plots generated (missing or empty analysis CSV files).", flush=True)
        return

    print(f"[PLOT] Generated {len(outputs)} plots:", flush=True)
    for path in outputs:
        print(f"  - {path}", flush=True)


if __name__ == "__main__":
    main()
