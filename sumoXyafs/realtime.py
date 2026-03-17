"""Run SUMO and orchestrate multi-RSU fog control via YAFS bridge."""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from datetime import datetime
from pathlib import Path


def _resolve_path(path_str: str) -> str:
    path = Path(path_str)
    if path.exists():
        return str(path)

    base = Path(__file__).resolve().parent
    base_relative = base / path
    if base_relative.exists():
        return str(base_relative)

    # Accept defaults prefixed with "sumoXyafs/" when running from repo root.
    parts = list(path.parts)
    if parts and parts[0].lower() == base.name.lower():
        stripped = base / Path(*parts[1:])
        if stripped.exists():
            return str(stripped)

    return str(path)


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _prepare_run_dirs(results_root: str, run_name: str = "") -> tuple[Path, Path, Path]:
    base = Path(results_root)
    if not base.is_absolute():
        base = Path(__file__).resolve().parent / base
    base.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    label = run_name.strip() if run_name.strip() else f"run_{stamp}"

    run_dir = base / label
    suffix = 1
    while run_dir.exists():
        run_dir = base / f"{label}_{suffix:02d}"
        suffix += 1

    raw_dir = run_dir / "raw"
    analysis_dir = run_dir / "analysis"
    plots_dir = run_dir / "plots"
    raw_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, raw_dir, analysis_dir


def _load_traci():
    try:
        import traci  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("TraCI is not installed. Install SUMO Python tools and retry.") from exc
    return traci


def _apply_feedback(traci, feedbacks: list[dict], live_vehicle_ids: set[str], dashboard=None) -> None:
    for cmd in feedbacks:
        vid = cmd.get("id")
        if not vid or vid not in live_vehicle_ids:
            continue

        # Apply speed first, then route update, while tolerating disappearing vehicles.
        try:
            if "speed" in cmd and cmd["speed"] is not None:
                traci.vehicle.setSpeed(vid, float(cmd["speed"]))
            if bool(cmd.get("reroute", False)):
                traci.vehicle.rerouteTraveltime(vid)
            print(f"[FOG->SUMO] Applied {cmd}", flush=True)
            if dashboard is not None:
                dashboard.record_command(cmd)
        except Exception:
            pass


# Draw RSU locations as POIs in SUMO for better visualization
def _draw_rsu_pois(traci, rsus: list, layer: int = 5) -> None:
    existing_ids = set()
    try:
        existing_ids = set(traci.poi.getIDList())
    except Exception:
        pass

    for rsu in rsus:
        poi_id = f"rsu:{rsu.rsu_id}"
        if poi_id in existing_ids:
            continue
        try:
            traci.poi.add(
                poi_id,
                x=float(rsu.x),
                y=float(rsu.y),
                # color=(255, 0, 0, 255),
                color=(255, 255, 255, 255),
                poiType="rsu",
                layer=layer,
                imgFile="assets/rsu-icon.png",
                width=20,
                height=20,
                angle=0.0,
            )
            traci.poi.setParameter(poi_id, "name", rsu.rsu_id)
        except Exception:
            pass


# ADD vehicle text parameters for better visualization
def _set_vehicle_text_params(
    traci,
    vehicle_id: str,
    observed_rsu: str,
    execution_rsu: str,
    state: str,
    speed_mps: float,
    step: int,
    task_id: str = "",
) -> None:
    speed_kmh = float(speed_mps) * 3.6
    text = (
        f"rsu={observed_rsu} exec={execution_rsu} "
        f"state={state} v={speed_kmh:.1f}km/h"
    )
    if state == 'keep_speed':
        color = (0,0,255,255)
    else:
        color = (255,0,0,255)
    try:
        traci.vehicle.setParameter(vehicle_id, "current_rsu", observed_rsu)
        traci.vehicle.setParameter(vehicle_id, "execution_rsu", execution_rsu)
        traci.vehicle.setParameter(vehicle_id, "state", state)
        traci.vehicle.setParameter(vehicle_id, "speed_kmh", f"{speed_kmh:.1f}")
        traci.vehicle.setParameter(vehicle_id, "sim_step", str(step))
        traci.vehicle.setParameter(vehicle_id, "task_id", task_id)
        traci.vehicle.setParameter(vehicle_id, "vehicle_text", text)
        traci.vehicle.setColor(vehicle_id, color)
    except Exception:
        pass


def run_sumo_with_fog(
    sumo_cfg: str,
    port: int = 8873,
    start_sumo: bool = False,
    use_gui: bool = False,
    steps: int = 500,
    step_delay: float = 0.0,
    rsu_config: str = "sumoXyafs/rsu_config.json",
    task_deadline: float = 2.0,
    overload_queue_threshold: int = 20,
    distance_latency_base: float = 0.0,
    distance_latency_max: float = 0.0,
    distance_latency_norm: float = 500.0,
    distance_error: float = 0.0,
    adaptive_latency_eta: bool = False,
    adaptive_latency_alpha: float = 0.25,
    adaptive_latency_weight: float = 1.0,
    reroute_cooldown: float = 5.0,
    rsu_capacity_default: int = 20,
    dashboard: bool = False,
    dashboard_rows: int = 20,
    results_root: str = "results",
    run_name: str = "",
    generate_plots: bool = True,
) -> None:
    from modules.models import Telemetry
    from modules.rsu_registry import RSURegistry
    from modules.runtime_dashboard import RuntimeDashboard
    from analysis.generate_run_plots import generate_plots_for_run
    from yafs_core import MinimalFogSim

    traci = _load_traci()
    sumo_cfg = _resolve_path(sumo_cfg)
    rsu_config = _resolve_path(rsu_config)
    run_dir, raw_dir, analysis_dir = _prepare_run_dirs(results_root=results_root, run_name=run_name)
    print(f"[RUN] Results directory: {run_dir}", flush=True)

    registry = RSURegistry.from_json(rsu_config, capacity_default=rsu_capacity_default)
    fog = MinimalFogSim(
        rsu_registry=registry,
        task_deadline_s=task_deadline,
        compute_time_s=2.0,
        overload_threshold=overload_queue_threshold,
        reroute_cooldown_s=reroute_cooldown,
        distance_latency_base_s=distance_latency_base,
        distance_latency_max_s=distance_latency_max,
        distance_latency_norm_m=distance_latency_norm,
        distance_error=distance_error,
        adaptive_latency_eta=adaptive_latency_eta,
        adaptive_latency_alpha=adaptive_latency_alpha,
        adaptive_latency_weight=adaptive_latency_weight,
        results_prefix=str(raw_dir / "yafs_runtime"),
    )
    dash = RuntimeDashboard(enabled=dashboard, max_rows=dashboard_rows)
    dash.configure_rsus(registry.all_rsus())

    # Start SUMO process or connect to an existing one.
    if start_sumo:
        cmd = ["sumo-gui" if use_gui else "sumo", "-c", sumo_cfg, "--step-length", "1"]
        traci.start(cmd, port=port)
    else:
        traci.init(port=port, numRetries=10, host="127.0.0.1")

    _draw_rsu_pois(traci, registry.all_rsus())

    telemetry_rows: list[dict] = []
    decision_rows: list[dict] = []
    step_rows: list[dict] = []
    rsu_rows: list[dict] = []

    try:
        step = 0
        while step < steps:
            dash.begin_step(step=step, steps_total=steps)
            traci.simulationStep()
            live_vehicle_ids = set(traci.vehicle.getIDList())
            fog.prune_missing(live_vehicle_ids)

            telemetry_batch: list[Telemetry] = []
            telemetry_by_vehicle: dict[str, Telemetry] = {}

            # Capture telemetry, perform RSU scheduling, and enqueue fog feedback.
            for veh in live_vehicle_ids:
                position = traci.vehicle.getPosition(veh)
                speed = float(traci.vehicle.getSpeed(veh))

                telemetry = Telemetry(
                    vehicle_id=veh,
                    position=(float(position[0]), float(position[1])),
                    speed=speed,
                    step=step,
                    observed_rsu="",
                )
                telemetry_batch.append(telemetry)
                telemetry_by_vehicle[telemetry.vehicle_id] = telemetry

            step_records = fog.process_step(telemetry_batch=telemetry_batch, step=step)

            for record in step_records:
                decision = record["decision"]
                command = record["feedback"]
                telemetry = telemetry_by_vehicle.get(decision.vehicle_id)
                if telemetry is None:
                    continue

                distance_to_exec = registry.distance_to_rsu(telemetry.position, decision.execution_rsu)
                telemetry_rows.append(
                    {
                        "step": step,
                        "vehicle_id": telemetry.vehicle_id,
                        "position_x": telemetry.position[0],
                        "position_y": telemetry.position[1],
                        "speed_mps": telemetry.speed,
                        "observed_rsu": decision.observed_rsu,
                        "execution_rsu": decision.execution_rsu,
                        "action": decision.action,
                    }
                )
                decision_rows.append(
                    {
                        "step": step,
                        "vehicle_id": decision.vehicle_id,
                        "observed_rsu": decision.observed_rsu,
                        "owner_rsu": decision.owner_rsu,
                        "execution_rsu": decision.execution_rsu,
                        "action": decision.action,
                        "reason": decision.reason,
                        "estimated_finish_step": decision.estimated_finish_step,
                        "deadline_step": decision.deadline_step,
                        "distance_to_exec_rsu_m": distance_to_exec,
                        "reroute": bool(command.get("reroute", False)),
                        "command_action": str(command.get("action", "keep_speed")),
                    }
                )

                if decision.action != "keep_local":
                    dash.record_scheduler(
                        vehicle_id=decision.vehicle_id,
                        action=decision.action,
                        reason=decision.reason,
                        src_rsu=decision.observed_rsu,
                        exec_rsu=decision.execution_rsu,
                    )
                    print(
                        f"[SCHED] veh={decision.vehicle_id} obs={decision.observed_rsu} owner={decision.owner_rsu} "
                        f"exec={decision.execution_rsu} action={decision.action} reason={decision.reason} "
                        f"eta={decision.estimated_finish_step} ddl={decision.deadline_step}",
                        flush=True,
                    )

                _set_vehicle_text_params(
                    traci,
                    vehicle_id=telemetry.vehicle_id,
                    observed_rsu=decision.observed_rsu,
                    execution_rsu=decision.execution_rsu,
                    state=str(command.get("action", "keep_speed")),
                    speed_mps=telemetry.speed,
                    step=telemetry.step,
                    task_id=str(command.get("task_id", "")),
                )
                dash.update_vehicle(
                    vehicle_id=telemetry.vehicle_id,
                    position=telemetry.position,
                    observed_rsu=decision.observed_rsu,
                    execution_rsu=decision.execution_rsu,
                    speed=telemetry.speed,
                    action=str(command.get("action", "keep_speed")),
                    reroute=bool(command.get("reroute", False)),
                )

            feedbacks = [record["feedback"] for record in step_records]
            _apply_feedback(traci, feedbacks, live_vehicle_ids, dashboard=dash)

            current_queue_depths = fog.active_queue_depths()
            metrics = fog.last_metrics_snapshot()
            action_counter = Counter(record["decision"].action for record in step_records)

            step_rows.append(
                {
                    "step": step,
                    "vehicle_count": len(live_vehicle_ids),
                    "active_tasks": sum(current_queue_depths.values()),
                    "total_queue_depth": sum(current_queue_depths.values()),
                    "events": metrics.get("events", 0),
                    "links": metrics.get("links", 0),
                    "messages": metrics.get("messages", 0),
                    "avg_end_to_end_s": metrics.get("avg_end_to_end_s", 0.0),
                    "avg_service_s": metrics.get("avg_service_s", 0.0),
                    "avg_link_latency_s": metrics.get("avg_link_latency_s", 0.0),
                    "keep_local_count": action_counter.get("keep_local", 0),
                    "sticky_count": action_counter.get("sticky", 0),
                    "migrate_count": action_counter.get("migrate", 0),
                    "offload_count": action_counter.get("offload", 0),
                }
            )

            assigned_counter = Counter(record["decision"].execution_rsu for record in step_records)
            for rsu in registry.all_rsus():
                queue_depth = current_queue_depths.get(rsu.rsu_id, 0)
                capacity = max(1, int(rsu.capacity))
                rsu_rows.append(
                    {
                        "step": step,
                        "rsu_id": rsu.rsu_id,
                        "queue_depth": queue_depth,
                        "capacity": capacity,
                        "utilization": queue_depth / capacity,
                        "assigned_count": assigned_counter.get(rsu.rsu_id, 0),
                    }
                )

            dash.finish_step(live_vehicle_ids)
            dash.set_snapshot(
                step=step + 1,
                steps_total=steps,
                vehicles=len(live_vehicle_ids),
                active_tasks=sum(current_queue_depths.values()),
                queue_depths=current_queue_depths,
            )

            # Periodic metrics log keeps run-time output concise while exposing YAFS-level timing.
            if (step + 1) % 20 == 0 and metrics.get("messages", 0) > 0:
                print(
                    f"[YAFS] step={step + 1} events={metrics.get('events', 0)} links={metrics.get('links', 0)} "
                    f"e2e={metrics.get('avg_end_to_end_s', 0.0):.4f}s "
                    f"service={metrics.get('avg_service_s', 0.0):.4f}s "
                    f"link={metrics.get('avg_link_latency_s', 0.0):.4f}s",
                    flush=True,
                )
                per_rsu = metrics.get("per_rsu", {})
                if isinstance(per_rsu, dict) and per_rsu:
                    top = sorted(
                        per_rsu.items(),
                        key=lambda item: item[1].get("messages", 0),
                        reverse=True,
                    )[:2]
                    for rsu_id, rsu_stats in top:
                        print(
                            f"[YAFS-RSU] step={step + 1} rsu={rsu_id} messages={rsu_stats.get('messages', 0)} "
                            f"e2e={rsu_stats.get('avg_end_to_end_s', 0.0):.4f}s "
                            f"service={rsu_stats.get('avg_service_s', 0.0):.4f}s",
                            flush=True,
                        )
            step += 1

            if step_delay > 0:
                time.sleep(step_delay)

        _write_csv(
            analysis_dir / "telemetry.csv",
            telemetry_rows,
            [
                "step",
                "vehicle_id",
                "position_x",
                "position_y",
                "speed_mps",
                "observed_rsu",
                "execution_rsu",
                "action",
            ],
        )
        _write_csv(
            analysis_dir / "decisions.csv",
            decision_rows,
            [
                "step",
                "vehicle_id",
                "observed_rsu",
                "owner_rsu",
                "execution_rsu",
                "action",
                "reason",
                "estimated_finish_step",
                "deadline_step",
                "distance_to_exec_rsu_m",
                "reroute",
                "command_action",
            ],
        )
        _write_csv(
            analysis_dir / "step_metrics.csv",
            step_rows,
            [
                "step",
                "vehicle_count",
                "active_tasks",
                "total_queue_depth",
                "events",
                "links",
                "messages",
                "avg_end_to_end_s",
                "avg_service_s",
                "avg_link_latency_s",
                "keep_local_count",
                "sticky_count",
                "migrate_count",
                "offload_count",
            ],
        )
        _write_csv(
            analysis_dir / "rsu_step_metrics.csv",
            rsu_rows,
            [
                "step",
                "rsu_id",
                "queue_depth",
                "capacity",
                "utilization",
                "assigned_count",
            ],
        )

        metadata = {
            "sumo_cfg": sumo_cfg,
            "rsu_config": rsu_config,
            "steps": steps,
            "port": port,
            "start_sumo": start_sumo,
            "gui": use_gui,
            "run_dir": str(run_dir),
            "raw_dir": str(raw_dir),
            "analysis_dir": str(analysis_dir),
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        }
        (run_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        if generate_plots:
            outputs = generate_plots_for_run(run_dir)
            if outputs:
                print(f"[PLOT] Generated {len(outputs)} plots under {run_dir / 'plots'}", flush=True)
            else:
                print("[PLOT] No plots generated from current run data.", flush=True)
    finally:
        dash.close()
        try:
            traci.close()
        except Exception:
            pass


def main() -> None:
    p = argparse.ArgumentParser(description="Run SUMO and forward telemetry to a YAFS-based fog sim")
    p.add_argument("--sumo-cfg", default="sumo/delhi/osm.sumocfg", help="Path to SUMO .sumocfg file")
    p.add_argument("--port", type=int, default=8873, help="TraCI remote port")
    p.add_argument("--start-sumo", action="store_true", help="Start SUMO from this script")
    p.add_argument("--gui", action="store_true", help="Start SUMO GUI when --start-sumo is used")
    p.add_argument("--steps", type=int, default=500, help="Number of simulation steps to run")
    p.add_argument("--step-delay", type=float, default=0.0, help="Delay after each simulation step for GUI playback")
    p.add_argument("--rsu-config", default="sumoXyafs/rsu_config.json", help="Path to RSU anchor configuration JSON")
    p.add_argument("--task-deadline", type=float, default=2.0, help="Task deadline in simulation seconds")
    p.add_argument("--overload-queue-threshold", type=int, default=20, help="Queue depth threshold for RSU overload")
    p.add_argument("--distance-latency-base", type=float, default=0.0, help="Base link latency (s) added to ETA")
    p.add_argument("--distance-latency-max", type=float, default=0.0, help="Latency term (s) at distance == --distance-latency-norm")
    p.add_argument("--distance-latency-norm", type=float, default=500.0, help="Distance (m) scale for uncapped linear latency growth")
    p.add_argument("--distance-error", type=float, default=0.0, help="Relative distance latency error factor (distance_latency += distance_error * distance_latency)")
    p.add_argument("--adaptive-latency-eta", action="store_true", help="Blend measured YAFS link latency into ETA model")
    p.add_argument("--adaptive-latency-alpha", type=float, default=0.25, help="EWMA smoothing factor for measured link latency [0,1]")
    p.add_argument("--adaptive-latency-weight", type=float, default=1.0, help="Multiplier for measured link latency contribution to ETA")
    p.add_argument("--reroute-cooldown", type=float, default=5.0, help="Minimum seconds between reroutes per vehicle")
    p.add_argument("--rsu-capacity-default", type=int, default=20, help="Default RSU capacity if missing in config")
    p.add_argument("--dashboard", action="store_true", help="Show a basic Tkinter runtime dashboard")
    p.add_argument("--dashboard-rows", type=int, default=20, help="Max rows shown in dashboard event lists")
    p.add_argument("--results-root", default="results", help="Results root directory (default: sumoXyafs/results)")
    p.add_argument("--run-name", default="", help="Optional run folder name inside results root")
    p.add_argument("--no-plots", action="store_true", help="Disable auto-generation of analysis plots")
    args = p.parse_args()

    run_sumo_with_fog(
        args.sumo_cfg,
        port=args.port,
        start_sumo=args.start_sumo,
        use_gui=args.gui,
        steps=args.steps,
        step_delay=args.step_delay,
        rsu_config=args.rsu_config,
        task_deadline=args.task_deadline,
        overload_queue_threshold=args.overload_queue_threshold,
        distance_latency_base=args.distance_latency_base,
        distance_latency_max=args.distance_latency_max,
        distance_latency_norm=args.distance_latency_norm,
        distance_error=args.distance_error,
        adaptive_latency_eta=args.adaptive_latency_eta,
        adaptive_latency_alpha=args.adaptive_latency_alpha,
        adaptive_latency_weight=args.adaptive_latency_weight,
        reroute_cooldown=args.reroute_cooldown,
        rsu_capacity_default=args.rsu_capacity_default,
        dashboard=args.dashboard,
        dashboard_rows=args.dashboard_rows,
        results_root=args.results_root,
        run_name=args.run_name,
        generate_plots=not args.no_plots,
    )


if __name__ == "__main__":
    main()
