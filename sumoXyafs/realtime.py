"""Run SUMO and orchestrate multi-RSU fog control via YAFS bridge."""
from __future__ import annotations

import argparse
import time
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
    reroute_cooldown: float = 5.0,
    rsu_capacity_default: int = 20,
    dashboard: bool = False,
    dashboard_rows: int = 20,
) -> None:
    from modules.models import Telemetry
    from modules.rsu_registry import RSURegistry
    from modules.runtime_dashboard import RuntimeDashboard
    from yafs_core import MinimalFogSim

    traci = _load_traci()
    sumo_cfg = _resolve_path(sumo_cfg)
    rsu_config = _resolve_path(rsu_config)

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
                nearest = registry.nearest_rsu(position)

                telemetry = Telemetry(
                    vehicle_id=veh,
                    position=(float(position[0]), float(position[1])),
                    speed=speed,
                    step=step,
                    observed_rsu=nearest.rsu_id,
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
            dash.finish_step(live_vehicle_ids)
            dash.set_snapshot(
                step=step + 1,
                steps_total=steps,
                vehicles=len(live_vehicle_ids),
                active_tasks=sum(current_queue_depths.values()),
                queue_depths=current_queue_depths,
            )
            step += 1

            if step_delay > 0:
                time.sleep(step_delay)
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
    p.add_argument("--reroute-cooldown", type=float, default=5.0, help="Minimum seconds between reroutes per vehicle")
    p.add_argument("--rsu-capacity-default", type=int, default=20, help="Default RSU capacity if missing in config")
    p.add_argument("--dashboard", action="store_true", help="Show a basic Tkinter runtime dashboard")
    p.add_argument("--dashboard-rows", type=int, default=20, help="Max rows shown in dashboard event lists")
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
        reroute_cooldown=args.reroute_cooldown,
        rsu_capacity_default=args.rsu_capacity_default,
        dashboard=args.dashboard,
        dashboard_rows=args.dashboard_rows,
    )


if __name__ == "__main__":
    main()
