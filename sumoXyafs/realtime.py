import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
for p in [
    str(BASE_DIR),
    str(PROJECT_ROOT),
    str(PROJECT_ROOT / "YAFS" / "src"),
    str(PROJECT_ROOT / "YAIFS" / ".venv" / "Lib" / "site-packages"),
    str(BASE_DIR / "venv" / "src" / "yafs" / "src"),
    str(BASE_DIR / "venv" / "lib" / "python3.12" / "site-packages"),
    str(PROJECT_ROOT / ".venv" / "src" / "yafs" / "src"),
]:
    if p not in sys.path and Path(p).exists():
        sys.path.append(p)

import argparse
import csv
import json
import time
from collections import Counter
from datetime import datetime


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
            # Only apply speed override if action is slow_down and target speed > 0.5 m/s.
            # In SUMO TraCI, calling setSpeed(vid, 0.0) permanently freezes the vehicle at rest.
            if cmd.get("action") == "slow_down" and "speed" in cmd and cmd["speed"] is not None:
                spd = float(cmd["speed"])
                if spd > 0.5:
                    traci.vehicle.setSpeed(vid, spd)
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


_PASSENGER_EDGE_CACHE = None
_PASSENGER_EDGE_CENTERS = None


def _get_passenger_edge_cache(traci_module):
    global _PASSENGER_EDGE_CACHE, _PASSENGER_EDGE_CENTERS
    if _PASSENGER_EDGE_CACHE is None:
        _PASSENGER_EDGE_CACHE = []
        _PASSENGER_EDGE_CENTERS = {}
        for edge in traci_module.edge.getIDList():
            if not edge.startswith(":"):
                lane0 = f"{edge}_0"
                try:
                    allowed = traci_module.lane.getAllowed(lane0)
                    disallowed = traci_module.lane.getDisallowed(lane0)
                    if ("passenger" in allowed or not allowed) and "passenger" not in disallowed:
                        _PASSENGER_EDGE_CACHE.append(edge)
                        shape = traci_module.lane.getShape(lane0)
                        if shape:
                            _PASSENGER_EDGE_CENTERS[edge] = shape[len(shape) // 2]
                except Exception:
                    pass
    return _PASSENGER_EDGE_CACHE, _PASSENGER_EDGE_CENTERS


def _find_best_passenger_edge(traci_module, wx, wy):
    import math
    p_edges, p_centers = _get_passenger_edge_cache(traci_module)
    if not p_edges:
        return ""
    try:
        road_info = traci_module.simulation.convertRoad(wx, wy)
        edge = road_info[0] if road_info else ""
        if edge in p_centers:
            return edge
    except Exception:
        pass

    best_edge = p_edges[0]
    min_dist = float("inf")
    for e, (cx, cy) in p_centers.items():
        d = math.hypot(wx - cx, wy - cy)
        if d < min_dist:
            min_dist = d
            best_edge = e
    return best_edge


def _process_web_requests(traci_module, registry, fog):
    if "web_server" not in sys.modules:
        return
    try:
        import web_server
        from modules.models import RSUNode
        state = web_server.SIM_STATE

        # 1. Spawn Vehicles (Tap canvas to add vehicle)
        spawns = state.consume_pending_spawns()
        for idx, v in enumerate(spawns):
            vid = v["vehicle_id"]
            wx, wy = v["x"], v["y"]
            try:
                start_edge = _find_best_passenger_edge(traci_module, wx, wy)
                p_edges, _ = _get_passenger_edge_cache(traci_module)
                dest_edge = p_edges[(idx * 17 + len(vid)) % len(p_edges)] if p_edges else start_edge

                if start_edge:
                    route_res = traci_module.simulation.findRoute(start_edge, dest_edge, vType="veh_passenger")
                    route_edges = list(route_res.edges) if (route_res and route_res.edges) else [start_edge]
                    route_id = f"route_{vid}"
                    traci_module.route.add(route_id, route_edges)

                    available_types = traci_module.vehicletype.getIDList()
                    vtype = "veh_passenger" if "veh_passenger" in available_types else "DEFAULT_VEHTYPE"
                    traci_module.vehicle.add(vid, route_id, typeID=vtype, departPos="0")
                    traci_module.vehicle.setSpeed(vid, 12.0)
                    print(f"[WEB➔SUMO] Spawned vehicle '{vid}' on SUMO edge '{start_edge}' -> '{dest_edge}' ({len(route_edges)} edges)", flush=True)
            except Exception as e:
                print(f"[WEB➔SUMO WARN] Could not spawn {vid} in SUMO: {e}", flush=True)

        # 2. Add RSUs (Build mode)
        add_rsus = state.consume_pending_add_rsus()
        for r in add_rsus:
            rsu_id = r["rsu_id"]
            if not registry.contains(rsu_id):
                node = RSUNode(
                    rsu_id=rsu_id,
                    node_id=len(registry._rsus) + 1,
                    x=r["x"],
                    y=r["y"],
                    capacity=r["capacity"],
                    hardware_profile=r["hardware_profile"],
                    v2x_protocol=r["v2x_protocol"],
                )
                registry.add_rsu(node)
                try:
                    traci_module.poi.add(rsu_id, r["x"], r["y"], (6, 182, 212, 255), poiType="RSU", layer=10)
                    print(f"[WEB➔SUMO] Created RSU '{rsu_id}' POI in SUMO", flush=True)
                except Exception:
                    pass

        # 3. Delete RSUs
        del_rsus = state.consume_pending_delete_rsus()
        for rsu_id in del_rsus:
            if registry.contains(rsu_id):
                registry.delete_rsu(rsu_id)
                try:
                    traci_module.poi.remove(rsu_id)
                    print(f"[WEB➔SUMO] Deleted RSU '{rsu_id}' POI from SUMO", flush=True)
                except Exception:
                    pass

        # 4. Manual Task Reassignments (Drag vehicle line to target RSU)
        with state.lock:
            for vid, target_rsu in list(state.manual_reassignments.items()):
                fog.task_manager.set_manual_override(vid, target_rsu)
                target_node = registry.get(target_rsu)
                if target_node and vid in traci_module.vehicle.getIDList():
                    try:
                        road_info = traci_module.simulation.convertRoad(target_node.x, target_node.y)
                        target_edge = road_info[0]
                        if target_edge and not target_edge.startswith(":"):
                            traci_module.vehicle.changeTarget(vid, target_edge)
                            print(f"[WEB➔SUMO] Rerouted vehicle '{vid}' towards target RSU '{target_rsu}' ({target_edge})", flush=True)
                    except Exception:
                        pass

        # 5. Live RSU Queue Depth Overrides (Sliders)
        with state.lock:
            for rsu_id, qdepth in state.queue_depth_overrides.items():
                fog.task_manager.set_rsu_queue_override(rsu_id, qdepth)

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
    algorithm: str = "greedy",
    log_dataset: bool = False,
    pretrained_model_dir: str = "",
    dashboard: bool = False,
    dashboard_rows: int = 20,
    results_root: str = "results",
    run_name: str = "",
    generate_plots: bool = True,
    vehicle_rsu: int = 0,
    max_vehicles: int | None = None,
    vrsu_capacity: int = 15,
    vrsu_range: float = 350.0,
    vrsu_hardware: str = "jetson_agx_orin",
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

    if log_dataset:
        from modules.algorithms import DatasetLogger
        dataset_logger = DatasetLogger(analysis_dir / "offline_dataset.csv")
    else:
        dataset_logger = None

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
        algorithm=algorithm,
        dataset_logger=dataset_logger,
        pretrained_model_dir=pretrained_model_dir,
        results_prefix=str(raw_dir / "yafs_runtime"),
    )
    dash = RuntimeDashboard(enabled=dashboard, max_rows=dashboard_rows)
    dash.configure_rsus(registry.all_rsus())

    # Start SUMO process or connect to an existing one.
    if start_sumo:
        cmd = ["sumo-gui" if use_gui else "sumo", "-c", sumo_cfg, "--step-length", "1"]
        if use_gui:
            cmd.extend(["--start", "--quit-on-end"])
        if max_vehicles is not None and max_vehicles > 0:
            cmd.extend(["--max-num-vehicles", str(max_vehicles)])
        traci.start(cmd, port=port, numRetries=30)
    else:
        traci.init(port=port, numRetries=20, host="127.0.0.1")

    _draw_rsu_pois(traci, registry.all_rsus())

    active_vrsus: dict[str, str] = {}

    telemetry_rows: list[dict] = []
    decision_rows: list[dict] = []
    step_rows: list[dict] = []
    rsu_rows: list[dict] = []

    try:
        step = 0
        while step < steps:
            if "web_server" in sys.modules:
                import web_server
                while web_server.SIM_STATE.is_paused:
                    time.sleep(0.1)

            dash.begin_step(step=step, steps_total=steps)

            # Sync Web Dashboard User Actions into Live SUMO TraCI
            _process_web_requests(traci, registry, fog)

            traci.simulationStep()
            live_vehicle_ids = set(traci.vehicle.getIDList())
            fog.prune_missing(live_vehicle_ids)

            # Dynamic Volunteered Vehicular RSU (vRSU) lifecycle management
            if vehicle_rsu > 0:
                # 1. Unregister any vRSUs whose vehicles have departed
                departed_vrsus = [rid for rid, vid in active_vrsus.items() if vid not in live_vehicle_ids]
                for rid in departed_vrsus:
                    departed_vid = active_vrsus.pop(rid)
                    registry.unregister_vehicular_rsu(rid)
                    print(f"[vRSU] Vehicular RSU '{rid}' (veh='{departed_vid}') departed the simulation", flush=True)

                # 2. Enlist new volunteer vehicles to maintain target vRSU count
                needed = vehicle_rsu - len(active_vrsus)
                if needed > 0:
                    assigned_vids = set(active_vrsus.values())
                    candidates = sorted([vid for vid in live_vehicle_ids if vid not in assigned_vids])
                    for new_vid in candidates[:needed]:
                        vrsu_id = f"vrsu_{new_vid}"
                        try:
                            pos = traci.vehicle.getPosition(new_vid)
                            spd = float(traci.vehicle.getSpeed(new_vid))
                            try:
                                ang = float(traci.vehicle.getAngle(new_vid))
                            except Exception:
                                ang = 0.0
                        except Exception:
                            continue
                        registry.register_vehicular_rsu(
                            rsu_id=vrsu_id,
                            vehicle_id=new_vid,
                            x=float(pos[0]),
                            y=float(pos[1]),
                            node_id=1,
                            capacity=vrsu_capacity,
                            coverage_radius=vrsu_range,
                            hardware_profile=vrsu_hardware,
                            speed=spd,
                            angle=ang,
                        )
                        active_vrsus[vrsu_id] = new_vid
                        print(f"[vRSU] Vehicle '{new_vid}' volunteered as Vehicular RSU '{vrsu_id}' at ({pos[0]:.1f}, {pos[1]:.1f})", flush=True)

                # 3. Dynamically update positions and velocities of active vRSUs
                for vrsu_id, v_id in list(active_vrsus.items()):
                    try:
                        pos = traci.vehicle.getPosition(v_id)
                        spd = float(traci.vehicle.getSpeed(v_id))
                        try:
                            ang = float(traci.vehicle.getAngle(v_id))
                        except Exception:
                            ang = 0.0
                        registry.update_vehicular_rsu_position(vrsu_id, float(pos[0]), float(pos[1]), speed=spd, angle=ang)
                        traci.vehicle.setColor(v_id, (255, 191, 0, 255))
                        traci.vehicle.setParameter(v_id, "is_vrsu", "true")
                        traci.vehicle.setParameter(v_id, "rsu_id", vrsu_id)
                    except Exception:
                        pass

                # Propagate updated moving RSU coordinates to both dashboards
                dash.update_rsus(registry.all_rsus())
                if "web_server" in sys.modules:
                    try:
                        import web_server
                        web_server.SIM_STATE.sync_rsus(registry.all_rsus())
                    except Exception:
                        pass

            telemetry_batch: list[Telemetry] = []
            telemetry_by_vehicle: dict[str, Telemetry] = {}

            vehicle_angles: dict[str, float] = {}
            for veh in live_vehicle_ids:
                position = traci.vehicle.getPosition(veh)
                speed = float(traci.vehicle.getSpeed(veh))
                try:
                    vehicle_angles[veh] = float(traci.vehicle.getAngle(veh))
                except Exception:
                    vehicle_angles[veh] = 0.0

                telemetry = Telemetry(
                    vehicle_id=veh,
                    position=(float(position[0]), float(position[1])),
                    speed=speed,
                    step=step,
                    observed_rsu="",
                    angle=vehicle_angles.get(veh, 0.0),
                )
                telemetry_batch.append(telemetry)
                telemetry_by_vehicle[telemetry.vehicle_id] = telemetry

            step_records = fog.process_step(telemetry_batch=telemetry_batch, step=step)

            for rec in step_records:
                veh_id = rec["decision"].vehicle_id
                if veh_id in telemetry_by_vehicle:
                    rec["position"] = telemetry_by_vehicle[veh_id].position
                    rec["speed"] = telemetry_by_vehicle[veh_id].speed
                    rec["angle"] = vehicle_angles.get(veh_id, 0.0)

            if "web_server" in sys.modules:
                try:
                    import web_server
                    web_server.SIM_STATE.sync_sumo_step(step_records, step)
                except Exception:
                    pass

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
                        "estimated_network_latency_s": decision.estimated_network_latency_s,
                        "estimated_deadline_slack_steps": decision.deadline_step - decision.estimated_finish_step,
                        "distance_to_exec_rsu_m": distance_to_exec,
                        "execution_tier": getattr(decision, "execution_tier", "V2I"),
                        "transmission_latency_s": getattr(decision, "transmission_latency_s", 0.0),
                        "queue_latency_s": getattr(decision, "queue_latency_s", 0.0),
                        "compute_latency_s": getattr(decision, "compute_latency_s", 0.0),
                        "sinr_linear": getattr(decision, "sinr_linear", 0.0),
                        "data_rate_mbps": getattr(decision, "data_rate_mbps", 0.0),
                        "contact_duration_s": getattr(decision, "contact_duration_s", float("inf")),
                        "is_vrsu": getattr(decision, "is_vrsu", False),
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
            avg_estimated_network_latency = 0.0
            avg_estimated_deadline_slack = 0.0
            if step_records:
                avg_estimated_network_latency = sum(
                    float(record["decision"].estimated_network_latency_s) for record in step_records
                ) / len(step_records)
                avg_estimated_deadline_slack = sum(
                    float(record["decision"].deadline_step - record["decision"].estimated_finish_step) for record in step_records
                ) / len(step_records)

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
                    "completed_tasks": metrics.get("completed_tasks", 0),
                    "avg_estimated_network_latency_s": avg_estimated_network_latency,
                    "avg_estimated_deadline_slack_steps": avg_estimated_deadline_slack,
                    "keep_local_count": action_counter.get("keep_local", 0),
                    "sticky_count": action_counter.get("sticky", 0),
                    "migrate_count": action_counter.get("migrate", 0),
                    "offload_count": action_counter.get("offload", 0),
                    "vrsu_count": len(active_vrsus),
                    "vrsu_tasks_active": sum(current_queue_depths.get(rid, 0) for rid in active_vrsus),
                    "static_rsu_tasks": sum(current_queue_depths.get(r.rsu_id, 0) for r in registry.static_rsus()),
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
                        "is_vehicular": bool(getattr(rsu, "is_vehicular", False)),
                        "x": float(rsu.x),
                        "y": float(rsu.y),
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
                "estimated_network_latency_s",
                "estimated_deadline_slack_steps",
                "distance_to_exec_rsu_m",
                "execution_tier",
                "transmission_latency_s",
                "queue_latency_s",
                "compute_latency_s",
                "sinr_linear",
                "data_rate_mbps",
                "contact_duration_s",
                "is_vrsu",
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
                "completed_tasks",
                "avg_estimated_network_latency_s",
                "avg_estimated_deadline_slack_steps",
                "keep_local_count",
                "sticky_count",
                "migrate_count",
                "offload_count",
                "vrsu_count",
                "vrsu_tasks_active",
                "static_rsu_tasks",
            ],
        )
        _write_csv(
            analysis_dir / "rsu_step_metrics.csv",
            rsu_rows,
            [
                "step",
                "rsu_id",
                "is_vehicular",
                "x",
                "y",
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
            "algorithm": algorithm,
            "log_dataset": log_dataset,
            "pretrained_model_dir": pretrained_model_dir,
            "vehicle_rsu": vehicle_rsu,
            "max_vehicles": max_vehicles,
            "vrsu_capacity": vrsu_capacity,
            "vrsu_range": vrsu_range,
            "vrsu_hardware": vrsu_hardware,
            "run_dir": str(run_dir),
            "raw_dir": str(raw_dir),
            "analysis_dir": str(analysis_dir),
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        }
        (run_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        if dataset_logger is not None:
            dataset_logger.flush()

        if fog.scheduler is not None and hasattr(fog.scheduler.selector, "save"):
            checkpoint_path = analysis_dir / f"{algorithm}_checkpoint.json"
            try:
                fog.scheduler.selector.save(checkpoint_path)
                print(f"[MODEL] Saved final {algorithm} model checkpoint to {checkpoint_path}", flush=True)
            except Exception as se:
                print(f"[MODEL] Warning: Could not save model checkpoint: {se}", flush=True)

        if generate_plots:
            try:
                from analysis.generate_run_plots import generate_plots_for_run
                outputs = generate_plots_for_run(run_dir)
                if outputs:
                    print(f"[PLOT] Generated {len(outputs)} plots under {run_dir / 'plots'}", flush=True)
                else:
                    print("[PLOT] No plots generated from current run data.", flush=True)
            except Exception as pe:
                print(f"[PLOT] Plot generation skipped: {pe}", flush=True)
    finally:
        dash.close()
        try:
            traci.close()
        except Exception:
            pass


def main() -> None:
    p = argparse.ArgumentParser(description="Run SUMO and forward telemetry to a YAFS-based fog sim")
    p.add_argument("--sumo-cfg", default="sumo/delhi/osm.sumocfg", help="Path to SUMO .sumocfg file")
    p.add_argument("--map", dest="map_cfg", default=None, help="Path to SUMO .sumocfg file (alias for --sumo-cfg)")
    p.add_argument("--vehicle", "--vehicles", dest="vehicle", type=int, default=None, help="Total vehicles in the simulation (caps max-num-vehicles in SUMO)")
    p.add_argument("--vehicle-rsu", "--vehicle-rsus", "--vrsu", dest="vehicle_rsu", type=int, default=0, help="Number of volunteered vehicular RSUs (vRSUs)")
    p.add_argument("--vrsu-capacity", type=int, default=15, help="Queue capacity of each vehicular RSU")
    p.add_argument("--vrsu-range", type=float, default=350.0, help="Wireless coverage radius (meters) of vehicular RSUs")
    p.add_argument("--vrsu-hardware", default="jetson_agx_orin", help="Hardware profile for vehicular RSUs")
    p.add_argument("--port", type=int, default=8873, help="TraCI remote port")
    p.add_argument("--start-sumo", action="store_true", help="Start SUMO from this script")
    p.add_argument("--gui", action="store_true", help="Start SUMO GUI when --start-sumo is used")
    p.add_argument("--steps", type=int, default=500, help="Number of simulation steps to run")
    p.add_argument("--step-delay", type=float, default=0.0, help="Delay after each simulation step for GUI playback")
    p.add_argument("--rsu-config", default="sumoXyafs/rsu_config.json", help="Path to RSU anchor configuration JSON")
    p.add_argument("--task-deadline", type=float, default=2.0, help="Task deadline in simulation seconds")
    p.add_argument("--overload-queue-threshold", type=int, default=2, help="Queue depth threshold for RSU overload")
    p.add_argument("--distance-latency-base", type=float, default=0.0, help="Base link latency (s) added to ETA")
    p.add_argument("--distance-latency-max", type=float, default=0.0, help="Latency term (s) at distance == --distance-latency-norm")
    p.add_argument("--distance-latency-norm", type=float, default=500.0, help="Distance (m) scale for uncapped linear latency growth")
    p.add_argument("--distance-error", type=float, default=0.0, help="Relative distance latency error factor (distance_latency += distance_error * distance_latency)")
    p.add_argument("--adaptive-latency-eta", action="store_true", help="Blend measured YAFS link latency into ETA model")
    p.add_argument("--adaptive-latency-alpha", type=float, default=0.25, help="EWMA smoothing factor for measured link latency [0,1]")
    p.add_argument("--adaptive-latency-weight", type=float, default=1.0, help="Multiplier for measured link latency contribution to ETA")
    p.add_argument("--reroute-cooldown", type=float, default=5.0, help="Minimum seconds between reroutes per vehicle")
    p.add_argument("--rsu-capacity-default", type=int, default=20, help="Default RSU capacity if missing in config")
    p.add_argument(
        "--algorithm",
        default="greedy",
        choices=["greedy", "linucb", "thompson", "ridge", "random_forest", "markov", "decision_tree", "q_learning", "kmeans"],
        help="RSU selection and offloading algorithm (default: greedy)",
    )
    p.add_argument("--log-dataset", action="store_true", help="Log (state, action, latency, deadline_met) to offline_dataset.csv for training")
    p.add_argument("--pretrained-model-dir", default="", help="Path to directory containing pre-trained model files")
    p.add_argument("--dashboard", action="store_true", help="Show a basic Tkinter runtime dashboard")
    p.add_argument("--web-dashboard", action="store_true", help="Launch interactive web dashboard server (http://localhost:8080)")
    p.add_argument("--web-port", type=int, default=8080, help="Web dashboard HTTP port (default 8080)")
    p.add_argument("--dashboard-rows", type=int, default=20, help="Max rows shown in dashboard event lists")
    p.add_argument("--results-root", default="results", help="Results root directory (default: sumoXyafs/results)")
    p.add_argument("--run-name", default="", help="Optional run folder name inside results root")
    p.add_argument("--no-plots", action="store_true", help="Disable auto-generation of analysis plots")
    args = p.parse_args()

    if args.map_cfg:
        args.sumo_cfg = args.map_cfg

    if args.web_dashboard:
        import threading
        from web_server import run_web_server, SIM_STATE
        SIM_STATE.algorithm = args.algorithm
        web_thread = threading.Thread(target=run_web_server, kwargs={"port": args.web_port}, daemon=True)
        web_thread.start()
        if args.step_delay == 0.0:
            args.step_delay = 0.2
        if args.steps == 500:
            args.steps = 2000

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
        algorithm=args.algorithm,
        log_dataset=args.log_dataset,
        pretrained_model_dir=args.pretrained_model_dir,
        dashboard=args.dashboard,
        dashboard_rows=args.dashboard_rows,
        results_root=args.results_root,
        run_name=args.run_name,
        generate_plots=not args.no_plots,
        vehicle_rsu=args.vehicle_rsu,
        max_vehicles=args.vehicle,
        vrsu_capacity=args.vrsu_capacity,
        vrsu_range=args.vrsu_range,
        vrsu_hardware=args.vrsu_hardware,
    )


if __name__ == "__main__":
    main()
