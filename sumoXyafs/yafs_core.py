from __future__ import annotations

import json
import csv
from pathlib import Path
from queue import Queue
from typing import Optional

from yafs.application import Application, Message
from yafs.core import Sim
from yafs.placement import Placement
from yafs.selection import First_ShortestPath
from yafs.topology import Topology

from modules.control_policy import ControlPolicy
from modules.models import SchedulingDecision, Telemetry
from modules.rsu_registry import RSURegistry
from modules.scheduler import Scheduler
from modules.task_manager import TaskManager
from population import SUMOPopulation


class FogPlacement(Placement):
    def __init__(self, name, mapping, app=None):
        super().__init__(name)
        self.mapping = mapping
        self.app = app

    def initial_allocation(self, sim, app_name):
        if app_name not in sim.apps:
            return
        app = sim.apps[app_name]
        services = getattr(app, "services", {})

        # Deploy modules using YAFS APIs to keep alloc_module/alloc_DES populated.
        for module, node in self.mapping.items():
            ids = [node] if isinstance(node, int) else list(node)
            try:
                sim.deploy_module(app_name, module, services[module], ids)
            except Exception:
                continue


class MinimalFogSim:
    def __init__(
        self,
        topology_path: Optional[str] = None,
        rsu_registry: Optional[RSURegistry] = None,
        task_deadline_s: float = 2.0,
        compute_time_s: float = 2.0,
        overload_threshold: int = 20,
        reroute_cooldown_s: float = 5.0,
        distance_latency_base_s: float = 0.0,
        distance_latency_max_s: float = 0.0,
        distance_latency_norm_m: float = 500.0,
        adaptive_latency_eta: bool = False,
        adaptive_latency_alpha: float = 0.25,
        adaptive_latency_weight: float = 1.0,
        results_prefix: Optional[str] = None,
    ):
        base_dir = Path(__file__).resolve().parent
        topo_path = Path(topology_path) if topology_path else (base_dir / "minimal_topology.json")
        if not topo_path.exists():
            raise FileNotFoundError(f"Topology file not found: {topo_path}")

        topo_data = json.loads(topo_path.read_text(encoding="utf-8"))
        self.topo = Topology()
        self.topo.load(topo_data)
        self.fog_node = 1

        default_prefix = base_dir / "results" / "yafs_runtime"
        self.results_prefix = Path(results_prefix) if results_prefix else default_prefix
        self.results_prefix.parent.mkdir(parents=True, exist_ok=True)
        self._events_csv = Path(f"{self.results_prefix}.csv")
        self._links_csv = Path(f"{self.results_prefix}_link.csv")
        self._event_cursor = 0
        self._link_cursor = 0
        self._last_metrics_snapshot = {
            "events": 0,
            "links": 0,
            "avg_end_to_end_s": 0.0,
            "avg_service_s": 0.0,
            "avg_link_latency_s": 0.0,
            "messages": 0,
        }

        self.app = Application(name="vehicular_app")
        self.app.set_modules(
            [
                {"Sensor": {"Type": Application.TYPE_SOURCE}},
                {"FogProcessor": {"Type": Application.TYPE_SINK}},
            ]
        )

        msg = Message(
            "telemetry",
            "Sensor",
            "FogProcessor",
            instructions=50,
            bytes=100,
        )
        self.app.add_service_module("Sensor", "FogProcessor", msg)

        self.rsu_registry = rsu_registry
        self.processor_nodes = self._resolve_processor_nodes()

        # External queues keep bridge data decoupled from the YAFS internals.
        self.external_queue: Queue = Queue()
        self.feedback_queue: Queue = Queue()
        self.pop = SUMOPopulation(self.app, self.external_queue, rsu_registry=self.rsu_registry)

        self.task_manager = None
        self.scheduler = None
        self.control_policy = None
        if self.rsu_registry is not None:
            self.task_manager = TaskManager(task_deadline_s=task_deadline_s, compute_time_s=compute_time_s)
            self.scheduler = Scheduler(
                self.rsu_registry,
                self.task_manager,
                overload_threshold=overload_threshold,
                distance_latency_base_s=distance_latency_base_s,
                distance_latency_max_s=distance_latency_max_s,
                distance_latency_norm_m=distance_latency_norm_m,
                adaptive_latency_eta=adaptive_latency_eta,
                adaptive_latency_alpha=adaptive_latency_alpha,
                adaptive_latency_weight=adaptive_latency_weight,
            )
            self.control_policy = ControlPolicy(reroute_cooldown_s=reroute_cooldown_s)

        selector = First_ShortestPath()
        placement = FogPlacement(
            name="FogPlacement",
            mapping={"Sensor": self.processor_nodes, "FogProcessor": self.processor_nodes},
            app=self.app,
        )

        self.sim = Sim(self.topo, default_results_path=str(self.results_prefix))
        self.sim.deploy_app(self.app, placement=placement, selector=selector)

    def ingest(self, packet: dict) -> None:
        self.external_queue.put(packet)

    def send_to_fog(self, packet: dict) -> None:
        self.ingest(packet)

    def submit_feedback(self, command: dict) -> None:
        self.feedback_queue.put(command)

    def drain_feedback(self) -> list[dict]:
        out: list[dict] = []
        while not self.feedback_queue.empty():
            out.append(self.feedback_queue.get())
        return out

    def process_step(
        self,
        telemetry_batch: list[Telemetry],
        step: int,
    ) -> list[dict]:
        if self.scheduler is None or self.task_manager is None or self.control_policy is None:
            raise RuntimeError("MinimalFogSim was created without RSU-aware orchestration components")

        queue_depths = self.task_manager.active_queue_depths()
        records: list[dict] = []

        # Keep telemetry ingress explicit so YAFS remains the transport/compute timeline.
        for telemetry in telemetry_batch:
            normalized = self._normalize_telemetry(telemetry)
            decision = self.scheduler.schedule(normalized, queue_depths)
            task = self.task_manager.get(normalized.vehicle_id)
            if task is None:
                continue

            packet = {
                "id": normalized.vehicle_id,
                "position": normalized.position,
                "speed": normalized.speed,
                "time": normalized.step,
                "region_id": normalized.observed_rsu,
                "assigned_rsu": decision.execution_rsu,
                "owner_rsu": decision.owner_rsu,
                "task_id": task.task_id,
                "deadline_step": task.deadline_step,
                "scheduler_action": decision.action,
                "scheduler_reason": decision.reason,
            }
            self.ingest(packet)
            command = self.control_policy.build_command(normalized, decision, task)
            feedback = command.to_packet()
            self.submit_feedback(feedback)
            records.append({"decision": decision, "feedback": feedback})

        self.tick()
        feedbacks = self.drain_feedback()
        feedback_by_vehicle = {str(fb.get("id", "")): fb for fb in feedbacks}
        self._last_metrics_snapshot = self._collect_metrics_snapshot()
        if self.scheduler is not None:
            self.scheduler.set_measured_link_latency(self._last_metrics_snapshot.get("avg_link_latency_s", 0.0))

        # Match legacy behavior: task completion advances once per outer SUMO step.
        self.task_manager.tick_all(step_size_s=1.0)
        current_queue_depths = self.task_manager.active_queue_depths()

        out: list[dict] = []
        for record in records:
            decision = record["decision"]
            fallback_feedback = record["feedback"]
            feedback = feedback_by_vehicle.get(decision.vehicle_id, fallback_feedback)
            out.append(
                {
                    "decision": decision,
                    "feedback": feedback,
                    "queue_depths": current_queue_depths,
                    "metrics": self._last_metrics_snapshot,
                    "step": step,
                }
            )
        return out

    def _normalize_telemetry(self, telemetry: Telemetry) -> Telemetry:
        observed_rsu = telemetry.observed_rsu
        if not observed_rsu:
            observed_rsu = self.rsu_registry.nearest_rsu(telemetry.position).rsu_id

        return Telemetry(
            vehicle_id=telemetry.vehicle_id,
            position=telemetry.position,
            speed=telemetry.speed,
            step=telemetry.step,
            observed_rsu=observed_rsu,
        )

    def _resolve_processor_nodes(self) -> list[int]:
        if self.rsu_registry is None:
            return [self.fog_node]

        known_nodes = set(self.topo.G.nodes)
        nodes = sorted({r.node_id for r in self.rsu_registry.all_rsus() if r.node_id in known_nodes})
        return nodes if nodes else [self.fog_node]

    def prune_missing(self, live_vehicle_ids: set[str]) -> None:
        if self.task_manager is None:
            return
        self.task_manager.prune_missing(live_vehicle_ids)

    def active_queue_depths(self) -> dict[str, int]:
        if self.task_manager is None:
            return {}
        return self.task_manager.active_queue_depths()

    def last_metrics_snapshot(self) -> dict:
        return dict(self._last_metrics_snapshot)

    def tick(self) -> None:
        # Flush incoming telemetry into YAFS before advancing simulation time.
        try:
            self.pop.run(self.sim, self.topo)
        except Exception:
            pass

        # Advance the DES by one simulation-time unit per SUMO step.
        try:
            current = self.sim.env.now
            self.sim.run(current + 1)
        except Exception:
            try:
                self.sim.run(self.sim.env.now + 0.001)
            except Exception:
                pass

    def _collect_metrics_snapshot(self) -> dict:
        events = self._read_new_csv_rows(self._events_csv, is_event=True)
        links = self._read_new_csv_rows(self._links_csv, is_event=False)

        avg_end_to_end = 0.0
        avg_service = 0.0
        avg_link_latency = 0.0
        message_count = 0

        if events:
            total_end_to_end = 0.0
            total_service = 0.0
            for row in events:
                try:
                    emit = float(row.get("time_emit", 0.0))
                    reception = float(row.get("time_reception", 0.0))
                    service = float(row.get("service", 0.0))
                except Exception:
                    continue
                total_end_to_end += max(0.0, reception - emit)
                total_service += max(0.0, service)
                message_count += 1

            if message_count > 0:
                avg_end_to_end = total_end_to_end / message_count
                avg_service = total_service / message_count

        if links:
            total_link_latency = 0.0
            link_count = 0
            for row in links:
                try:
                    latency = float(row.get("latency", 0.0))
                except Exception:
                    continue
                total_link_latency += max(0.0, latency)
                link_count += 1
            if link_count > 0:
                avg_link_latency = total_link_latency / link_count

        return {
            "events": len(events),
            "links": len(links),
            "avg_end_to_end_s": avg_end_to_end,
            "avg_service_s": avg_service,
            "avg_link_latency_s": avg_link_latency,
            "messages": message_count,
        }

    def _read_new_csv_rows(self, csv_path: Path, is_event: bool) -> list[dict]:
        if not csv_path.exists():
            return []

        # Flush latest YAFS writes before reading incremental traces.
        try:
            self.sim.metrics.flush()
        except Exception:
            pass

        with csv_path.open("r", encoding="utf-8", newline="") as fh:
            reader = list(csv.DictReader(fh))

        if is_event:
            start = min(self._event_cursor, len(reader))
            self._event_cursor = len(reader)
        else:
            start = min(self._link_cursor, len(reader))
            self._link_cursor = len(reader)

        return reader[start:]
