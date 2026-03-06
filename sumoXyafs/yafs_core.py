from __future__ import annotations

import json
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
    ):
        base_dir = Path(__file__).resolve().parent
        topo_path = Path(topology_path) if topology_path else (base_dir / "minimal_topology.json")
        if not topo_path.exists():
            raise FileNotFoundError(f"Topology file not found: {topo_path}")

        topo_data = json.loads(topo_path.read_text(encoding="utf-8"))
        self.topo = Topology()
        self.topo.load(topo_data)
        self.fog_node = 1

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

        # External queues keep bridge data decoupled from the YAFS internals.
        self.external_queue: Queue = Queue()
        self.feedback_queue: Queue = Queue()
        self.pop = SUMOPopulation(self.app, self.external_queue)

        self.rsu_registry = rsu_registry
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
            )
            self.control_policy = ControlPolicy(reroute_cooldown_s=reroute_cooldown_s)

        selector = First_ShortestPath()
        placement = FogPlacement(
            name="FogPlacement",
            mapping={"Sensor": self.fog_node, "FogProcessor": self.fog_node},
            app=self.app,
        )

        self.sim = Sim(self.topo)
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

    def prune_missing(self, live_vehicle_ids: set[str]) -> None:
        if self.task_manager is None:
            return
        self.task_manager.prune_missing(live_vehicle_ids)

    def active_queue_depths(self) -> dict[str, int]:
        if self.task_manager is None:
            return {}
        return self.task_manager.active_queue_depths()

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
