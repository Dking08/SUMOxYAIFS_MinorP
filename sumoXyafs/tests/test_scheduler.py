from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.models import RSUNode, Telemetry
from modules.rsu_registry import RSURegistry
from modules.scheduler import Scheduler
from modules.task_manager import TaskManager


class SchedulerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = RSURegistry(
            rsus=[
                RSUNode("rsu_1", node_id=1, x=0.0, y=0.0, capacity=2),
                RSUNode("rsu_2", node_id=2, x=100.0, y=0.0, capacity=2),
            ]
        )
        self.tm = TaskManager(task_deadline_s=2.0, compute_time_s=2.0)
        self.scheduler = Scheduler(self.registry, self.tm, overload_threshold=2)

    def test_deadline_risk_triggers_migration(self) -> None:
        t0 = Telemetry(vehicle_id="v1", position=(0.0, 0.0), speed=10.0, step=0, observed_rsu="rsu_1")
        d0 = self.scheduler.schedule(t0, queue_depths={})
        self.assertEqual(d0.action, "keep_local")

        # Heavy queue on old execution RSU should force migrate after region switch.
        t1 = Telemetry(vehicle_id="v1", position=(100.0, 0.0), speed=9.0, step=1, observed_rsu="rsu_2")
        d1 = self.scheduler.schedule(t1, queue_depths={"rsu_1": 3})
        self.assertEqual(d1.action, "migrate")
        self.assertEqual(d1.owner_rsu, "rsu_2")

    def test_overload_triggers_offload(self) -> None:
        t0 = Telemetry(vehicle_id="v2", position=(0.0, 0.0), speed=8.0, step=0, observed_rsu="rsu_1")
        d0 = self.scheduler.schedule(t0, queue_depths={"rsu_1": 3, "rsu_2": 0})
        self.assertEqual(d0.action, "offload")
        self.assertEqual(d0.execution_rsu, "rsu_2")

    def test_distance_latency_is_uncapped_for_far_offload(self) -> None:
        scheduler = Scheduler(
            self.registry,
            TaskManager(task_deadline_s=5.0, compute_time_s=2.0),
            overload_threshold=2,
            distance_latency_base_s=0.0,
            distance_latency_max_s=2.0,
            distance_latency_norm_m=50.0,
        )

        t0 = Telemetry(vehicle_id="v3", position=(0.0, 0.0), speed=8.0, step=0, observed_rsu="rsu_1")
        d0 = scheduler.schedule(t0, queue_depths={"rsu_1": 3, "rsu_2": 0})
        self.assertEqual(d0.action, "offload")
        self.assertEqual(d0.execution_rsu, "rsu_2")
        self.assertEqual(d0.estimated_finish_step, 7)

    def test_adaptive_latency_increases_eta_when_enabled(self) -> None:
        scheduler = Scheduler(
            self.registry,
            TaskManager(task_deadline_s=5.0, compute_time_s=2.0),
            overload_threshold=2,
            distance_latency_base_s=0.0,
            distance_latency_max_s=0.0,
            distance_latency_norm_m=50.0,
            adaptive_latency_eta=True,
            adaptive_latency_alpha=1.0,
            adaptive_latency_weight=2.0,
        )

        scheduler.set_measured_link_latency(0.75)
        d0 = scheduler.schedule(
            Telemetry(vehicle_id="v4", position=(0.0, 0.0), speed=8.0, step=0, observed_rsu="rsu_1"),
            queue_depths={"rsu_1": 0},
        )

        self.assertEqual(d0.action, "keep_local")
        self.assertEqual(d0.estimated_finish_step, 5)

    def test_adaptive_latency_ewma_updates(self) -> None:
        scheduler = Scheduler(
            self.registry,
            TaskManager(task_deadline_s=5.0, compute_time_s=2.0),
            adaptive_latency_eta=True,
            adaptive_latency_alpha=0.5,
            adaptive_latency_weight=1.0,
        )

        scheduler.set_measured_link_latency(1.0)
        scheduler.set_measured_link_latency(0.0)
        d0 = scheduler.schedule(
            Telemetry(vehicle_id="v5", position=(0.0, 0.0), speed=8.0, step=0, observed_rsu="rsu_1"),
            queue_depths={"rsu_1": 0},
        )

        self.assertEqual(d0.estimated_finish_step, 4)


if __name__ == "__main__":
    unittest.main()
