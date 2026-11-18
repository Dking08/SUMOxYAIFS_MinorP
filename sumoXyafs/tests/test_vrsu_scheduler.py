"""Unit tests for Vehicular RSU (vRSU) registration, relative mobility contact duration,
smart assignment, and dynamic handover algorithms."""
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


class VRSURegistryAndSchedulerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.static_rsu1 = RSUNode("rsu_1", node_id=1, x=100.0, y=100.0, capacity=2, coverage_radius=500.0)
        self.static_rsu2 = RSUNode("rsu_2", node_id=2, x=800.0, y=800.0, capacity=2, coverage_radius=500.0)
        self.registry = RSURegistry(rsus=[self.static_rsu1, self.static_rsu2])
        self.tm = TaskManager(task_deadline_s=3.0, compute_time_s=1.0)
        self.scheduler = Scheduler(self.registry, self.tm, overload_threshold=2)

    def test_vrsu_registration_and_update(self) -> None:
        vrsu = self.registry.register_vehicular_rsu(
            rsu_id="vrsu_veh0",
            vehicle_id="veh0",
            x=150.0,
            y=150.0,
            capacity=10,
            coverage_radius=300.0,
            speed=12.0,
            angle=90.0,
        )
        self.assertTrue(vrsu.is_vehicular)
        self.assertEqual(vrsu.vehicle_id, "veh0")
        self.assertEqual(len(self.registry.vehicular_rsus()), 1)
        self.assertEqual(len(self.registry.static_rsus()), 2)

        # Dynamic position update
        self.registry.update_vehicular_rsu_position("vrsu_veh0", x=180.0, y=150.0, speed=15.0, angle=90.0)
        updated = self.registry.get("vrsu_veh0")
        self.assertEqual(updated.x, 180.0)
        self.assertEqual(updated.speed, 15.0)

    def test_contact_duration_same_speed_tandem(self) -> None:
        # Both client and vRSU moving east (angle 90 deg) at 10 m/s, 50m apart
        vrsu = self.registry.register_vehicular_rsu(
            rsu_id="vrsu_v1",
            vehicle_id="v1",
            x=100.0,
            y=100.0,
            coverage_radius=300.0,
            speed=10.0,
            angle=90.0,
        )
        dur = self.registry.calc_contact_duration(
            client_pos=(150.0, 100.0),
            client_speed=10.0,
            client_angle=90.0,
            rsu=vrsu,
        )
        # In tandem relative velocity ~ 0 -> durable contact
        self.assertGreaterEqual(dur, 100.0)

    def test_contact_duration_diverging(self) -> None:
        # vRSU at (0, 0) moving North (0 deg) at 10 m/s; client at (0, 0) moving South (180 deg) at 10 m/s
        # Relative speed is 20 m/s. Coverage radius is 200m.
        # Distance reaches 200m in exactly 10.0 seconds.
        vrsu = self.registry.register_vehicular_rsu(
            rsu_id="vrsu_div",
            vehicle_id="v_div",
            x=0.0,
            y=0.0,
            coverage_radius=200.0,
            speed=10.0,
            angle=0.0,
        )
        dur = self.registry.calc_contact_duration(
            client_pos=(0.0, 0.0),
            client_speed=10.0,
            client_angle=180.0,
            rsu=vrsu,
        )
        self.assertAlmostEqual(dur, 10.0, places=1)

    def test_contact_duration_out_of_coverage_is_zero(self) -> None:
        vrsu = self.registry.register_vehicular_rsu(
            rsu_id="vrsu_far",
            vehicle_id="v_far",
            x=0.0,
            y=0.0,
            coverage_radius=100.0,
        )
        dur = self.registry.calc_contact_duration(
            client_pos=(500.0, 500.0),
            client_speed=10.0,
            client_angle=0.0,
            rsu=vrsu,
        )
        self.assertEqual(dur, 0.0)

    def test_smart_vrsu_assignment_when_static_rsu_overloaded(self) -> None:
        # Register vRSU near client (120, 100)
        self.registry.register_vehicular_rsu(
            rsu_id="vrsu_helper",
            vehicle_id="veh_helper",
            x=120.0,
            y=100.0,
            capacity=10,
            coverage_radius=300.0,
            speed=5.0,
            angle=90.0,
        )
        # Static rsu_1 is overloaded (depth 3 >= threshold 2)
        q_depths = {"rsu_1": 3, "rsu_2": 0, "vrsu_helper": 0}
        t0 = Telemetry(
            vehicle_id="client_1",
            position=(110.0, 100.0),
            speed=5.0,
            step=1,
            observed_rsu="rsu_1",
            angle=90.0,
        )
        decision = self.scheduler.schedule(t0, q_depths)
        self.assertEqual(decision.execution_rsu, "vrsu_helper")
        self.assertEqual(decision.execution_tier, "vRSU")
        self.assertEqual(decision.action, "offload")
        self.assertEqual(decision.reason, "overload_absorbed_by_vrsu")
        self.assertTrue(decision.is_vrsu)

    def test_handover_when_vrsu_coverage_exceeded(self) -> None:
        # Task already assigned to vRSU
        vrsu = self.registry.register_vehicular_rsu(
            rsu_id="vrsu_moving",
            vehicle_id="veh_moving",
            x=100.0,
            y=100.0,
            capacity=10,
            coverage_radius=150.0,
            speed=10.0,
            angle=90.0,
        )
        t0 = Telemetry(
            vehicle_id="c_move",
            position=(100.0, 100.0),
            speed=10.0,
            step=1,
            observed_rsu="rsu_1",
            angle=90.0,
        )
        task = self.tm.upsert_task(t0, owner_rsu="rsu_1")
        self.tm.set_execution_rsu(task, "vrsu_moving", tier="vRSU")

        # Now client drives far away (out of 150m coverage radius)
        t1 = Telemetry(
            vehicle_id="c_move",
            position=(400.0, 100.0),
            speed=15.0,
            step=2,
            observed_rsu="rsu_1",
            angle=90.0,
        )
        decision = self.scheduler.schedule(t1, queue_depths={"rsu_1": 0, "vrsu_moving": 0})
        # Handover triggered: migrated back to static macro RSU
        self.assertEqual(decision.action, "migrate")
        self.assertEqual(decision.reason, "vrsu_coverage_exit")
        self.assertEqual(decision.execution_rsu, "rsu_1")
        self.assertEqual(decision.execution_tier, "V2I")


if __name__ == "__main__":
    unittest.main()

