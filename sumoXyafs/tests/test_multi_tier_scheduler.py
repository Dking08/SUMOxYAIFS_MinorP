"""Unit tests for 4-Tier offloading scheduler."""
import unittest

from modules.models import Telemetry
from modules.rsu_registry import RSURegistry
from modules.scheduler import Scheduler
from modules.task_manager import TaskManager


class TestMultiTierScheduler(unittest.TestCase):
    def setUp(self):
        self.registry = RSURegistry.from_json("sumoXyafs/rsu_config.json")
        self.task_manager = TaskManager(task_deadline_s=2.0, compute_time_s=2.0)
        self.scheduler = Scheduler(self.registry, self.task_manager, overload_threshold=5)

    def test_v2i_standard_scheduling(self):
        telemetry = Telemetry(
            vehicle_id="veh_1",
            position=(450.0, 900.0),
            speed=10.0,
            step=1,
            observed_rsu="rsu_1",
        )
        queue_depths = {}
        decision = self.scheduler.schedule(telemetry, queue_depths)
        self.assertEqual(decision.vehicle_id, "veh_1")
        self.assertIn(decision.execution_tier, ["V2I", "V2V"])
        self.assertGreater(decision.sinr_linear, 0.0)
        self.assertGreater(decision.data_rate_mbps, 0.0)

    def test_fog_to_fog_offloading_when_overloaded(self):
        # Fill rsu_1 queue up to overload threshold (5)
        queue_depths = {"rsu_1": 5}
        telemetry = Telemetry(
            vehicle_id="veh_overloaded",
            position=(450.0, 900.0),
            speed=10.0,
            step=1,
            observed_rsu="rsu_1",
        )
        decision = self.scheduler.schedule(telemetry, queue_depths)
        # Should offload away from rsu_1
        self.assertNotEqual(decision.execution_rsu, "rsu_1")
        self.assertIn(decision.action, ["offload", "migrate"])

    def test_cloud_offload_for_heavy_global_task(self):
        telemetry = Telemetry(
            vehicle_id="veh_heavy",
            position=(450.0, 900.0),
            speed=10.0,
            step=1,
            observed_rsu="rsu_1",
        )
        # Force a heavy global task
        state = self.task_manager.upsert_task(telemetry, owner_rsu="rsu_1", category="global")
        self.assertEqual(state.task_category, "global")

        queue_depths = {}
        decision = self.scheduler.schedule(telemetry, queue_depths)
        self.assertEqual(decision.execution_tier, "CLOUD")
        self.assertEqual(decision.execution_rsu, "cloud_node")


if __name__ == "__main__":
    unittest.main()
