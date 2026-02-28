from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.control_policy import ControlPolicy
from modules.models import SchedulingDecision, TaskState, Telemetry


class ControlPolicyTests(unittest.TestCase):
    def test_reroute_respects_cooldown(self) -> None:
        policy = ControlPolicy(reroute_cooldown_s=5.0)
        task = TaskState(
            task_id="t1",
            vehicle_id="v1",
            owner_rsu="rsu_1",
            execution_rsu="rsu_1",
            created_step=0,
            deadline_step=2,
            remaining_compute_s=1.0,
            progress=0.5,
            last_position=(0.0, 0.0),
            last_speed=20.0,
            migrations=0,
        )
        decision = SchedulingDecision(
            vehicle_id="v1",
            observed_rsu="rsu_1",
            owner_rsu="rsu_1",
            execution_rsu="rsu_1",
            action="offload",
            reason="overloaded_exec_rsu",
            estimated_finish_step=3,
            deadline_step=2,
        )

        c1 = policy.build_command(
            Telemetry("v1", position=(0.0, 0.0), speed=18.0, step=0, observed_rsu="rsu_1"),
            decision,
            task,
        )
        self.assertTrue(c1.reroute)

        c2 = policy.build_command(
            Telemetry("v1", position=(1.0, 0.0), speed=18.0, step=1, observed_rsu="rsu_1"),
            decision,
            task,
        )
        self.assertFalse(c2.reroute)


if __name__ == "__main__":
    unittest.main()
