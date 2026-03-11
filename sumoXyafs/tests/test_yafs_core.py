from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.models import RSUNode
from modules.rsu_registry import RSURegistry


class MinimalFogSimInitTests(unittest.TestCase):
    def test_init_with_rsu_registry_does_not_raise_attribute_error(self) -> None:
        try:
            from yafs_core import MinimalFogSim
        except ModuleNotFoundError as exc:
            self.skipTest(f"yafs dependency unavailable in test environment: {exc}")

        registry = RSURegistry(
            rsus=[
                RSUNode("rsu_1", node_id=1, x=0.0, y=0.0, capacity=2),
                RSUNode("rsu_2", node_id=2, x=100.0, y=0.0, capacity=2),
            ]
        )

        sim = MinimalFogSim(
            rsu_registry=registry,
            results_prefix=str(ROOT / "results" / "test_yafs_runtime"),
        )

        self.assertIsNotNone(sim.rsu_registry)
        self.assertTrue(sim.processor_nodes)


if __name__ == "__main__":
    unittest.main()
