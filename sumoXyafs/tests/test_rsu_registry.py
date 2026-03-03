from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.models import RSUNode
from modules.rsu_registry import RSURegistry


class RSURegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = RSURegistry(
            rsus=[
                RSUNode("rsu_1", node_id=1, x=0.0, y=0.0, capacity=2),
                RSUNode("rsu_2", node_id=2, x=10.0, y=0.0, capacity=2),
                RSUNode("rsu_3", node_id=3, x=20.0, y=0.0, capacity=2),
            ]
        )

    def test_nearest_rsu(self) -> None:
        rsu = self.registry.nearest_rsu((1.0, 0.2))
        self.assertEqual(rsu.rsu_id, "rsu_1")

    def test_nearest_available_skips_overloaded(self) -> None:
        queue_depths = {"rsu_1": 3, "rsu_2": 0, "rsu_3": 0}
        rsu = self.registry.nearest_available_rsu((0.0, 0.0), queue_depths, threshold=2, exclude={"rsu_1"})
        self.assertIsNotNone(rsu)
        self.assertEqual(rsu.rsu_id, "rsu_2")


if __name__ == "__main__":
    unittest.main()
