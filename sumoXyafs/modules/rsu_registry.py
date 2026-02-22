"""RSU configuration loader and nearest-anchor lookup helpers."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable, Optional

from modules.models import RSUNode


class RSURegistry:
    def __init__(self, rsus: list[RSUNode], capacity_default: int = 20):
        if not rsus:
            raise ValueError("RSU registry cannot be empty")

        self._rsus = rsus
        self._by_id = {r.rsu_id: r for r in rsus}
        self.capacity_default = capacity_default

    @classmethod
    def from_json(cls, config_path: str, capacity_default: int = 20) -> "RSURegistry":
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"RSU config not found: {config_path}")

        data = json.loads(path.read_text(encoding="utf-8"))
        raw_rsus = data.get("rsus", [])
        defaults = data.get("defaults", {})
        default_capacity = int(defaults.get("capacity", capacity_default))

        seen_ids: set[str] = set()
        rsus: list[RSUNode] = []
        for entry in raw_rsus:
            rsu_id = str(entry.get("id", "")).strip()
            if not rsu_id:
                raise ValueError("Each RSU entry must include a non-empty id")
            if rsu_id in seen_ids:
                raise ValueError(f"Duplicate RSU id found: {rsu_id}")
            seen_ids.add(rsu_id)

            if "x" not in entry or "y" not in entry:
                raise ValueError(f"RSU {rsu_id} is missing x/y coordinates")
            x = float(entry["x"])
            y = float(entry["y"])

            node_id = int(entry.get("node_id", 1))
            capacity = int(entry.get("capacity", default_capacity))
            if capacity <= 0:
                raise ValueError(f"RSU {rsu_id} has invalid capacity {capacity}")

            neighbors = [str(x) for x in entry.get("neighbors", [])]
            rsus.append(
                RSUNode(
                    rsu_id=rsu_id,
                    node_id=node_id,
                    x=x,
                    y=y,
                    capacity=capacity,
                    neighbors=neighbors,
                )
            )

        return cls(rsus=rsus, capacity_default=default_capacity)

    def rsu_ids(self) -> list[str]:
        return list(self._by_id.keys())

    def all_rsus(self) -> list[RSUNode]:
        return list(self._rsus)

    def get(self, rsu_id: str) -> RSUNode:
        return self._by_id[rsu_id]

    def nearest_rsu(self, position: tuple[float, float]) -> RSUNode:
        x, y = position
        best: Optional[RSUNode] = None
        best_dist = float("inf")

        # Nearest-anchor selection defines the region assignment.
        for rsu in self._rsus:
            dist = self._distance(x, y, rsu.x, rsu.y)
            if dist < best_dist:
                best = rsu
                best_dist = dist

        if best is None:
            raise RuntimeError("No RSU available in registry")
        return best

    def is_overloaded(self, rsu_id: str, queue_depths: dict[str, int], threshold: int) -> bool:
        rsu = self._by_id[rsu_id]
        queue_depth = int(queue_depths.get(rsu_id, 0))
        hard_limit = min(max(1, threshold), rsu.capacity)
        return queue_depth >= hard_limit

    def nearest_available_rsu(
        self,
        position: tuple[float, float],
        queue_depths: dict[str, int],
        threshold: int,
        exclude: Optional[Iterable[str]] = None,
    ) -> Optional[RSUNode]:
        excluded = set(exclude or [])
        x, y = position
        best: Optional[RSUNode] = None
        best_dist = float("inf")

        # Offload target is chosen by geographic proximity among non-overloaded RSUs.
        for rsu in self._rsus:
            if rsu.rsu_id in excluded:
                continue
            if self.is_overloaded(rsu.rsu_id, queue_depths, threshold):
                continue
            dist = self._distance(x, y, rsu.x, rsu.y)
            if dist < best_dist:
                best = rsu
                best_dist = dist

        return best

    @staticmethod
    def _distance(x1: float, y1: float, x2: float, y2: float) -> float:
        return math.hypot(x1 - x2, y1 - y2)
