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

            default_hw = str(defaults.get("hardware_profile", "siemens_rs22x"))
            default_v2x = str(defaults.get("v2x_protocol", "c-v2x"))

            neighbors = [str(x) for x in entry.get("neighbors", [])]
            hw_profile = str(entry.get("hardware_profile", default_hw))
            v2x_proto = str(entry.get("v2x_protocol", default_v2x))

            rsus.append(
                RSUNode(
                    rsu_id=rsu_id,
                    node_id=node_id,
                    x=x,
                    y=y,
                    capacity=capacity,
                    neighbors=neighbors,
                    hardware_profile=hw_profile,
                    v2x_protocol=v2x_proto,
                )
            )

        return cls(rsus=rsus, capacity_default=default_capacity)

    def rsu_ids(self) -> list[str]:
        return list(self._by_id.keys())

    def all_rsus(self) -> list[RSUNode]:
        return list(self._rsus)

    def contains(self, rsu_id: str) -> bool:
        return rsu_id in self._by_id

    def add_rsu(self, node: RSUNode):
        if node.rsu_id not in self._by_id:
            self._rsus.append(node)
            self._by_id[node.rsu_id] = node

    def delete_rsu(self, rsu_id: str):
        if rsu_id in self._by_id:
            self._by_id.pop(rsu_id)
            self._rsus = [r for r in self._rsus if r.rsu_id != rsu_id]

    def get(self, rsu_id: str) -> Optional[RSUNode]:
        return self._by_id.get(rsu_id)

    def distance_to_rsu(self, position: tuple[float, float], rsu_id: str) -> float:
        rsu = self._by_id.get(rsu_id)
        if rsu is None:
            if rsu_id == "cloud_node":
                return 5000.0
            return 0.0
        return self._distance(float(position[0]), float(position[1]), rsu.x, rsu.y)

    def static_rsus(self) -> list[RSUNode]:
        return [r for r in self._rsus if not getattr(r, "is_vehicular", False)]

    def vehicular_rsus(self) -> list[RSUNode]:
        return [r for r in self._rsus if getattr(r, "is_vehicular", False)]

    def register_vehicular_rsu(
        self,
        rsu_id: str,
        vehicle_id: str,
        x: float,
        y: float,
        node_id: int = 1,
        capacity: int = 15,
        coverage_radius: float = 350.0,
        hardware_profile: str = "jetson_agx_orin",
        v2x_protocol: str = "c-v2x",
        speed: float = 0.0,
        angle: float = 0.0,
    ) -> RSUNode:
        if rsu_id in self._by_id:
            node = self._by_id[rsu_id]
            node.x = float(x)
            node.y = float(y)
            node.speed = float(speed)
            node.angle = float(angle)
            node.vehicle_id = vehicle_id
            node.coverage_radius = float(coverage_radius)
            return node

        node = RSUNode(
            rsu_id=rsu_id,
            node_id=node_id,
            x=float(x),
            y=float(y),
            capacity=capacity,
            neighbors=[],
            hardware_profile=hardware_profile,
            v2x_protocol=v2x_protocol,
            is_vehicular=True,
            vehicle_id=vehicle_id,
            speed=float(speed),
            angle=float(angle),
            coverage_radius=float(coverage_radius),
        )
        self.add_rsu(node)
        return node

    def update_vehicular_rsu_position(
        self,
        rsu_id: str,
        x: float,
        y: float,
        speed: float = 0.0,
        angle: float = 0.0,
    ) -> None:
        if rsu_id in self._by_id:
            node = self._by_id[rsu_id]
            node.x = float(x)
            node.y = float(y)
            node.speed = float(speed)
            node.angle = float(angle)

    def unregister_vehicular_rsu(self, rsu_id: str) -> None:
        self.delete_rsu(rsu_id)

    def is_in_coverage(self, position: tuple[float, float], rsu_id: str) -> bool:
        node = self._by_id.get(rsu_id)
        if node is None:
            return False
        d = self.distance_to_rsu(position, rsu_id)
        return d <= float(getattr(node, "coverage_radius", 500.0))

    def calc_contact_duration(
        self,
        client_pos: tuple[float, float],
        client_speed: float,
        client_angle: float,
        rsu: RSUNode,
    ) -> float:
        """Calculates predicted contact duration (seconds) before client exits RSU coverage.
        
        Uses relative mobility vector arithmetic with SUMO heading angles (0 deg North/Y+, 90 deg East/X+).
        """
        cov_r = float(getattr(rsu, "coverage_radius", 500.0))
        dx = float(client_pos[0]) - float(rsu.x)
        dy = float(client_pos[1]) - float(rsu.y)
        d = math.hypot(dx, dy)
        if d > cov_r:
            return 0.0

        # Velocity vectors from SUMO angles
        c_rad = math.radians(float(client_angle))
        vcx = float(client_speed) * math.sin(c_rad)
        vcy = float(client_speed) * math.cos(c_rad)

        if getattr(rsu, "is_vehicular", False):
            r_rad = math.radians(float(getattr(rsu, "angle", 0.0)))
            vrx = float(getattr(rsu, "speed", 0.0)) * math.sin(r_rad)
            vry = float(getattr(rsu, "speed", 0.0)) * math.cos(r_rad)
        else:
            vrx, vry = 0.0, 0.0

        rel_vx = vcx - vrx
        rel_vy = vcy - vry
        a = (rel_vx * rel_vx) + (rel_vy * rel_vy)
        if a < 1e-5:
            # Vehicles moving in tandem or nearly stationary relative to each other
            return 120.0

        b = 2.0 * (dx * rel_vx + dy * rel_vy)
        c = (d * d) - (cov_r * cov_r)  # <= 0 since d <= cov_r

        disc = (b * b) - (4.0 * a * c)
        if disc < 0.0:
            return 0.0

        t_exit = (-b + math.sqrt(disc)) / (2.0 * a)
        return max(0.0, float(t_exit))

    def find_best_vrsu(
        self,
        client_pos: tuple[float, float],
        client_speed: float,
        client_angle: float,
        task_gflops: float,
        payload_mbits: float,
        queue_depths: dict[str, int],
        overload_threshold: int,
        safety_margin: float = 0.15,
    ) -> Optional[tuple[RSUNode, float, float]]:
        """Finds the most suitable vehicular RSU based on coverage, contact duration, queue depth and latency.
        
        Returns (best_vrsu_node, contact_duration_s, est_finish_time_s) or None if no suitable vRSU.
        """
        from modules.wireless_channel import calc_transmission_time_s

        vrsus = self.vehicular_rsus()
        if not vrsus:
            return None

        candidates: list[tuple[float, RSUNode, float, float]] = []
        for vrsu in vrsus:
            dist = self.distance_to_rsu(client_pos, vrsu.rsu_id)
            cov_r = float(getattr(vrsu, "coverage_radius", 350.0))
            if dist > cov_r:
                continue

            if self.is_overloaded(vrsu.rsu_id, queue_depths, overload_threshold):
                continue

            contact_dur = self.calc_contact_duration(client_pos, client_speed, client_angle, vrsu)
            trans_s = calc_transmission_time_s(dist, payload_mbits=payload_mbits, protocol=vrsu.protocol_spec)
            q_len = queue_depths.get(vrsu.rsu_id, 0)
            node_gflops = max(100.0, float(vrsu.spec.gflops))
            queue_s = (q_len * task_gflops) / node_gflops
            comp_s = task_gflops / node_gflops
            est_total_s = trans_s + queue_s + comp_s

            # Stability condition: contact duration must comfortably exceed estimated service time
            if contact_dur < (est_total_s * (1.0 + safety_margin)):
                continue

            # Candidate score (lower is better): balance finish time, distance, and relative stability
            score = (
                est_total_s * 0.5
                + (dist / cov_r) * 0.3
                + (float(q_len) / max(1.0, float(vrsu.capacity))) * 0.2
                - min(1.0, contact_dur / 30.0) * 0.15
            )
            candidates.append((score, vrsu, contact_dur, est_total_s))

        if not candidates:
            return None

        candidates.sort(key=lambda item: item[0])
        best_score, best_node, best_contact, best_finish = candidates[0]
        return best_node, best_contact, best_finish

    def nearest_rsu(self, position: tuple[float, float], include_vehicular: bool = False) -> RSUNode:
        x, y = position
        best: Optional[RSUNode] = None
        best_dist = float("inf")

        candidate_pool = self._rsus if include_vehicular else self.static_rsus()
        if not candidate_pool:
            candidate_pool = self._rsus

        for rsu in candidate_pool:
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
        include_vehicular: bool = True,
    ) -> Optional[RSUNode]:
        excluded = set(exclude or [])
        x, y = position
        best: Optional[RSUNode] = None
        best_dist = float("inf")

        pool = self._rsus if include_vehicular else self.static_rsus()
        # Offload target is chosen by geographic proximity among non-overloaded RSUs.
        for rsu in pool:
            if rsu.rsu_id in excluded:
                continue
            if getattr(rsu, "is_vehicular", False):
                # Vehicular RSU must be within its coverage radius
                if self._distance(x, y, rsu.x, rsu.y) > getattr(rsu, "coverage_radius", 350.0):
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
