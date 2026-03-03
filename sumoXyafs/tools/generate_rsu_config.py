"""Generate an initial RSU anchor config from a SUMO net file."""
from __future__ import annotations

import argparse
import gzip
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path


def _open_maybe_gzip(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore")
    return path.open("rt", encoding="utf-8", errors="ignore")


def parse_junctions(net_file: Path) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []

    # Junction coordinates are sufficient to seed region anchors.
    with _open_maybe_gzip(net_file) as stream:
        for _, elem in ET.iterparse(stream, events=("end",)):
            if elem.tag != "junction":
                elem.clear()
                continue
            if elem.attrib.get("type") == "internal":
                elem.clear()
                continue
            x = elem.attrib.get("x")
            y = elem.attrib.get("y")
            if x is None or y is None:
                elem.clear()
                continue
            points.append((float(x), float(y)))
            elem.clear()

    if not points:
        raise ValueError(f"No junction coordinates found in {net_file}")
    return points


def farthest_point_sampling(points: list[tuple[float, float]], count: int) -> list[tuple[float, float]]:
    count = max(1, min(count, len(points)))
    selected = [points[0]]
    remaining = points[1:]

    # Spread anchors across map by iteratively maximizing minimum distance.
    while len(selected) < count and remaining:
        best_idx = 0
        best_score = -1.0
        for idx, point in enumerate(remaining):
            min_dist = min(distance(point, anchor) for anchor in selected)
            if min_dist > best_score:
                best_score = min_dist
                best_idx = idx
        selected.append(remaining.pop(best_idx))

    return selected


def build_neighbors(points: list[tuple[float, float]], k: int = 2) -> list[list[int]]:
    neighbors: list[list[int]] = []
    for i, p in enumerate(points):
        ranked = sorted(
            ((j, distance(p, other)) for j, other in enumerate(points) if j != i),
            key=lambda x: x[1],
        )
        neighbors.append([idx for idx, _ in ranked[:k]])
    return neighbors


def distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate static RSU anchor JSON from SUMO net")
    parser.add_argument("--net-file", default="sumo/delhi/osm.net.xml.gz", help="Path to SUMO .net.xml or .net.xml.gz")
    parser.add_argument("--count", type=int, default=9, help="Number of RSUs to generate")
    parser.add_argument("--output", default="rsu_config.json", help="Output JSON file path")
    parser.add_argument("--capacity", type=int, default=20, help="Default RSU capacity")
    parser.add_argument("--node-ids", default="1,2,3", help="Comma-separated YAFS node ids to cycle through")
    args = parser.parse_args()

    net_file = Path(args.net_file)
    points = parse_junctions(net_file)
    anchors = farthest_point_sampling(points, args.count)
    neighbors = build_neighbors(anchors, k=2)
    node_ids = [int(x.strip()) for x in args.node_ids.split(",") if x.strip()]
    if not node_ids:
        raise ValueError("At least one node id is required")

    rsus = []
    for i, (x, y) in enumerate(anchors):
        rsu_id = f"rsu_{i + 1}"
        rsus.append(
            {
                "id": rsu_id,
                "node_id": node_ids[i % len(node_ids)],
                "x": round(x, 2),
                "y": round(y, 2),
                "capacity": int(args.capacity),
                "neighbors": [f"rsu_{j + 1}" for j in neighbors[i]],
            }
        )

    config = {"defaults": {"capacity": int(args.capacity)}, "rsus": rsus}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Wrote {len(rsus)} RSUs to {out}")


if __name__ == "__main__":
    main()
