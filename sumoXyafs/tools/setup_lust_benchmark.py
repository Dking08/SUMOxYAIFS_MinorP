"""Setup Utility for the Luxembourg SUMO Traffic (LuST) Standard Benchmark.

LuST is the de-facto gold-standard microscopic vehicular traffic scenario used
across IEEE vehicular edge computing, V2X, and intelligent transportation literature:
Reference: L. Codeca et al., "Luxembourg SUMO Traffic (LuST) Scenario: 24 Hours of
           Mobility for Large-Scale Evaluating Vehicular Networks," IEEE VNC.
GitHub:    https://github.com/lcodeca/LuSTScenario

This script:
1. Clones or downloads the LuST scenario repository into sumoXyafs/sumo/lust/
2. Extracts the arterial road network (lust.net.xml) and peak morning demand (dua.act.08.rou.xml)
3. Runs farthest-point sampling to seed Roadside Units (RSUs) across the Luxembourg arterial grid
4. Writes lust.sumocfg and rsu_config_lust.json for seamless execution with CoFog-SUMO
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


LUST_REPO_URL = "https://github.com/lcodeca/LuSTScenario.git"


def clone_lust_scenario(target_dir: Path) -> bool:
    """Clone the LuST scenario git repository if not already present."""
    if (target_dir / "scenario" / "lust.net.xml").exists():
        print(f"[+] Found existing LuST scenario files in {target_dir}")
        return True

    git_bin = shutil.which("git")
    if not git_bin:
        print("[-] Git binary not found. Please install Git or clone manually:")
        print(f"    git clone {LUST_REPO_URL} {target_dir}")
        return False

    print(f"[*] Cloning LuST scenario from {LUST_REPO_URL} into {target_dir}...")
    cmd = [git_bin, "clone", "--depth", "1", LUST_REPO_URL, str(target_dir)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode == 0:
        print("[+] LuST repository cloned successfully.")
        return True
    else:
        print(f"[-] Git clone failed: {res.stderr}")
        return False


def parse_junctions(net_file: Path) -> list[tuple[float, float]]:
    """Parse non-internal junctions from LuST net.xml."""
    points = []
    tree = ET.parse(net_file)
    root = tree.getroot()
    for j in root.findall("junction"):
        if j.attrib.get("type") == "internal":
            continue
        x = j.attrib.get("x")
        y = j.attrib.get("y")
        if x and y:
            points.append((float(x), float(y)))
    return points


def farthest_point_sampling(points: list[tuple[float, float]], count: int) -> list[tuple[float, float]]:
    count = max(1, min(count, len(points)))
    selected = [points[0]]
    remaining = points[1:]

    while len(selected) < count and remaining:
        best_idx = 0
        best_score = -1.0
        for idx, point in enumerate(remaining):
            min_dist = min(math.hypot(point[0] - anchor[0], point[1] - anchor[1]) for anchor in selected)
            if min_dist > best_score:
                best_score = min_dist
                best_idx = idx
        selected.append(remaining.pop(best_idx))

    return selected


def build_rsu_config(anchors: list[tuple[float, float]], output_path: Path) -> None:
    rsus = []
    for i, (x, y) in enumerate(anchors):
        rsu_id = f"rsu_{i + 1}"
        dists = []
        for j, other in enumerate(anchors):
            if i != j:
                d = math.hypot(x - other[0], y - other[1])
                dists.append((j, d))
        dists.sort(key=lambda t: t[1])
        neighbors = [f"rsu_{dists[0][0] + 1}", f"rsu_{dists[1][0] + 1}"]

        rsus.append({
            "id": rsu_id,
            "node_id": (i % 3) + 1,
            "x": round(x, 2),
            "y": round(y, 2),
            "capacity": 20,
            "neighbors": neighbors,
        })

    config = {
        "metadata": {
            "scenario": "Luxembourg SUMO Traffic (LuST) Benchmark",
            "citation": "L. Codeca et al., IEEE VNC",
        },
        "defaults": {"capacity": 20},
        "rsus": rsus,
    }
    output_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"[+] Wrote {len(rsus)} LuST benchmark RSUs to {output_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="LuST Standard Benchmark Setup Utility")
    parser.add_argument("--dest", default="sumoXyafs/sumo/lust", help="LuST target directory")
    parser.add_argument("--rsu-count", type=int, default=12, help="Number of RSUs for LuST scenario")
    args = parser.parse_args()

    dest_dir = Path(args.dest)
    dest_dir.mkdir(parents=True, exist_ok=True)

    print("==================================================================")
    print("  CoFog-SUMO: Setting up LuST (Luxembourg) Standard Benchmark     ")
    print("==================================================================")

    cloned = clone_lust_scenario(dest_dir)
    if not cloned:
        print("\n[*] Manual Setup Instructions:")
        print(f"1. Download LuST from: {LUST_REPO_URL}")
        print(f"2. Place scenario/lust.net.xml inside {dest_dir}")
        print("3. Re-run this script to generate RSU anchors and .sumocfg.")
        return

    net_file = dest_dir / "scenario" / "lust.net.xml"
    if not net_file.exists():
        print(f"[-] net file not found at {net_file}")
        return

    junctions = parse_junctions(net_file)
    anchors = farthest_point_sampling(junctions, args.rsu_count)
    rsu_cfg = dest_dir / "rsu_config_lust.json"
    build_rsu_config(anchors, rsu_cfg)

    # SUMO config for LuST peak hour slice
    sumo_cfg = dest_dir / "lust.sumocfg"
    sumocfg_content = """<?xml version="1.0" encoding="UTF-8"?>
<sumoConfiguration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">
    <input>
        <net-file value="scenario/lust.net.xml"/>
        <route-files value="scenario/DUARoutes/local.0.rou.xml"/>
        <additional-files value="scenario/vtypes.add.xml"/>
    </input>
    <processing>
        <ignore-route-errors value="true"/>
        <time-to-teleport value="600"/>
    </processing>
    <report>
        <verbose value="true"/>
        <no-step-log value="true"/>
    </report>
</sumoConfiguration>
"""
    sumo_cfg.write_text(sumocfg_content.strip(), encoding="utf-8")
    print(f"[+] Created {sumo_cfg.name}")
    print("\n[SUCCESS] LuST Benchmark scenario ready!")
    print("Run command: python realtime.py --sumo-cfg sumo/lust/lust.sumocfg --rsu-config sumo/lust/rsu_config_lust.json --start-sumo --gui")


if __name__ == "__main__":
    main()
