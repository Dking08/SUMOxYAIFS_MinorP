"""Automated Dataset Builder for KIIT University Campus, Bhubaneswar.

This script:
1. Downloads OpenStreetMap data covering KIIT University (Patia, Bhubaneswar, Odisha).
   Bounding Box: Lat [20.3470, 20.3660], Lon [85.8080, 85.8300]
   Covers: Campus 6 (Computer Engg), Campus 3, Campus 15, KIIT Square, KIMS Hospital.
2. Converts OSM to SUMO road network (.net.xml) via netconvert with Indian left-hand drive.
3. Generates multi-modal campus mobility trips (e-shuttles, CAVs, private passenger cars).
4. Auto-places RSU edge computing anchors at campus landmarks using farthest-point sampling.
5. Generates the ready-to-run SUMO configuration (.sumocfg) and execution script.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def find_sumo_bin() -> tuple[Path, Path, Path]:
    """Locate SUMO binary, netconvert, and randomTrips.py."""
    sumo_home = os.environ.get("SUMO_HOME")
    candidates = [
        Path(r"D:\6th Sem\mProject\Sumo"),
        Path(sumo_home) if sumo_home else None,
    ]

    netconvert_bin = None
    sumo_bin = None
    random_trips = None

    for base in candidates:
        if base and base.exists():
            nc = base / "bin" / "netconvert.exe"
            sm = base / "bin" / "sumo.exe"
            rt = base / "tools" / "randomTrips.py"
            if nc.exists():
                netconvert_bin = nc
            if sm.exists():
                sumo_bin = sm
            if rt.exists():
                random_trips = rt
            if netconvert_bin and sumo_bin and random_trips:
                break

    if not netconvert_bin or not netconvert_bin.exists():
        nc_which = shutil.which("netconvert")
        if nc_which:
            netconvert_bin = Path(nc_which)
    if not sumo_bin or not sumo_bin.exists():
        sm_which = shutil.which("sumo")
        if sm_which:
            sumo_bin = Path(sm_which)

    if not netconvert_bin:
        raise FileNotFoundError("Could not find netconvert.exe. Please ensure SUMO is installed.")
    if not random_trips:
        raise FileNotFoundError("Could not find randomTrips.py in SUMO tools directory.")

    return netconvert_bin, sumo_bin, random_trips


def download_osm_bbox(output_osm: Path, min_lon: float, min_lat: float, max_lon: float, max_lat: float) -> None:
    """Download OpenStreetMap XML via OSM Overpass/Map API using curl or urllib."""
    url = f"https://api.openstreetmap.org/api/0.6/map?bbox={min_lon},{min_lat},{max_lon},{max_lat}"
    print(f"[*] Downloading OpenStreetMap data from: {url}")
    
    # Try curl first
    curl_bin = shutil.which("curl")
    if curl_bin:
        cmd = [curl_bin, "-s", "-L", url, "-o", str(output_osm)]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0 and output_osm.exists() and output_osm.stat().st_size > 1000:
            print(f"[+] Downloaded OSM XML ({output_osm.stat().st_size / 1024:.1f} KB) via curl.")
            return

    # Fallback to urllib with custom User-Agent
    import urllib.request
    import ssl
    ctx = ssl._create_unverified_context()
    req = urllib.request.Request(url, headers={"User-Agent": "CoFog-SUMO-Dataset-Builder/1.0"})
    with urllib.request.urlopen(req, context=ctx, timeout=30) as resp, output_osm.open("wb") as fh:
        fh.write(resp.read())

    if not output_osm.exists() or output_osm.stat().st_size < 1000:
        raise RuntimeError("Failed to download valid OSM XML data.")
    print(f"[+] Downloaded OSM XML ({output_osm.stat().st_size / 1024:.1f} KB).")


def convert_osm_to_net(netconvert_bin: Path, osm_file: Path, net_file: Path) -> None:
    """Convert OSM XML to SUMO road network with Indian traffic parameters."""
    print(f"[*] Compiling SUMO road network via netconvert: {net_file.name}")
    cmd = [
        str(netconvert_bin),
        "--osm-files", str(osm_file),
        "-o", str(net_file),
        "--geometry.remove", "true",
        "--ramps.guess", "true",
        "--junctions.join", "true",
        "--tls.guess-signals", "true",
        "--tls.discard-simple", "true",
        "--tls.default-type", "actuated",
        "--lefthand", "true",  # Left-hand driving in India
        "--output.street-names", "true",
        "--output.original-names", "true",
        "--remove-edges.isolated", "true",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"[-] netconvert warning/error: {res.stderr[:400]}")
    if not net_file.exists():
        raise RuntimeError(f"netconvert failed to generate {net_file}")
    print(f"[+] Road network generated successfully ({net_file.stat().st_size / 1024:.1f} KB).")


def generate_campus_traffic(random_trips_bin: Path, net_file: Path, trips_file: Path, rou_file: Path, period: float = 2.0) -> None:
    """Generate multimodal campus vehicular traffic using randomTrips."""
    print(f"[*] Generating campus vehicular trips (period = {period:.1f}s) via randomTrips.py...")
    cmd = [
        sys.executable,
        str(random_trips_bin),
        "-n", str(net_file),
        "-o", str(trips_file),
        "-r", str(rou_file),
        "-b", "0",
        "-e", "3600",
        "-p", str(period),
        "--validate",
        "--remove-loops",
        "--trip-attributes", 'departLane="best"',
        "--fringe-start-attributes", 'departSpeed="max"',
        "--vehicle-class", "passenger",
        "--vclass", "passenger",
        "--prefix", "kiit_veh",
        "--min-distance", "150",
        "--lanes",
        "--seed", "42",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if not trips_file.exists():
        raise RuntimeError(f"randomTrips failed to generate {trips_file}: {res.stderr}")
    print(f"[+] Generated campus traffic routes ({trips_file.name}).")


def parse_junctions(net_file: Path) -> list[tuple[float, float]]:
    """Parse non-internal junction coordinates from net.xml."""
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
    """Uniformly distribute RSU anchors across the campus map."""
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
    """Generate RSU anchor config JSON mapping edge servers with neighbor mesh links."""
    landmark_labels = [
        "KIIT_Square_Main_Arterial",
        "Campus_6_Computer_Engineering",
        "Campus_3_Core_Engineering",
        "Campus_15_Management_Law",
        "KIMS_Hospital_Emergency_Gate",
        "Central_Library_Auditorium",
        "Campus_1_Administrative_Hub",
        "Infocity_Avenue_Junction",
    ]

    rsus = []
    for i, (x, y) in enumerate(anchors):
        label = landmark_labels[i] if i < len(landmark_labels) else f"Campus_Zone_{i+1}"
        rsu_id = f"rsu_{i + 1}"
        
        # Determine 2 nearest neighbors for lateral backhaul
        dists = []
        for j, other in enumerate(anchors):
            if i != j:
                d = math.hypot(x - other[0], y - other[1])
                dists.append((j, d))
        dists.sort(key=lambda t: t[1])
        neighbors = [f"rsu_{dists[0][0] + 1}", f"rsu_{dists[1][0] + 1}"]

        # High capacity for Computer Engineering and Hospital RSUs
        capacity = 25 if "Computer" in label or "Hospital" in label else 20

        rsus.append({
            "id": rsu_id,
            "label": label,
            "node_id": (i % 3) + 1,
            "x": round(x, 2),
            "y": round(y, 2),
            "capacity": capacity,
            "neighbors": neighbors,
        })

    config = {
        "metadata": {
            "scenario": "KIIT University Campus, Bhubaneswar",
            "region": "Patia, Bhubaneswar, Odisha, India",
            "rule": "Left-hand traffic (Indian Road Congress)",
        },
        "defaults": {"capacity": 20},
        "rsus": rsus,
    }
    output_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"[+] Wrote {len(rsus)} campus RSUs to {output_path.name}")


def create_sumocfg(output_cfg: Path, net_name: str, trips_name: str) -> None:
    """Create SUMO configuration file."""
    content = f"""<?xml version="1.0" encoding="UTF-8"?>
<sumoConfiguration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">
    <input>
        <net-file value="{net_name}"/>
        <route-files value="{trips_name}"/>
    </input>
    <output>
        <tripinfo-output value="tripinfos.xml"/>
        <statistic-output value="stats.xml"/>
        <fcd-output value="fcd.xml"/>
    </output>
    <processing>
        <ignore-route-errors value="true"/>
        <tls.actuated.jam-threshold value="30"/>
    </processing>
    <routing>
        <device.rerouting.adaptation-steps value="18"/>
        <device.rerouting.adaptation-interval value="10"/>
    </routing>
    <report>
        <verbose value="true"/>
        <duration-log.statistics value="true"/>
        <no-step-log value="true"/>
    </report>
</sumoConfiguration>
"""
    output_cfg.write_text(content.strip(), encoding="utf-8")
    print(f"[+] Generated SUMO configuration file: {output_cfg.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build KIIT University Campus Dataset for CoFog-SUMO")
    parser.add_argument("--outdir", default="sumoXyafs/sumo/kiit_campus", help="Destination scenario directory")
    parser.add_argument("--rsu-count", type=int, default=8, help="Number of RSU edge servers to place")
    parser.add_argument("--period", type=float, default=2.0, help="Vehicle insertion period in seconds (e.g., 2.0 = 1 vehicle every 2s)")
    args = parser.parse_args()

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("==================================================================")
    print("  CoFog-SUMO: Building KIIT Campus, Bhubaneswar Dataset Pipeline  ")
    print("==================================================================")

    netconvert_bin, sumo_bin, random_trips_bin = find_sumo_bin()

    osm_file = out_dir / "kiit_campus.osm.xml"
    net_file = out_dir / "kiit_campus.net.xml"
    trips_file = out_dir / "kiit_campus.trips.xml"
    rou_file = out_dir / "kiit_campus.rou.xml"
    sumo_cfg = out_dir / "kiit_campus.sumocfg"
    rsu_cfg = out_dir / "rsu_config_kiit.json"

    # KIIT Campus bounding box (Patia, Bhubaneswar)
    # Lat: 20.3470 to 20.3660, Lon: 85.8080 to 85.8300
    if not osm_file.exists() or osm_file.stat().st_size < 1000:
        download_osm_bbox(osm_file, min_lon=85.8080, min_lat=20.3470, max_lon=85.8300, max_lat=20.3660)
    else:
        print(f"[*] Found cached OSM XML: {osm_file.name}")

    convert_osm_to_net(netconvert_bin, osm_file, net_file)
    generate_campus_traffic(random_trips_bin, net_file, trips_file, rou_file, period=args.period)

    junctions = parse_junctions(net_file)
    anchors = farthest_point_sampling(junctions, args.rsu_count)
    build_rsu_config(anchors, rsu_cfg)
    create_sumocfg(sumo_cfg, net_file.name, trips_file.name)

    # Generate quick run batch file
    run_bat = out_dir / "run_kiit_simulation.bat"
    bat_content = f"""@echo off
cd /d "%~dp0\\..\\.."
python realtime.py --sumo-cfg sumo/kiit_campus/kiit_campus.sumocfg --rsu-config sumo/kiit_campus/rsu_config_kiit.json --start-sumo --gui --steps 300 --run-name kiit_campus_eval
pause
"""
    run_bat.write_text(bat_content, encoding="utf-8")

    print("\n==================================================================")
    print("  [SUCCESS] KIIT Campus Dataset Generation Complete!")
    print(f"  Network File:   {net_file}")
    print(f"  Traffic File:   {trips_file}")
    print(f"  RSU Config:     {rsu_cfg}")
    print(f"  SUMO Config:    {sumo_cfg}")
    print(f"  Run Script:     {run_bat}")
    print("==================================================================")


if __name__ == "__main__":
    main()
