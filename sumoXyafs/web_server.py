from __future__ import annotations

import json
import os
import sys
import threading
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

# Ensure sumoXyafs modules can be imported
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from modules.models import Telemetry, RSUNode
from modules.rsu_registry import RSURegistry
from modules.scheduler import Scheduler
from modules.task_manager import TaskManager
from modules.wireless_channel import calc_path_loss_db, calc_shannon_rate_bps, calc_sinr_linear, calc_transmission_time_s


class FogSimulatorState:
    def __init__(self, rsu_config_path: str | None = None):
        cfg_path = rsu_config_path or str(BASE_DIR / "rsu_config.json")
        self.registry = RSURegistry.from_json(cfg_path)
        self.task_manager = TaskManager(task_deadline_s=5.0, compute_time_s=2.0)
        self.scheduler = Scheduler(self.registry, self.task_manager, overload_threshold=2)

        self.step = 0
        self.completed_tasks = 0
        self.algorithm = "greedy"
        self.vehicles: dict[str, dict] = {}
        self.queue_depth_overrides: dict[str, int] = {rsu.rsu_id: 0 for rsu in self.registry.all_rsus()}
        self.manual_reassignments: dict[str, str] = {}
        self.recent_decisions: list[dict] = []
        self.auto_step_running = True  # Default running
        self.is_paused = False         # Synced Play/Pause state with SUMO
        self.is_sumo_connected = False
        self.lock = threading.Lock()

        # Queues for two-way sync to live SUMO TraCI loop
        self.pending_spawns: list[dict] = []
        self.pending_add_rsus: list[dict] = []
        self.pending_delete_rsus: list[str] = []

        # Pre-populate sample sandbox vehicles if SUMO is not connected
        self._init_sample_vehicles()

    def _init_sample_vehicles(self):
        sample_positions = [
            ("veh_1", 450.0, 1100.0, "traffic", 90.0),
            ("veh_2", 800.0, 1400.0, "safety", 135.0),
            ("veh_3", 1850.0, 1600.0, "traffic", 45.0),
            ("veh_4", 2200.0, 4100.0, "global", 270.0),
        ]
        for vid, x, y, cat, ang in sample_positions:
            obs = self.registry.nearest_rsu((x, y)).rsu_id
            self.vehicles[vid] = {
                "vehicle_id": vid,
                "x": x,
                "y": y,
                "speed": 12.0,
                "angle": ang,
                "observed_rsu": obs,
                "category": cat,
                "progress": 0.0,
                "task_gflops": 2.0 if cat == "traffic" else (0.5 if cat == "safety" else 50.0),
            }

    def sync_rsus(self, rsu_nodes: list[RSUNode]):
        with self.lock:
            current_ids = {r.rsu_id for r in rsu_nodes}
            self.registry._rsus = list(rsu_nodes)
            self.registry._by_id = {r.rsu_id: r for r in rsu_nodes}
            for r in rsu_nodes:
                if r.rsu_id not in self.queue_depth_overrides:
                    self.queue_depth_overrides[r.rsu_id] = 0
            for rid in list(self.queue_depth_overrides.keys()):
                if rid not in current_ids and rid != "cloud_node":
                    self.queue_depth_overrides.pop(rid, None)

    def sync_sumo_step(self, step_records: list[dict], step_num: int):
        """Synchronizes live vehicle positions and scheduler decisions from SUMO TraCI step loop."""
        with self.lock:
            self.is_sumo_connected = True
            self.step = step_num
            active_vids = set()
            new_decisions = []

            for rec in step_records:
                dec = rec["decision"]
                vid = dec.vehicle_id
                active_vids.add(vid)

                pos = rec.get("position", (0.0, 0.0))
                speed = rec.get("speed", 12.0)
                angle = rec.get("angle", 0.0)
                obs_rsu = dec.observed_rsu
                exec_rsu = dec.execution_rsu
                tier = getattr(dec, "execution_tier", "V2I")

                v_curr = self.vehicles.get(vid, {})
                prev_prog = v_curr.get("progress", 0.0)
                new_prog = min(1.0, prev_prog + 0.1)

                if new_prog >= 1.0:
                    new_prog = 0.0
                    self.completed_tasks += 1

                cat = "safety" if "safety" in vid or tier == "V2V" else ("global" if "heavy" in vid or tier == "CLOUD" else "traffic")
                status_str = f"{dec.action.upper()} ({int(new_prog*100)}%)"

                self.vehicles[vid] = {
                    "vehicle_id": vid,
                    "x": pos[0],
                    "y": pos[1],
                    "speed": speed,
                    "angle": angle,
                    "observed_rsu": obs_rsu,
                    "execution_rsu": exec_rsu,
                    "tier": tier,
                    "category": cat,
                    "progress": new_prog,
                    "status": status_str,
                    "sinr": getattr(dec, "sinr_linear", 100.0),
                    "data_rate": getattr(dec, "data_rate_mbps", 50.0),
                    "trans_latency": getattr(dec, "transmission_latency_s", 0.01),
                }

                new_decisions.append({
                    "step": self.step,
                    "vehicle_id": vid,
                    "observed_rsu": obs_rsu,
                    "execution_rsu": exec_rsu,
                    "action": dec.action,
                    "reason": dec.reason,
                    "tier": tier,
                    "sinr": getattr(dec, "sinr_linear", 100.0),
                    "data_rate_mbps": getattr(dec, "data_rate_mbps", 50.0),
                    "trans_latency_s": getattr(dec, "transmission_latency_s", 0.01),
                    "compute_latency_s": getattr(dec, "compute_latency_s", 0.001),
                    "status": status_str,
                })

            # Retain active SUMO vehicles + manual user spawned vehicles
            self.vehicles = {vid: v for vid, v in self.vehicles.items() if vid in active_vids or vid.startswith("user_")}

            # Fallback assignment for user-spawned or pending vehicles not yet in SUMO step_records
            for vid, vdata in list(self.vehicles.items()):
                if vid not in active_vids:
                    nearest = self.registry.nearest_rsu((vdata["x"], vdata["y"]))
                    obs = nearest.rsu_id
                    exec_target = self.manual_reassignments.get(vid, obs)
                    tier = "CLOUD" if exec_target == "cloud_node" else ("V2V" if exec_target == "local_obu" else "V2I")
                    vdata["observed_rsu"] = obs
                    vdata["execution_rsu"] = exec_target
                    vdata["tier"] = tier
                    vdata["progress"] = min(1.0, vdata.get("progress", 0.0) + 0.1)
                    if vdata["progress"] >= 1.0:
                        vdata["progress"] = 0.0
                        self.completed_tasks += 1
                    vdata["status"] = f"COMPUTING ({int(vdata['progress']*100)}%)"

            if new_decisions:
                self.recent_decisions = (new_decisions + self.recent_decisions)[:30]

    def add_vehicle(self, x: float, y: float, category: str = "traffic") -> str:
        with self.lock:
            vid = f"user_veh_{len(self.vehicles) + 1}_{int(time.time()) % 1000}"
            obs = self.registry.nearest_rsu((x, y)).rsu_id
            vdata = {
                "vehicle_id": vid,
                "x": x,
                "y": y,
                "speed": 10.0,
                "observed_rsu": obs,
                "category": category,
                "progress": 0.0,
                "task_gflops": 2.0 if category == "traffic" else (0.5 if category == "safety" else 50.0),
            }
            self.vehicles[vid] = vdata
            self.pending_spawns.append(vdata)
            return vid

    def add_rsu(self, rsu_id: str, x: float, y: float, hardware_profile: str = "jetson_agx_orin", v2x_protocol: str = "c-v2x", capacity: int = 20):
        with self.lock:
            node = RSUNode(
                rsu_id=rsu_id,
                node_id=len(self.registry._rsus) + 1,
                x=x,
                y=y,
                capacity=capacity,
                hardware_profile=hardware_profile,
                v2x_protocol=v2x_protocol,
            )
            self.registry._rsus.append(node)
            self.registry._by_id[rsu_id] = node
            self.queue_depth_overrides[rsu_id] = 0
            self.pending_add_rsus.append({
                "rsu_id": rsu_id,
                "x": x,
                "y": y,
                "hardware_profile": hardware_profile,
                "v2x_protocol": v2x_protocol,
                "capacity": capacity,
            })

    def delete_rsu(self, rsu_id: str):
        with self.lock:
            if rsu_id in self.registry._by_id:
                node = self.registry._by_id.pop(rsu_id)
                self.registry._rsus = [r for r in self.registry._rsus if r.rsu_id != rsu_id]
                self.queue_depth_overrides.pop(rsu_id, None)
                self.pending_delete_rsus.append(rsu_id)

    def manual_reassign(self, vehicle_id: str, target_rsu: str):
        with self.lock:
            self.manual_reassignments[vehicle_id] = target_rsu

    def set_rsu_queue(self, rsu_id: str, depth: int):
        with self.lock:
            self.queue_depth_overrides[rsu_id] = max(0, int(depth))

    def consume_pending_spawns(self) -> list[dict]:
        with self.lock:
            spawns = list(self.pending_spawns)
            self.pending_spawns.clear()
            return spawns

    def consume_pending_add_rsus(self) -> list[dict]:
        with self.lock:
            rsus = list(self.pending_add_rsus)
            self.pending_add_rsus.clear()
            return rsus

    def consume_pending_delete_rsus(self) -> list[str]:
        with self.lock:
            dels = list(self.pending_delete_rsus)
            self.pending_delete_rsus.clear()
            return dels

    def step_simulation(self):
        with self.lock:
            self.step += 1
            # Move synthetic vehicles if SUMO is not connected
            if not self.is_sumo_connected:
                for vdata in self.vehicles.values():
                    vdata["x"] += (vdata.get("speed", 10.0) * 0.5) % 20.0 - 5.0
                    vdata["observed_rsu"] = self.registry.nearest_rsu((vdata["x"], vdata["y"])).rsu_id

            self._process_step_decisions()

    def _process_step_decisions(self):
        decisions = []
        active_q = dict(self.queue_depth_overrides)

        for vid, vdata in list(self.vehicles.items()):
            obs_rsu = vdata.get("observed_rsu") or self.registry.nearest_rsu((vdata["x"], vdata["y"])).rsu_id
            cat = vdata.get("category", "traffic")

            telemetry = Telemetry(
                vehicle_id=vid,
                position=(vdata["x"], vdata["y"]),
                speed=vdata["speed"],
                step=self.step,
                observed_rsu=obs_rsu,
            )

            st = self.task_manager.upsert_task(telemetry, obs_rsu, category=cat)

            # Explicit Tier-1 V2V (Peer-to-Peer OBU) Trigger for Safety Tasks
            if cat == "safety":
                dec_exec_rsu = "local_obu"
                dec_tier = "V2V"
                dec_action = "v2v_local_obu"
                dec_reason = "safety_critical_local_obu"
                sinr = 1000.0
                rate = 100.0
                trans_lat = 0.0005
                comp_lat = 0.5 / 472.0  # Jetson Nano OBU compute
            elif vid in self.manual_reassignments:
                override_target = self.manual_reassignments[vid]
                dec = self.scheduler.schedule(telemetry, active_q)
                dec_exec_rsu = override_target
                dec_action = "manual_override"
                dec_reason = f"user_dragged_to_{override_target}"
                dec_tier = "CLOUD" if override_target == "cloud_node" else ("V2V" if override_target == "local_obu" else "FOG_MIGRATE")
                sinr, rate, trans_lat, comp_lat = dec.sinr_linear, dec.data_rate_mbps, dec.transmission_latency_s, dec.compute_latency_s
            else:
                dec = self.scheduler.schedule(telemetry, active_q)
                dec_exec_rsu, dec_tier, dec_action, dec_reason = dec.execution_rsu, dec.execution_tier, dec.action, dec.reason
                sinr, rate, trans_lat, comp_lat = dec.sinr_linear, dec.data_rate_mbps, dec.transmission_latency_s, dec.compute_latency_s

            vdata["execution_rsu"] = dec_exec_rsu
            vdata["tier"] = dec_tier
            vdata["sinr"] = sinr
            vdata["data_rate"] = rate
            vdata["trans_latency"] = trans_lat

            # Advance task compute progress realistically
            exec_node = self.registry.get(dec_exec_rsu) if dec_exec_rsu in self.registry._by_id else None
            gflops = exec_node.spec.gflops if exec_node else (472.0 if dec_exec_rsu == "local_obu" else 1000000.0)
            step_gflops = gflops * 0.05
            vdata["progress"] = min(1.0, vdata["progress"] + step_gflops / max(0.1, st.total_gflops))

            status_str = f"COMPUTING ({int(vdata['progress']*100)}%)"
            if vdata["progress"] >= 1.0:
                status_str = "FINISHED ✔"
                self.completed_tasks += 1
                vdata["progress"] = 0.0

            vdata["status"] = status_str

            decisions.append({
                "step": self.step,
                "vehicle_id": vid,
                "observed_rsu": obs_rsu,
                "execution_rsu": dec_exec_rsu,
                "action": dec_action,
                "reason": dec_reason,
                "tier": dec_tier,
                "sinr": sinr,
                "data_rate_mbps": rate,
                "trans_latency_s": trans_lat,
                "compute_latency_s": comp_lat,
                "status": status_str,
            })

        self.recent_decisions = (decisions + self.recent_decisions)[:30]

    def get_snapshot(self) -> dict:
        with self.lock:
            rsus_data = []

            # Calculate active hardware resource load per RSU
            rsu_task_counts = {}
            for v in self.vehicles.values():
                exec_rsu = v.get("execution_rsu", "")
                rsu_task_counts[exec_rsu] = rsu_task_counts.get(exec_rsu, 0) + 1

            for rsu in self.registry.all_rsus():
                connected_links = rsu_task_counts.get(rsu.rsu_id, 0)
                active_q = self.queue_depth_overrides.get(rsu.rsu_id, 0)
                cpu_load_pct = min(100, int(((connected_links + active_q) / max(1, rsu.capacity)) * 100))
                ram_used_mb = (connected_links + active_q) * 256.0

                rsus_data.append({
                    "rsu_id": rsu.rsu_id,
                    "x": rsu.x,
                    "y": rsu.y,
                    "capacity": rsu.capacity,
                    "hardware_profile": rsu.hardware_profile,
                    "protocol": rsu.v2x_protocol,
                    "is_vehicular": bool(getattr(rsu, "is_vehicular", False)),
                    "vehicle_id": getattr(rsu, "vehicle_id", ""),
                    "speed": getattr(rsu, "speed", 0.0),
                    "angle": getattr(rsu, "angle", 0.0),
                    "coverage_radius": float(getattr(rsu, "coverage_radius", 350.0 if getattr(rsu, "is_vehicular", False) else 500.0)),
                    "neighbors": list(getattr(rsu, "neighbors", [])),
                    "gflops": rsu.spec.gflops,
                    "ram_gb": rsu.spec.ram_mb / 1024.0,
                    "ram_used_mb": ram_used_mb,
                    "cpu_load_pct": cpu_load_pct,
                    "queue_depth": active_q,
                    "connected_links": connected_links,
                })

            # Cloud datacenter node
            cloud_links = rsu_task_counts.get("cloud_node", 0)
            rsus_data.append({
                "rsu_id": "cloud_node",
                "x": 3600.0,
                "y": 500.0,
                "capacity": 10000,
                "hardware_profile": "cloud_datacenter",
                "protocol": "wan_5g",
                "is_vehicular": False,
                "vehicle_id": "",
                "speed": 0.0,
                "angle": 0.0,
                "coverage_radius": 5000.0,
                "gflops": 1000000.0,
                "ram_gb": 512.0,
                "ram_used_mb": cloud_links * 512.0,
                "cpu_load_pct": min(100, cloud_links * 2),
                "queue_depth": 0,
                "connected_links": cloud_links,
            })

            tier_counts = {"V2V": 0, "V2I": 0, "vRSU": 0, "FOG_MIGRATE": 0, "CLOUD": 0}
            for v in self.vehicles.values():
                t = v.get("tier", "V2I")
                if t not in tier_counts:
                    tier_counts[t] = 0
                tier_counts[t] = tier_counts.get(t, 0) + 1

            return {
                "step": self.step,
                "algorithm": self.algorithm,
                "sumo_connected": self.is_sumo_connected,
                "is_paused": self.is_paused,
                "completed_tasks": self.completed_tasks,
                "rsus": rsus_data,
                "vehicles": list(self.vehicles.values()),
                "decisions": self.recent_decisions[:15],
                "tier_counts": tier_counts,
                "queue_overrides": self.queue_depth_overrides,
            }


SIM_STATE = FogSimulatorState()


def _background_stepper():
    while True:
        if SIM_STATE.auto_step_running and not SIM_STATE.is_sumo_connected:
            SIM_STATE.step_simulation()
        time.sleep(0.5)


stepper_thread = threading.Thread(target=_background_stepper, daemon=True)
stepper_thread.start()


class DashboardHTTPRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        web_dir = str(BASE_DIR / "web_dashboard")
        super().__init__(*args, directory=web_dir, **kwargs)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/state":
            self._send_json(SIM_STATE.get_snapshot())
        else:
            super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        content_len = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_len).decode("utf-8") if content_len > 0 else "{}"
        try:
            data = json.loads(body)
        except Exception:
            data = {}

        if parsed.path == "/api/spawn_vehicle":
            x = float(data.get("x", 500.0))
            y = float(data.get("y", 500.0))
            cat = str(data.get("category", "traffic"))
            vid = SIM_STATE.add_vehicle(x, y, cat)
            self._send_json({"status": "ok", "vehicle_id": vid})

        elif parsed.path == "/api/add_rsu":
            rsu_id = str(data.get("rsu_id", f"rsu_{len(SIM_STATE.registry.all_rsus())+1}"))
            x = float(data.get("x", 1000.0))
            y = float(data.get("y", 1000.0))
            hw = str(data.get("hardware_profile", "jetson_agx_orin"))
            v2x = str(data.get("v2x_protocol", "c-v2x"))
            cap = int(data.get("capacity", 20))
            SIM_STATE.add_rsu(rsu_id, x, y, hw, v2x, cap)
            self._send_json({"status": "ok", "rsu_id": rsu_id})

        elif parsed.path == "/api/delete_rsu":
            rsu_id = str(data.get("rsu_id", ""))
            SIM_STATE.delete_rsu(rsu_id)
            self._send_json({"status": "ok", "rsu_id": rsu_id})

        elif parsed.path == "/api/reassign_task":
            vid = str(data.get("vehicle_id", ""))
            target = str(data.get("target_rsu", ""))
            SIM_STATE.manual_reassign(vid, target)
            self._send_json({"status": "ok", "vehicle_id": vid, "target": target})

        elif parsed.path == "/api/update_rsu_load":
            rsu_id = str(data.get("rsu_id", ""))
            depth = int(data.get("queue_depth", 0))
            SIM_STATE.set_rsu_queue(rsu_id, depth)
            self._send_json({"status": "ok", "rsu_id": rsu_id, "depth": depth})

        elif parsed.path == "/api/step":
            SIM_STATE.step_simulation()
            self._send_json({"status": "ok", "step": SIM_STATE.step})

        elif parsed.path == "/api/toggle_auto":
            with SIM_STATE.lock:
                SIM_STATE.is_paused = not SIM_STATE.is_paused
                SIM_STATE.auto_step_running = not SIM_STATE.is_paused
            self._send_json({"status": "ok", "auto_running": not SIM_STATE.is_paused, "is_paused": SIM_STATE.is_paused})

        else:
            self.send_error(404, "Endpoint not found")

    def _send_json(self, data: dict):
        body = json.dumps(data).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass


class ReusableHTTPServer(HTTPServer):
    allow_reuse_address = True


def run_web_server(port: int = 8080, host: str = "0.0.0.0"):
    for try_port in range(port, port + 10):
        try:
            server_address = (host, try_port)
            httpd = ReusableHTTPServer(server_address, DashboardHTTPRequestHandler)
            print(f"\n[WEB] Interactive Fog Computing Playground running at http://localhost:{try_port}/", flush=True)
            print("[WEB] Features: Tap-to-Spawn vehicles, Drag-and-Drop task links, Live RSU Load Sliders\n", flush=True)
            httpd.serve_forever()
            break
        except OSError as e:
            if e.errno == 98 and try_port < port + 9:
                print(f"[WEB] Port {try_port} busy, trying {try_port + 1}...", flush=True)
                continue
            raise


if __name__ == "__main__":
    run_web_server(port=8080)
