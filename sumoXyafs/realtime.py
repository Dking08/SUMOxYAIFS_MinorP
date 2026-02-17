"""Run SUMO (start or connect) and forward vehicle telemetry to MinimalFogSim."""
from __future__ import annotations

import argparse
import time
import traci
from yafs_core import MinimalFogSim


def run_sumo_with_fog(sumo_cfg: str, port: int = 8873, start_sumo: bool = False, use_gui: bool = False, steps: int = 500) -> None:
    fog = MinimalFogSim()

    # Start SUMO process or connect to an existing one
    if start_sumo:
        # traci.start will set the remote port; pass it via the port= argument
        cmd = ["sumo-gui" if use_gui else "sumo", "-c", sumo_cfg, "--step-length", "1"]
        traci.start(cmd, port=port)
    else:
        # connect to already-running SUMO (must be started with --remote-port)
        # Try a few times in case SUMO is still starting or port not yet accepting
        connected = False
        attempts = 0
        max_attempts = 10
        while not connected and attempts < max_attempts:
            try:
                traci.connect(host="127.0.0.1", port=port)
                connected = True
            except Exception:
                attempts += 1
                if attempts >= max_attempts:
                    raise
                time.sleep(0.5)

    try:
        step = 0
        while step < steps:
            traci.simulationStep()
            for veh in traci.vehicle.getIDList():
                packet = {
                    "id": veh,
                    "position": traci.vehicle.getPosition(veh),
                    "speed": traci.vehicle.getSpeed(veh),
                    "time": step,
                }
                fog.send_to_fog(packet)

            fog.tick()
            # Pull any feedback commands from fog and apply to SUMO via TraCI
            try:
                feedbacks = fog.drain_feedback()
            except Exception:
                feedbacks = []

            for cmd in feedbacks:
                vid = cmd.get("id")
                speed = cmd.get("speed")
                # Apply speed change if vehicle still exists
                try:
                    if vid in traci.vehicle.getIDList():
                        # setSpeed sets the speed for the next simulation step
                        traci.vehicle.setSpeed(vid, float(speed))
                        # lightweight logging for verification
                        print(f"[FOG->SUMO] Applied {cmd}", flush=True)
                except Exception:
                    # ignore any TraCI errors (vehicle may have left)
                    pass
            step += 1
            # yield briefly
            time.sleep(0)
    finally:
        try:
            traci.close()
        except Exception:
            pass


def main() -> None:
    p = argparse.ArgumentParser(description="Run SUMO and forward telemetry to a YAFS-based fog sim")
    p.add_argument("--sumo-cfg", default="sumo/delhi/osm.sumocfg", help="Path to SUMO .sumocfg file")
    p.add_argument("--port", type=int, default=8873, help="TraCI remote port")
    p.add_argument("--start-sumo", action="store_true", help="Start SUMO from this script")
    p.add_argument("--gui", action="store_true", help="Start SUMO GUI when --start-sumo is used")
    p.add_argument("--steps", type=int, default=500, help="Number of simulation steps to run")
    args = p.parse_args()

    run_sumo_with_fog(args.sumo_cfg, port=args.port, start_sumo=args.start_sumo, use_gui=args.gui, steps=args.steps)


if __name__ == "__main__":
    main()
