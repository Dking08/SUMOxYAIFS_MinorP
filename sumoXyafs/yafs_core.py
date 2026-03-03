from __future__ import annotations

import json
from pathlib import Path
from queue import Queue
from typing import Optional

from yafs.application import Application, Message
from yafs.core import Sim
from yafs.placement import Placement
from yafs.selection import First_ShortestPath
from yafs.topology import Topology

from population import SUMOPopulation


class FogPlacement(Placement):
    def __init__(self, name, mapping, app=None):
        super().__init__(name)
        self.mapping = mapping
        self.app = app

    def initial_allocation(self, sim, app_name):
        if app_name not in sim.apps:
            return
        app = sim.apps[app_name]
        services = getattr(app, "services", {})

        # Deploy modules using YAFS APIs to keep alloc_module/alloc_DES populated.
        for module, node in self.mapping.items():
            ids = [node] if isinstance(node, int) else list(node)
            try:
                sim.deploy_module(app_name, module, services[module], ids)
            except Exception:
                continue


class MinimalFogSim:
    def __init__(self, topology_path: Optional[str] = None):
        base_dir = Path(__file__).resolve().parent
        topo_path = Path(topology_path) if topology_path else (base_dir / "minimal_topology.json")
        if not topo_path.exists():
            raise FileNotFoundError(f"Topology file not found: {topo_path}")

        topo_data = json.loads(topo_path.read_text(encoding="utf-8"))
        self.topo = Topology()
        self.topo.load(topo_data)
        self.fog_node = 1

        self.app = Application(name="vehicular_app")
        self.app.set_modules(
            [
                {"Sensor": {"Type": Application.TYPE_SOURCE}},
                {"FogProcessor": {"Type": Application.TYPE_SINK}},
            ]
        )

        msg = Message(
            "telemetry",
            "Sensor",
            "FogProcessor",
            instructions=50,
            bytes=100,
        )
        self.app.add_service_module("Sensor", "FogProcessor", msg)

        # External queues keep bridge data decoupled from the YAFS internals.
        self.external_queue: Queue = Queue()
        self.feedback_queue: Queue = Queue()
        self.pop = SUMOPopulation(self.app, self.external_queue)

        selector = First_ShortestPath()
        placement = FogPlacement(
            name="FogPlacement",
            mapping={"Sensor": self.fog_node, "FogProcessor": self.fog_node},
            app=self.app,
        )

        self.sim = Sim(self.topo)
        self.sim.deploy_app(self.app, placement=placement, selector=selector)

    def ingest(self, packet: dict) -> None:
        self.external_queue.put(packet)

    def send_to_fog(self, packet: dict) -> None:
        self.ingest(packet)

    def submit_feedback(self, command: dict) -> None:
        self.feedback_queue.put(command)

    def drain_feedback(self) -> list[dict]:
        out: list[dict] = []
        while not self.feedback_queue.empty():
            out.append(self.feedback_queue.get())
        return out

    def tick(self) -> None:
        # Flush incoming telemetry into YAFS before advancing simulation time.
        try:
            self.pop.run(self.sim, self.topo)
        except Exception:
            pass

        # Advance the DES by one simulation-time unit per SUMO step.
        try:
            current = self.sim.env.now
            self.sim.run(current + 1)
        except Exception:
            try:
                self.sim.run(self.sim.env.now + 0.001)
            except Exception:
                pass
