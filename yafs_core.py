from yafs.topology import Topology
from yafs.application import Application, Message
from yafs.core import Sim
from yafs.selection import First_ShortestPath
from yafs.placement import Placement
from queue import Queue
from population import SUMOPopulation
import json

class FogPlacement(Placement):
    def __init__(self, name, mapping, app=None):
        super().__init__(name)
        self.mapping = mapping
        self.app = app

    def initial_allocation(self, sim, app_name):
        # Use YAFS APIs to deploy modules properly so alloc_module/alloc_DES are populated.
        # Get services definition from the registered app in sim
        if app_name not in sim.apps:
            # Nothing to do if app not registered
            return
        app = sim.apps[app_name]
        services = getattr(app, "services", {})

        for module, node in self.mapping.items():
            ids = [node] if isinstance(node, int) else list(node)
            # deploy_module will create DES processes and populate sim.alloc_module/alloc_DES
            try:
                sim.deploy_module(app_name, module, services[module], ids)
            except Exception:
                # fallback: ignore and continue
                continue


class MinimalFogSim:
    def __init__(self):
        with open("minimal_topology.json") as f:
            topo_data = json.load(f)

        self.topo = Topology()
        self.topo.load(topo_data)
        self.fog_node = 1

        # App
        self.app = Application(name="vehicular_app")

        # Modules (must be list of dicts!)
        self.app.set_modules([
            {"Sensor": {"Type": Application.TYPE_SOURCE}},
            {"FogProcessor": {"Type": Application.TYPE_SINK}}
        ])

        # Message (src, dst)
        self.msg = Message(
            "telemetry",
            "Sensor",
            "FogProcessor",
            instructions=50,
            bytes=100
        )

        self.app.add_service_module("Sensor", "FogProcessor", self.msg)

        # SUMO → Fog queue
        self.external_queue = Queue()

        # Population consumer
        self.pop = SUMOPopulation(self.app, self.external_queue)

        selector = First_ShortestPath()

        placement = FogPlacement(
            name="FogPlacement",
            mapping={
                "Sensor": self.fog_node,
                "FogProcessor": self.fog_node
            },
            app=self.app,
        )

        self.sim = Sim(self.topo)

        self.sim.deploy_app(
            self.app,
            placement=placement,
            selector=selector,
        )

    def send_to_fog(self, packet):
        self.external_queue.put(packet)

    def tick(self):
        # Advance the DES by one time unit relative to current simulation time
        try:
            current = self.sim.env.now
            self.sim.run(current + 1)
        except Exception:
            # Fallback: attempt a small advance
            try:
                self.sim.run(self.sim.env.now + 0.001)
            except Exception:
                pass
