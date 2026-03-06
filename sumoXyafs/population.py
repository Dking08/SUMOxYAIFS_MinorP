from yafs.population import Population
from yafs.application import Message
from typing import Optional
from modules.rsu_registry import RSURegistry

class SUMOPopulation(Population):
    def __init__(self, app, external_queue, rsu_registry: Optional[RSURegistry] = None):
        super().__init__("SUMOPop")
        self.app = app
        self.external_queue = external_queue
        self.rsu_registry = rsu_registry

        # Source control registration
        self.set_src_control({
            "id": "SUMOPop",
            "app": app.name,
            "module": "Sensor",
            "message": "telemetry"
        })

    def run(self, sim, topology):
        while not self.external_queue.empty():
            packet = self.external_queue.get()

            msg = Message(
                "telemetry",
                src="Sensor",
                dst="FogProcessor",
                instructions=50,
                bytes=100
            )
            msg.timestamp = float(packet.get("time", 0.0))

            source_des = self._resolve_source_des(sim, packet)
            if source_des is None:
                continue

            msg.original_DES_src = source_des
            if hasattr(sim, "_Sim__send_message"):
                sim._Sim__send_message(self.app.name, msg, source_des, sim.SOURCE_METRIC)
            elif hasattr(sim, "send_message"):
                sim.send_message(msg, self.app.name, source_des)

    def _resolve_source_des(self, sim, packet: dict):
        modules = getattr(sim, "alloc_module", {})
        app_modules = modules.get(self.app.name, {}) if isinstance(modules, dict) else {}
        sensor_des_ids = app_modules.get("Sensor", []) if isinstance(app_modules, dict) else []
        if not sensor_des_ids:
            return None

        # Prefer source DES deployed on the observed RSU node to preserve topology semantics.
        region_id = str(packet.get("region_id", ""))
        if self.rsu_registry is not None and region_id:
            try:
                node_id = self.rsu_registry.get(region_id).node_id
                for des_id in sensor_des_ids:
                    if sim.alloc_DES.get(des_id) == node_id:
                        return des_id
            except Exception:
                pass

        return sensor_des_ids[0]
