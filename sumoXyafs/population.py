from yafs.population import Population
from yafs.application import Message

class SUMOPopulation(Population):
    def __init__(self, app, external_queue):
        super().__init__("SUMOPop")
        self.app = app
        self.external_queue = external_queue

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

            sim.send_message(msg, self.app.name, packet["time"])
