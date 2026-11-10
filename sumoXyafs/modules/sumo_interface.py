import traci

class SumoInterface:
    def __init__(self, sumo_cfg_path: str, step_length: float = 1.0):
        self.sumo_cfg = sumo_cfg_path
        self.step_length = step_length

    def start(self):
        traci.start(["sumo", "-c", self.sumo_cfg, "--step-length", str(self.step_length)])

    def step(self):
        traci.simulationStep()
        veh_ids = traci.vehicle.getIDList()
        data = []
        for vid in veh_ids:
            pos = traci.vehicle.getPosition(vid)
            speed = traci.vehicle.getSpeed(vid)
            data.append({
                "id": vid,
                "position": pos,
                "speed": speed,
                "time": traci.simulation.getTime()
            })
        return data

    def close(self):
        traci.close()
