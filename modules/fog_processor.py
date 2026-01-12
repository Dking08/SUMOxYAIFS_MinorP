from queue import Queue


class FogProcessor:
    """Simple fog-side processor that inspects telemetry and emits commands.

    It is intentionally synchronous and minimal: if speed > threshold -> slow_down,
    otherwise keep_speed. Decisions are pushed to an internal feedback queue.
    """

    def __init__(self, speed_threshold: float = 10.0, slow_factor: float = 0.8):
        self.speed_threshold = speed_threshold
        self.slow_factor = slow_factor
        self.feedback_queue: Queue = Queue()

    def process(self, packet: dict) -> None:
        vid = packet.get("id")
        speed = packet.get("speed", 0.0)
        pos = packet.get("position")
        t = packet.get("time")

        # Basic rule: if current speed > threshold -> request slow down
        if speed is None:
            return

        if speed > self.speed_threshold:
            new_speed = max(0.0, speed * self.slow_factor)
            cmd = {
                "id": vid,
                "action": "slow_down",
                "speed": new_speed,
                "time": t,
                "position": pos,
            }
            decision = "slow_down"
        else:
            cmd = {
                "id": vid,
                "action": "keep_speed",
                "speed": speed,
                "time": t,
                "position": pos,
            }
            decision = "keep_speed"

        # debug trace for verification
        try:
            print(f"[FPROC] id={vid} speed={speed:.3f} threshold={self.speed_threshold} decision={decision}", flush=True)
        except Exception:
            pass

        self.feedback_queue.put(cmd)

    def drain_feedback(self) -> list:
        out = []
        while not self.feedback_queue.empty():
            out.append(self.feedback_queue.get())
        return out
