"""Fog-side control policy that maps scheduling state to SUMO commands."""
from __future__ import annotations

from modules.models import ControlCommand, SchedulingDecision, TaskState, Telemetry


class ControlPolicy:
    def __init__(
        self,
        speed_threshold: float = 12.0,
        slow_factor: float = 0.85,
        reroute_speed_threshold: float = 14.0,
        reroute_cooldown_s: float = 5.0,
    ):
        self.speed_threshold = float(speed_threshold)
        self.slow_factor = float(slow_factor)
        self.reroute_speed_threshold = float(reroute_speed_threshold)
        self.reroute_cooldown_s = float(reroute_cooldown_s)
        self._last_reroute_step: dict[str, int] = {}

    def build_command(
        self,
        telemetry: Telemetry,
        decision: SchedulingDecision,
        task: TaskState,
    ) -> ControlCommand:
        target_speed = float(telemetry.speed)
        action = "keep_speed"
        reasons: list[str] = []

        # The baseline control rule slows vehicles above threshold.
        if telemetry.speed > self.speed_threshold:
            target_speed = max(0.0, telemetry.speed * self.slow_factor)
            action = "slow_down"
            reasons.append("speed_threshold")

        # Handover or overload introduces uncertainty; apply a softer cap.
        if decision.action in {"migrate", "offload"}:
            target_speed = min(target_speed, max(2.0, telemetry.speed * 0.9))
            reasons.append(decision.action)

        # When deadline is already missed, apply stricter speed reduction.
        if telemetry.step > task.deadline_step:
            target_speed = min(target_speed, max(1.0, telemetry.speed * 0.7))
            reasons.append("deadline_missed")

        reroute = self._should_reroute(telemetry, decision)
        if reroute:
            reasons.append("reroute")

        reason = ",".join(reasons) if reasons else "nominal"
        return ControlCommand(
            vehicle_id=telemetry.vehicle_id,
            action=action,
            speed=target_speed,
            step=telemetry.step,
            position=telemetry.position,
            source_rsu=decision.execution_rsu,
            reroute=reroute,
            reason=reason,
            task_id=task.task_id,
        )

    def _should_reroute(self, telemetry: Telemetry, decision: SchedulingDecision) -> bool:
        if decision.action not in {"migrate", "offload"} and telemetry.speed <= self.reroute_speed_threshold:
            return False

        last = self._last_reroute_step.get(telemetry.vehicle_id)
        if last is not None and (telemetry.step - last) < self.reroute_cooldown_s:
            return False

        self._last_reroute_step[telemetry.vehicle_id] = telemetry.step
        return True
