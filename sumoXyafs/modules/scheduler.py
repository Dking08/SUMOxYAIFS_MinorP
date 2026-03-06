"""Scheduling policy for RSU ownership, handover, and overload offload."""
from __future__ import annotations

from modules.models import SchedulingDecision, Telemetry
from modules.rsu_registry import RSURegistry
from modules.task_manager import TaskManager


class Scheduler:
    def __init__(
        self,
        rsu_registry: RSURegistry,
        task_manager: TaskManager,
        overload_threshold: int = 20,
        distance_latency_base_s: float = 0.0,
        distance_latency_max_s: float = 0.0,
        distance_latency_norm_m: float = 500.0,
        distance_error: float = 0.0,
        adaptive_latency_eta: bool = False,
        adaptive_latency_alpha: float = 0.25,
        adaptive_latency_weight: float = 1.0,
    ):
        self.rsu_registry = rsu_registry
        self.task_manager = task_manager
        self.overload_threshold = max(1, int(overload_threshold))
        self.distance_latency_base_s = max(0.0, float(distance_latency_base_s))
        self.distance_latency_max_s = max(self.distance_latency_base_s, float(distance_latency_max_s))
        self.distance_latency_norm_m = max(1.0, float(distance_latency_norm_m))
        self.distance_error = max(0.0, float(distance_error))
        self.adaptive_latency_eta = bool(adaptive_latency_eta)
        self.adaptive_latency_alpha = min(1.0, max(0.0, float(adaptive_latency_alpha)))
        self.adaptive_latency_weight = max(0.0, float(adaptive_latency_weight))
        self._measured_link_latency_s = 0.0

    def schedule(
        self,
        telemetry: Telemetry,
        queue_depths: dict[str, int],
    ) -> SchedulingDecision:
        state = self.task_manager.upsert_task(telemetry, telemetry.observed_rsu)

        action = "keep_local"
        reason = "same_region"

        # Apply deadline-aware sticky handover when observed RSU changes.
        if telemetry.observed_rsu != state.owner_rsu:
            current_queue = queue_depths.get(state.execution_rsu, 0)
            sticky_latency = self._network_latency_s(telemetry.position, state.execution_rsu)
            estimated_finish = self.task_manager.estimate_finish_step(
                state,
                current_queue,
                telemetry.step,
                network_latency_s=sticky_latency,
            )
            if estimated_finish <= state.deadline_step:
                action = "sticky"
                reason = "can_finish_before_deadline"
            else:
                self.task_manager.migrate(state, telemetry.observed_rsu, telemetry.observed_rsu)
                action = "migrate"
                reason = "deadline_risk"

        # If the selected execution RSU is overloaded, offload to nearest available RSU.
        execution_rsu = state.execution_rsu
        if self.rsu_registry.is_overloaded(execution_rsu, queue_depths, self.overload_threshold):
            alt = self.rsu_registry.nearest_available_rsu(
                telemetry.position,
                queue_depths,
                self.overload_threshold,
                exclude={execution_rsu},
            )
            if alt is not None:
                self.task_manager.set_execution_rsu(state, alt.rsu_id)
                execution_rsu = alt.rsu_id
                if action == "keep_local":
                    action = "offload"
                    reason = "overloaded_exec_rsu"
                else:
                    reason = f"{reason}+offloaded"

        queue_depths[execution_rsu] = queue_depths.get(execution_rsu, 0) + 1
        exec_latency = self._network_latency_s(telemetry.position, execution_rsu)
        estimated_finish = self.task_manager.estimate_finish_step(
            state,
            queue_depths.get(execution_rsu, 0),
            telemetry.step,
            network_latency_s=exec_latency,
        )

        return SchedulingDecision(
            vehicle_id=telemetry.vehicle_id,
            observed_rsu=telemetry.observed_rsu,
            owner_rsu=state.owner_rsu,
            execution_rsu=execution_rsu,
            action=action,
            reason=reason,
            estimated_finish_step=estimated_finish,
            deadline_step=state.deadline_step,
        )

    def set_measured_link_latency(self, measured_latency_s: float) -> None:
        measured = max(0.0, float(measured_latency_s))
        alpha = self.adaptive_latency_alpha
        self._measured_link_latency_s = ((1.0 - alpha) * self._measured_link_latency_s) + (alpha * measured)

    def _network_latency_s(self, position: tuple[float, float], execution_rsu: str) -> float:
        baseline = self._distance_latency_s(position, execution_rsu)
        if not self.adaptive_latency_eta:
            return baseline
        return baseline + (self._measured_link_latency_s * self.adaptive_latency_weight)

    def _distance_latency_s(self, position: tuple[float, float], execution_rsu: str) -> float:
        if self.distance_latency_max_s <= 0.0:
            return 0.0

        # Use uncapped linear latency growth proportional to RSU distance.
        distance_m = self.rsu_registry.distance_to_rsu(position, execution_rsu)
        normalized = distance_m / self.distance_latency_norm_m
        distance_latency = self.distance_latency_base_s + (self.distance_latency_max_s - self.distance_latency_base_s) * normalized
        return distance_latency + (self.distance_error * distance_latency)
