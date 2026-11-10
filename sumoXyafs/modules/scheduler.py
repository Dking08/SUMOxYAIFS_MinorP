"""Scheduling policy for RSU ownership, handover, and overload offload."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

from modules.algorithms import DatasetLogger, RSUSelector, get_algorithm
from modules.models import DecisionFeedback, OffloadingState, SchedulingDecision, Telemetry
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
        algorithm: Optional[str | RSUSelector] = None,
        dataset_logger: Optional[DatasetLogger] = None,
        pretrained_model_dir: Optional[str | Path] = None,
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

        # Algorithmic selector and dataset logger
        self.algorithm_name = "greedy" if algorithm is None else (algorithm if isinstance(algorithm, str) else algorithm.name)
        self.selector = get_algorithm(
            algorithm,
            overload_threshold=self.overload_threshold,
            distance_latency_norm_m=self.distance_latency_norm_m,
            distance_latency_max_s=self.distance_latency_max_s,
        )
        self.dataset_logger = dataset_logger

        if pretrained_model_dir is not None:
            self._try_load_pretrained(Path(pretrained_model_dir))

    def _try_load_pretrained(self, model_dir: Path) -> None:
        if not model_dir.exists():
            return
        model_files = {
            "ridge": model_dir / "ridge_weights.json",
            "random_forest": model_dir / "random_forest_model.json",
            "markov": model_dir / "markov_transitions.json",
            "decision_tree": model_dir / "decision_tree_rules.json",
            "linucb": model_dir / "linucb_model.json",
            "q_learning": model_dir / "q_learning_model.json",
            "thompson": model_dir / "thompson_model.json",
        }
        target_file = model_files.get(self.selector.name)
        if target_file and target_file.exists():
            try:
                self.selector.load(target_file)
            except Exception:
                pass

    def schedule(
        self,
        telemetry: Telemetry,
        queue_depths: dict[str, int],
    ) -> SchedulingDecision:
        state = self.task_manager.upsert_task(telemetry, telemetry.observed_rsu)

        # Build candidate RSU distances and state snapshot
        rsu_distances = {r.rsu_id: self.rsu_registry.distance_to_rsu(telemetry.position, r.rsu_id) for r in self.rsu_registry.all_rsus()}
        rsu_capacities = {r.rsu_id: r.capacity for r in self.rsu_registry.all_rsus()}
        candidate_rsus = self.rsu_registry.rsu_ids()

        offload_state = OffloadingState(
            vehicle_id=telemetry.vehicle_id,
            position=telemetry.position,
            speed=telemetry.speed,
            step=telemetry.step,
            observed_rsu=telemetry.observed_rsu,
            owner_rsu=state.owner_rsu,
            execution_rsu=state.execution_rsu,
            queue_depths=dict(queue_depths),
            rsu_distances=rsu_distances,
            rsu_capacities=rsu_capacities,
            task_id=state.task_id,
            remaining_compute_s=state.remaining_compute_s,
            deadline_step=state.deadline_step,
            deadline_slack_steps=state.deadline_step - telemetry.step,
            heading_deg=0.0,
            candidate_rsus=candidate_rsus,
        )

        action = "keep_local"
        reason = "same_region"

        if self.algorithm_name == "greedy":
            # Baseline greedy scheduling
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
        else:
            # Algorithmic decision via unified choose_rsu interface
            chosen_rsu = self.selector.choose_rsu(offload_state)
            if not chosen_rsu or chosen_rsu not in candidate_rsus:
                chosen_rsu = telemetry.observed_rsu

            if chosen_rsu == state.execution_rsu:
                if chosen_rsu == telemetry.observed_rsu:
                    action = "keep_local"
                    reason = f"{self.selector.name}_local"
                else:
                    action = "sticky"
                    reason = f"{self.selector.name}_sticky"
            else:
                if chosen_rsu == telemetry.observed_rsu:
                    self.task_manager.migrate(state, telemetry.observed_rsu, telemetry.observed_rsu)
                    action = "migrate"
                    reason = f"{self.selector.name}_migrate"
                else:
                    self.task_manager.set_execution_rsu(state, chosen_rsu)
                    action = "offload"
                    reason = f"{self.selector.name}_offload"

            execution_rsu = chosen_rsu

        queue_depths[execution_rsu] = queue_depths.get(execution_rsu, 0) + 1
        exec_latency = self._network_latency_s(telemetry.position, execution_rsu)
        estimated_finish = self.task_manager.estimate_finish_step(
            state,
            queue_depths.get(execution_rsu, 0),
            telemetry.step,
            network_latency_s=exec_latency,
        )

        total_latency_s = float(estimated_finish - telemetry.step)
        deadline_met = bool(estimated_finish <= state.deadline_step)
        reward = 1.0 if deadline_met else -1.0

        feedback = DecisionFeedback(
            state=offload_state,
            action_rsu=execution_rsu,
            latency_s=total_latency_s,
            deadline_met=deadline_met,
            reward=reward,
            step=telemetry.step,
            vehicle_id=telemetry.vehicle_id,
        )

        # Online algorithm learning update
        self.selector.update(feedback)

        # Optional dataset logging for offline training
        if self.dataset_logger is not None:
            self.dataset_logger.log(feedback)

        return SchedulingDecision(
            vehicle_id=telemetry.vehicle_id,
            observed_rsu=telemetry.observed_rsu,
            owner_rsu=state.owner_rsu,
            execution_rsu=execution_rsu,
            action=action,
            reason=reason,
            estimated_finish_step=estimated_finish,
            deadline_step=state.deadline_step,
            estimated_network_latency_s=exec_latency,
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
