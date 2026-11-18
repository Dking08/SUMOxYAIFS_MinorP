"""Realistic 4-Tier Scheduling policy for V2V, V2I, Inter-RSU Fog-to-Fog, and Fog-to-Cloud offloading, with pluggable online and offline algorithms."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from modules.algorithms import DatasetLogger, RSUSelector, get_algorithm
from modules.models import DecisionFeedback, OffloadingState, SchedulingDecision, Telemetry
from modules.rsu_registry import RSURegistry
from modules.task_manager import TaskManager
from modules.wireless_channel import (
    calc_cloud_latency_s,
    calc_inter_rsu_latency_s,
    calc_shannon_rate_bps,
    calc_sinr_linear,
    calc_transmission_time_s,
)


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
        delayed_feedback: bool = False,
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
        self.delayed_feedback = bool(delayed_feedback)
        self._measured_link_latency_s = 0.0

        # Algorithmic selector and dataset logger
        self.algorithm_name = "greedy" if algorithm is None else (algorithm if isinstance(algorithm, str) else getattr(algorithm, "name", "custom"))
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

    def prepare_step(
        self,
        telemetry_batch: list[Telemetry],
        queue_depths: dict[str, int],
        step: int,
    ) -> None:
        """Invoked at the beginning of each simulation step before individual scheduling."""
        if hasattr(self.selector, "prepare_step"):
            try:
                self.selector.prepare_step(telemetry_batch, self.rsu_registry, queue_depths, step)
            except Exception:
                pass

    def on_task_completed(self, task: TaskState, completion_step: int) -> None:
        """Closed-loop delayed feedback from simulated task completion."""
        offload_state = getattr(task, "decision_offload_state", None)
        if offload_state is None:
            return

        action_rsu = getattr(task, "decision_action_rsu", task.execution_rsu)
        actual_latency_s = getattr(task, "actual_latency_s", float(completion_step - task.created_step))
        deadline_met = getattr(task, "deadline_met", completion_step <= task.deadline_step)
        if deadline_met is None:
            deadline_met = bool(completion_step <= task.deadline_step)

        reward = 1.0 if deadline_met else -1.0
        reward -= 0.05 * min(10.0, actual_latency_s)

        feedback = DecisionFeedback(
            state=offload_state,
            action_rsu=action_rsu,
            latency_s=actual_latency_s,
            deadline_met=bool(deadline_met),
            reward=reward,
            step=completion_step,
            vehicle_id=task.vehicle_id,
            is_terminal=True,
        )
        self.selector.update(feedback)
        if self.dataset_logger is not None:
            self.dataset_logger.log(feedback)

    def _record_feedback(
        self,
        offload_state: OffloadingState,
        action_rsu: str,
        estimated_finish: int,
        deadline_step: int,
        step: int,
        vehicle_id: str,
    ) -> None:
        if self.delayed_feedback:
            # When delayed feedback is active, updates occur in on_task_completed upon actual finish.
            return
        total_latency_s = float(estimated_finish - step)
        deadline_met = bool(estimated_finish <= deadline_step)
        reward = 1.0 if deadline_met else -1.0
        feedback = DecisionFeedback(
            state=offload_state,
            action_rsu=action_rsu,
            latency_s=total_latency_s,
            deadline_met=deadline_met,
            reward=reward,
            step=step,
            vehicle_id=vehicle_id,
        )
        self.selector.update(feedback)
        if self.dataset_logger is not None:
            self.dataset_logger.log(feedback)

    def schedule(
        self,
        telemetry: Telemetry,
        queue_depths: dict[str, int],
    ) -> SchedulingDecision:
        # 1. Create or retrieve task state
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
            heading_deg=getattr(telemetry, "angle", 0.0),
            candidate_rsus=candidate_rsus,
        )

        action = "keep_local"
        reason = "same_region"
        tier = "V2I"
        contact_duration = float("inf")
        is_vrsu = False

        if self.algorithm_name != "greedy":
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
                    self.task_manager.migrate(state, telemetry.observed_rsu, telemetry.observed_rsu, tier="V2I")
                    action = "migrate"
                    reason = f"{self.selector.name}_migrate"
                else:
                    self.task_manager.set_execution_rsu(state, chosen_rsu, tier="FOG_MIGRATE")
                    action = "offload"
                    reason = f"{self.selector.name}_offload"

            target_rsu_id = chosen_rsu
            execution_rsu_obj = self.rsu_registry.get(target_rsu_id)
            if execution_rsu_obj is not None and getattr(execution_rsu_obj, "is_vehicular", False):
                tier = "vRSU"
                is_vrsu = True
                contact_duration = self.rsu_registry.calc_contact_duration(
                    telemetry.position, telemetry.speed, getattr(telemetry, "angle", 0.0), execution_rsu_obj
                )
            elif target_rsu_id == state.owner_rsu or target_rsu_id == telemetry.observed_rsu:
                tier = "V2I"
                is_vrsu = False
            else:
                tier = "FOG_MIGRATE"
                is_vrsu = False
        else:
            # Baseline greedy multi-tier decision logic
            # Tier 0: Check if vehicle itself is a volunteering vehicular RSU (Self-Hosted Edge)
            self_vrsu_id = f"vrsu_{telemetry.vehicle_id}"
            self_vrsu = self.rsu_registry.get(self_vrsu_id)
            if self_vrsu is not None:
                tier = "vRSU"
                action = "keep_local"
                reason = "vrsu_self_hosted"
                state.execution_tier = tier
                state.execution_rsu = self_vrsu_id
                queue_depths[self_vrsu_id] = queue_depths.get(self_vrsu_id, 0) + 1
                finish_step = self.task_manager.estimate_finish_step(
                    state,
                    queue_depth=queue_depths[self_vrsu_id],
                    now_step=telemetry.step,
                    network_latency_s=0.001,
                    node_gflops=self_vrsu.spec.gflops,
                )
                self._record_feedback(
                    offload_state,
                    self_vrsu_id,
                    finish_step,
                    state.deadline_step,
                    telemetry.step,
                    telemetry.vehicle_id,
                )
                return SchedulingDecision(
                    vehicle_id=telemetry.vehicle_id,
                    observed_rsu=telemetry.observed_rsu,
                    owner_rsu=state.owner_rsu,
                    execution_rsu=self_vrsu_id,
                    action=action,
                    reason=reason,
                    estimated_finish_step=finish_step,
                    deadline_step=state.deadline_step,
                    estimated_network_latency_s=0.001,
                    execution_tier=tier,
                    transmission_latency_s=0.001,
                    queue_latency_s=0.0,
                    compute_latency_s=state.remaining_gflops / max(100.0, self_vrsu.spec.gflops),
                    sinr_linear=1000.0,
                    data_rate_mbps=100.0,
                    contact_duration_s=float("inf"),
                    is_vrsu=True,
                )

            # Tier 1: Check V2V (Peer-to-Peer Car-to-Car) for small/safety tasks
            if state.task_category == "safety" or state.total_gflops <= 0.5:
                # Check if execution on local OBU (500 GFLOPS) finishes before deadline
                obu_finish = self.task_manager.estimate_finish_step(
                    state,
                    queue_depth=0,
                    now_step=telemetry.step,
                    network_latency_s=0.01,  # V2V propagation delay
                    node_gflops=472.0,  # Jetson Nano
                )
                if obu_finish <= state.deadline_step:
                    tier = "V2V"
                    action = "keep_local"
                    reason = "v2v_obu_capable"
                    state.execution_tier = tier
                    self._record_feedback(
                        offload_state,
                        "local_obu",
                        obu_finish,
                        state.deadline_step,
                        telemetry.step,
                        telemetry.vehicle_id,
                    )
                    return SchedulingDecision(
                        vehicle_id=telemetry.vehicle_id,
                        observed_rsu=telemetry.observed_rsu,
                        owner_rsu=state.owner_rsu,
                        execution_rsu="local_obu",
                        action=action,
                        reason=reason,
                        estimated_finish_step=obu_finish,
                        deadline_step=state.deadline_step,
                        estimated_network_latency_s=0.01,
                        execution_tier=tier,
                        transmission_latency_s=0.01,
                        queue_latency_s=0.0,
                        compute_latency_s=state.remaining_gflops / 472.0,
                        sinr_linear=100.0,
                        data_rate_mbps=50.0,
                        contact_duration_s=float("inf"),
                        is_vrsu=False,
                    )

            # Tier 4: Heavy global tasks direct to Cloud
            if state.task_category == "global":
                tier = "CLOUD"
                action = "offload"
                reason = "heavy_task_cloud"
                cloud_wan_s = calc_cloud_latency_s(state.payload_in_mbits, wan_rtt_s=0.05)
                cloud_finish = self.task_manager.estimate_finish_step(
                    state,
                    queue_depth=0,
                    now_step=telemetry.step,
                    network_latency_s=cloud_wan_s,
                    node_gflops=1000000.0,  # Cloud server cluster
                )
                state.execution_tier = tier
                self._record_feedback(
                    offload_state,
                    "cloud_node",
                    cloud_finish,
                    state.deadline_step,
                    telemetry.step,
                    telemetry.vehicle_id,
                )
                return SchedulingDecision(
                    vehicle_id=telemetry.vehicle_id,
                    observed_rsu=telemetry.observed_rsu,
                    owner_rsu=state.owner_rsu,
                    execution_rsu="cloud_node",
                    action=action,
                    reason=reason,
                    estimated_finish_step=cloud_finish,
                    deadline_step=state.deadline_step,
                    estimated_network_latency_s=cloud_wan_s,
                    execution_tier=tier,
                    transmission_latency_s=cloud_wan_s,
                    queue_latency_s=0.0,
                    compute_latency_s=state.remaining_gflops / 1000000.0,
                    sinr_linear=10.0,
                    data_rate_mbps=100.0,
                    contact_duration_s=float("inf"),
                    is_vrsu=False,
                )

            # Active task currently assigned to a Vehicular RSU (vRSU monitoring & handover)
            current_exec_rsu = self.rsu_registry.get(state.execution_rsu)
            if current_exec_rsu is not None and getattr(current_exec_rsu, "is_vehicular", False):
                is_vrsu = True
                cov_r = float(getattr(current_exec_rsu, "coverage_radius", 350.0))
                dist_curr = self.rsu_registry.distance_to_rsu(telemetry.position, current_exec_rsu.rsu_id)
                contact_duration = self.rsu_registry.calc_contact_duration(
                    telemetry.position, telemetry.speed, getattr(telemetry, "angle", 0.0), current_exec_rsu
                )
                q_depth = queue_depths.get(current_exec_rsu.rsu_id, 0)
                rem_service_s = (state.remaining_gflops + (q_depth * state.total_gflops)) / max(100.0, current_exec_rsu.spec.gflops)

                # Check for coverage exit or imminent disconnect
                if dist_curr > cov_r or contact_duration < rem_service_s:
                    # Handover needed: try alternative vRSU first, else handover to static macro RSU
                    alt_vrsu = self.rsu_registry.find_best_vrsu(
                        telemetry.position,
                        telemetry.speed,
                        getattr(telemetry, "angle", 0.0),
                        state.remaining_gflops,
                        state.payload_in_mbits,
                        queue_depths,
                        self.overload_threshold,
                    )
                    if alt_vrsu is not None and alt_vrsu[0].rsu_id != current_exec_rsu.rsu_id:
                        target_vrsu, alt_contact, _ = alt_vrsu
                        target_rsu_id = target_vrsu.rsu_id
                        action = "migrate"
                        reason = "vrsu_to_vrsu_handover"
                        tier = "vRSU"
                        contact_duration = alt_contact
                        self.task_manager.set_execution_rsu(state, target_rsu_id, tier=tier)
                    else:
                        target_rsu_id = telemetry.observed_rsu
                        action = "migrate"
                        reason = "vrsu_coverage_exit"
                        tier = "V2I"
                        is_vrsu = False
                        self.task_manager.migrate(state, telemetry.observed_rsu, target_rsu_id, tier=tier)
                elif self.rsu_registry.is_overloaded(current_exec_rsu.rsu_id, queue_depths, self.overload_threshold):
                    # vRSU overloaded: offload to static RSU
                    target_rsu_id = telemetry.observed_rsu
                    action = "offload"
                    reason = "vrsu_overloaded"
                    tier = "V2I"
                    is_vrsu = False
                    self.task_manager.migrate(state, telemetry.observed_rsu, target_rsu_id, tier=tier)
                else:
                    # Stable vRSU execution within coverage
                    target_rsu_id = current_exec_rsu.rsu_id
                    action = "keep_local"
                    reason = "vrsu_in_coverage"
                    tier = "vRSU"

                execution_rsu_obj = self.rsu_registry.get(target_rsu_id)
                queue_depths[target_rsu_id] = queue_depths.get(target_rsu_id, 0) + 1
                current_queue_len = queue_depths[target_rsu_id]
                exec_latency = self._network_latency_s(telemetry.position, execution_rsu_obj)
                estimated_finish = self.task_manager.estimate_finish_step(
                    state,
                    current_queue_len,
                    telemetry.step,
                    network_latency_s=exec_latency,
                    node_gflops=execution_rsu_obj.spec.gflops,
                )
                dist_m = self.rsu_registry.distance_to_rsu(telemetry.position, target_rsu_id)
                sinr = calc_sinr_linear(dist_m, protocol=execution_rsu_obj.protocol_spec)
                rate_bps = calc_shannon_rate_bps(dist_m, protocol=execution_rsu_obj.protocol_spec)
                trans_time_s = calc_transmission_time_s(dist_m, state.payload_in_mbits, protocol=execution_rsu_obj.protocol_spec)
                comp_time_s = state.remaining_gflops / max(100.0, execution_rsu_obj.spec.gflops)
                queue_time_s = (queue_depths.get(target_rsu_id, 0) * state.total_gflops) / max(100.0, execution_rsu_obj.spec.gflops)

                self._record_feedback(
                    offload_state,
                    target_rsu_id,
                    estimated_finish,
                    state.deadline_step,
                    telemetry.step,
                    telemetry.vehicle_id,
                )
                return SchedulingDecision(
                    vehicle_id=telemetry.vehicle_id,
                    observed_rsu=telemetry.observed_rsu,
                    owner_rsu=state.owner_rsu,
                    execution_rsu=target_rsu_id,
                    action=action,
                    reason=reason,
                    estimated_finish_step=estimated_finish,
                    deadline_step=state.deadline_step,
                    estimated_network_latency_s=exec_latency,
                    execution_tier=tier,
                    transmission_latency_s=trans_time_s,
                    queue_latency_s=queue_time_s,
                    compute_latency_s=comp_time_s,
                    sinr_linear=sinr,
                    data_rate_mbps=rate_bps / 1e6,
                    contact_duration_s=contact_duration,
                    is_vrsu=is_vrsu,
                )

            # Tier 2: V2I Primary RSU & Handover evaluation
            target_rsu_id = state.execution_rsu
            if telemetry.observed_rsu != state.owner_rsu:
                sticky_rsu = self.rsu_registry.get(state.execution_rsu)
                current_queue = queue_depths.get(state.execution_rsu, 0)
                sticky_latency = self._network_latency_s(telemetry.position, sticky_rsu)
                estimated_finish = self.task_manager.estimate_finish_step(
                    state,
                    current_queue,
                    telemetry.step,
                    network_latency_s=sticky_latency,
                    node_gflops=sticky_rsu.spec.gflops,
                )
                if estimated_finish <= state.deadline_step:
                    action = "sticky"
                    reason = "can_finish_before_deadline"
                else:
                    self.task_manager.migrate(state, telemetry.observed_rsu, telemetry.observed_rsu, tier="V2I")
                    action = "migrate"
                    reason = "deadline_risk"
                    target_rsu_id = telemetry.observed_rsu

            # Smart Opportunistic Assignment to nearby Vehicular RSU (vRSU)
            vrsu_candidate = self.rsu_registry.find_best_vrsu(
                telemetry.position,
                telemetry.speed,
                getattr(telemetry, "angle", 0.0),
                state.total_gflops,
                state.payload_in_mbits,
                queue_depths,
                self.overload_threshold,
            )

            is_target_overloaded = self.rsu_registry.is_overloaded(target_rsu_id, queue_depths, self.overload_threshold)

            if vrsu_candidate is not None:
                best_vrsu, vrsu_contact, vrsu_est_finish = vrsu_candidate
                dist_to_static = self.rsu_registry.distance_to_rsu(telemetry.position, target_rsu_id)
                dist_to_vrsu = self.rsu_registry.distance_to_rsu(telemetry.position, best_vrsu.rsu_id)

                # Assign to vRSU if:
                # 1. Target static RSU is overloaded (absorb load onto vehicular edge), OR
                # 2. vRSU is closer and has ample contact duration (> 2x finish time), saving V2I transmission latency
                if is_target_overloaded or (dist_to_vrsu < dist_to_static and vrsu_contact >= (vrsu_est_finish * 2.0)):
                    target_rsu_id = best_vrsu.rsu_id
                    tier = "vRSU"
                    is_vrsu = True
                    contact_duration = vrsu_contact
                    self.task_manager.set_execution_rsu(state, target_rsu_id, tier=tier)
                    if is_target_overloaded:
                        action = "offload"
                        reason = "overload_absorbed_by_vrsu"
                    else:
                        if action == "keep_local":
                            action = "offload"
                            reason = "smart_vrsu_assigned"
                        else:
                            reason = f"{reason}+vrsu_assigned"

            # Tier 3: Fog-to-Fog Inter-RSU Offloading if target RSU is STILL overloaded and no vRSU absorbed it
            execution_rsu_obj = self.rsu_registry.get(target_rsu_id)
            if self.rsu_registry.is_overloaded(target_rsu_id, queue_depths, self.overload_threshold):
                alt = self.rsu_registry.nearest_available_rsu(
                    telemetry.position,
                    queue_depths,
                    self.overload_threshold,
                    exclude={target_rsu_id},
                    include_vehicular=True,
                )
                if alt is not None:
                    tier = "vRSU" if getattr(alt, "is_vehicular", False) else "FOG_MIGRATE"
                    is_vrsu = getattr(alt, "is_vehicular", False)
                    self.task_manager.set_execution_rsu(state, alt.rsu_id, tier=tier)
                    execution_rsu_obj = alt
                    target_rsu_id = alt.rsu_id
                    if action == "keep_local":
                        action = "offload"
                        reason = "overloaded_exec_rsu"
                    else:
                        reason = f"{reason}+offloaded"
                else:
                    # All nearby RSUs overloaded -> Tier 4 Cloud fallback
                    tier = "CLOUD"
                    action = "offload"
                    reason = "rsu_overloaded_cloud_fallback"
                    cloud_wan_s = calc_cloud_latency_s(state.payload_in_mbits, wan_rtt_s=0.08)
                    cloud_finish = self.task_manager.estimate_finish_step(
                        state,
                        queue_depth=0,
                        now_step=telemetry.step,
                        network_latency_s=cloud_wan_s,
                        node_gflops=1000000.0,
                    )
                    state.execution_tier = tier
                    self._record_feedback(
                        offload_state,
                        "cloud_node",
                        cloud_finish,
                        state.deadline_step,
                        telemetry.step,
                        telemetry.vehicle_id,
                    )
                    return SchedulingDecision(
                        vehicle_id=telemetry.vehicle_id,
                        observed_rsu=telemetry.observed_rsu,
                        owner_rsu=state.owner_rsu,
                        execution_rsu="cloud_node",
                        action=action,
                        reason=reason,
                        estimated_finish_step=cloud_finish,
                        deadline_step=state.deadline_step,
                        estimated_network_latency_s=cloud_wan_s,
                        execution_tier=tier,
                        transmission_latency_s=cloud_wan_s,
                        queue_latency_s=0.0,
                        compute_latency_s=state.remaining_gflops / 1000000.0,
                        sinr_linear=10.0,
                        data_rate_mbps=100.0,
                        contact_duration_s=float("inf"),
                        is_vrsu=False,
                    )

        execution_rsu_obj = self.rsu_registry.get(target_rsu_id)
        queue_depths[target_rsu_id] = queue_depths.get(target_rsu_id, 0) + 1
        current_queue_len = queue_depths[target_rsu_id]
        exec_latency = self._network_latency_s(telemetry.position, execution_rsu_obj)
        estimated_finish = self.task_manager.estimate_finish_step(
            state,
            current_queue_len,
            telemetry.step,
            network_latency_s=exec_latency,
            node_gflops=execution_rsu_obj.spec.gflops,
        )

        # Calculate wireless physical layer stats for decision reporting
        dist_m = self.rsu_registry.distance_to_rsu(telemetry.position, target_rsu_id)
        sinr = calc_sinr_linear(dist_m, protocol=execution_rsu_obj.protocol_spec)
        rate_bps = calc_shannon_rate_bps(dist_m, protocol=execution_rsu_obj.protocol_spec)
        trans_time_s = calc_transmission_time_s(dist_m, state.payload_in_mbits, protocol=execution_rsu_obj.protocol_spec)
        if tier == "FOG_MIGRATE":
            trans_time_s += calc_inter_rsu_latency_s(state.payload_in_mbits, hops=1)

        comp_time_s = state.remaining_gflops / max(100.0, execution_rsu_obj.spec.gflops)
        queue_time_s = (queue_depths.get(target_rsu_id, 0) * state.total_gflops) / max(100.0, execution_rsu_obj.spec.gflops)

        state.decision_offload_state = offload_state
        state.decision_action_rsu = target_rsu_id
        state.decision_step = telemetry.step

        self._record_feedback(
            offload_state,
            target_rsu_id,
            estimated_finish,
            state.deadline_step,
            telemetry.step,
            telemetry.vehicle_id,
        )

        return SchedulingDecision(
            vehicle_id=telemetry.vehicle_id,
            observed_rsu=telemetry.observed_rsu,
            owner_rsu=state.owner_rsu,
            execution_rsu=target_rsu_id,
            action=action,
            reason=reason,
            estimated_finish_step=estimated_finish,
            deadline_step=state.deadline_step,
            estimated_network_latency_s=exec_latency,
            execution_tier=tier,
            transmission_latency_s=trans_time_s,
            queue_latency_s=queue_time_s,
            compute_latency_s=comp_time_s,
            sinr_linear=sinr,
            data_rate_mbps=rate_bps / 1e6,
            contact_duration_s=contact_duration,
            is_vrsu=is_vrsu,
        )

    def set_measured_link_latency(self, measured_latency_s: float) -> None:
        measured = max(0.0, float(measured_latency_s))
        alpha = self.adaptive_latency_alpha
        self._measured_link_latency_s = ((1.0 - alpha) * self._measured_link_latency_s) + (alpha * measured)

    def _network_latency_s(self, position: tuple[float, float], execution_rsu) -> float:
        rsu_id = execution_rsu.rsu_id if hasattr(execution_rsu, "rsu_id") else str(execution_rsu)
        baseline = self._distance_latency_s(position, rsu_id)
        if not self.adaptive_latency_eta:
            return baseline
        return baseline + (self._measured_link_latency_s * self.adaptive_latency_weight)

    def _distance_latency_s(self, position: tuple[float, float], execution_rsu_id: str) -> float:
        if execution_rsu_id not in self.rsu_registry._by_id:
            return 0.05  # Fallback for Cloud or non-RSU nodes

        rsu = self.rsu_registry.get(execution_rsu_id)
        distance_m = self.rsu_registry.distance_to_rsu(position, execution_rsu_id)

        if self.distance_latency_max_s > 0.0:
            normalized = distance_m / self.distance_latency_norm_m
            dist_latency = self.distance_latency_base_s + (self.distance_latency_max_s - self.distance_latency_base_s) * normalized
            return dist_latency + (self.distance_error * dist_latency)

        # Physical layer Shannon capacity transmission time calculation
        return calc_transmission_time_s(distance_m, payload_mbits=1.0, protocol=rsu.protocol_spec)
