from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.algorithms import (
    DatasetLogger,
    DecisionTreeSelector,
    GreedyBaselineSelector,
    KMeansClusterSelector,
    LinUCBSelector,
    MarkovChainSelector,
    RandomForestSelector,
    RidgeRegressionSelector,
    TabularQLearningSelector,
    ThompsonSamplingSelector,
    get_algorithm,
    load_dataset_from_run_dir,
    train_offline_models,
)
from modules.models import DecisionFeedback, OffloadingState, RSUNode, Telemetry
from modules.rsu_registry import RSURegistry
from modules.scheduler import Scheduler
from modules.task_manager import TaskManager


def make_dummy_state(
    vehicle_id: str = "v1",
    pos: tuple[float, float] = (100.0, 100.0),
    speed: float = 12.0,
    step: int = 10,
    observed: str = "rsu_1",
    owner: str = "rsu_1",
    execution: str = "rsu_1",
    queue_depths: dict[str, int] | None = None,
    deadline_slack: int = 2,
) -> OffloadingState:
    q = queue_depths or {"rsu_1": 1, "rsu_2": 2}
    return OffloadingState(
        vehicle_id=vehicle_id,
        position=pos,
        speed=speed,
        step=step,
        observed_rsu=observed,
        owner_rsu=owner,
        execution_rsu=execution,
        queue_depths=q,
        rsu_distances={"rsu_1": 150.0, "rsu_2": 800.0},
        rsu_capacities={"rsu_1": 10, "rsu_2": 10},
        task_id="task_1",
        remaining_compute_s=2.0,
        deadline_step=step + 4,
        deadline_slack_steps=deadline_slack,
        heading_deg=45.0,
        candidate_rsus=["rsu_1", "rsu_2"],
    )


class AlgorithmTests(unittest.TestCase):
    def test_greedy_selector(self) -> None:
        selector = GreedyBaselineSelector()
        state = make_dummy_state()
        self.assertEqual(selector.choose_rsu(state), "rsu_1")

        # Overloaded rsu_1 forces offload to rsu_2
        state_overloaded = make_dummy_state(queue_depths={"rsu_1": 20, "rsu_2": 2})
        self.assertEqual(selector.choose_rsu(state_overloaded), "rsu_2")

    def test_linucb_selector(self) -> None:
        selector = LinUCBSelector(alpha=0.5)
        state = make_dummy_state()
        chosen = selector.choose_rsu(state)
        self.assertIn(chosen, ["rsu_1", "rsu_2"])

        # Update with positive reward
        fb = DecisionFeedback(
            state=state,
            action_rsu="rsu_1",
            latency_s=2.0,
            deadline_met=True,
            reward=1.0,
            step=state.step,
            vehicle_id=state.vehicle_id,
        )
        selector.update(fb)
        self.assertIn("rsu_1", selector.A)

    def test_thompson_sampling_selector(self) -> None:
        selector = ThompsonSamplingSelector(prior_alpha=2.0, prior_beta=1.0)
        state = make_dummy_state()
        chosen = selector.choose_rsu(state)
        self.assertIn(chosen, ["rsu_1", "rsu_2"])

        fb = DecisionFeedback(
            state=state,
            action_rsu=chosen,
            latency_s=2.0,
            deadline_met=True,
            reward=1.0,
            step=state.step,
            vehicle_id=state.vehicle_id,
        )
        selector.update(fb)
        self.assertGreater(selector.alpha[chosen], 2.0)

    def test_ridge_regression_selector(self) -> None:
        selector = RidgeRegressionSelector()
        state = make_dummy_state()
        chosen = selector.choose_rsu(state)
        self.assertIn(chosen, ["rsu_1", "rsu_2"])

        # Offline training with synthetic records
        records = [
            {"norm_distance_to_action": 0.1, "queue_depth_action": 1, "speed": 10, "latency_s": 2.1},
            {"norm_distance_to_action": 0.8, "queue_depth_action": 15, "speed": 20, "latency_s": 6.5},
            {"norm_distance_to_action": 0.2, "queue_depth_action": 2, "speed": 5, "latency_s": 2.3},
            {"norm_distance_to_action": 0.9, "queue_depth_action": 18, "speed": 25, "latency_s": 7.8},
            {"norm_distance_to_action": 0.3, "queue_depth_action": 3, "speed": 12, "latency_s": 2.7},
            {"norm_distance_to_action": 0.7, "queue_depth_action": 12, "speed": 18, "latency_s": 5.8},
        ]
        selector.train_offline(records)
        # Higher distance and queue should lead to higher latency
        pred_low = selector._extract_pair_features(state, "rsu_1")
        pred_high = selector._extract_pair_features(state, "rsu_2")
        self.assertLess(selector.weights @ pred_low, selector.weights @ pred_high)

    def test_random_forest_selector(self) -> None:
        selector = RandomForestSelector(miss_threshold=0.5)
        state = make_dummy_state()
        chosen = selector.choose_rsu(state)
        self.assertIn(chosen, ["rsu_1", "rsu_2"])

        records = [
            {"norm_distance_to_action": 0.1, "queue_depth_action": 1, "deadline_met": 1},
            {"norm_distance_to_action": 0.9, "queue_depth_action": 19, "deadline_met": 0},
            {"norm_distance_to_action": 0.2, "queue_depth_action": 2, "deadline_met": 1},
            {"norm_distance_to_action": 0.8, "queue_depth_action": 18, "deadline_met": 0},
            {"norm_distance_to_action": 0.3, "queue_depth_action": 3, "deadline_met": 1},
            {"norm_distance_to_action": 0.85, "queue_depth_action": 16, "deadline_met": 0},
            {"norm_distance_to_action": 0.15, "queue_depth_action": 1, "deadline_met": 1},
            {"norm_distance_to_action": 0.75, "queue_depth_action": 15, "deadline_met": 0},
            {"norm_distance_to_action": 0.25, "queue_depth_action": 2, "deadline_met": 1},
            {"norm_distance_to_action": 0.95, "queue_depth_action": 19, "deadline_met": 0},
        ]
        selector.train_offline(records)
        self.assertIn(selector.choose_rsu(state), ["rsu_1", "rsu_2"])

    def test_markov_chain_selector(self) -> None:
        selector = MarkovChainSelector(proactive_distance_m=500.0)
        # Train with sequential transitions from rsu_1 -> rsu_2
        records = [
            {"vehicle_id": "v1", "step": 0, "observed_rsu": "rsu_1"},
            {"vehicle_id": "v1", "step": 1, "observed_rsu": "rsu_1"},
            {"vehicle_id": "v1", "step": 2, "observed_rsu": "rsu_2"},
            {"vehicle_id": "v2", "step": 0, "observed_rsu": "rsu_1"},
            {"vehicle_id": "v2", "step": 1, "observed_rsu": "rsu_2"},
        ]
        selector.train_offline(records)
        next_rsu, prob = selector.predict_next_rsu("rsu_1", ["rsu_1", "rsu_2"])
        self.assertEqual(next_rsu, "rsu_2")
        self.assertGreater(prob, 0.5)

    def test_decision_tree_selector(self) -> None:
        selector = DecisionTreeSelector()
        state = make_dummy_state()
        self.assertFalse(selector.should_migrate(state))
        self.assertEqual(selector.choose_rsu(state), "rsu_1")

        # Heavy queue and small slack triggers migration
        state_danger = make_dummy_state(queue_depths={"rsu_1": 18, "rsu_2": 1}, deadline_slack=-2)
        self.assertTrue(selector.should_migrate(state_danger))
        self.assertEqual(selector.choose_rsu(state_danger), "rsu_2")

    def test_tabular_q_learning_selector(self) -> None:
        selector = TabularQLearningSelector(alpha=0.3, epsilon=0.0)
        state = make_dummy_state()
        chosen = selector.choose_rsu(state)
        self.assertIn(chosen, ["rsu_1", "rsu_2"])

        fb = DecisionFeedback(
            state=state,
            action_rsu=chosen,
            latency_s=2.0,
            deadline_met=True,
            reward=1.0,
            step=state.step,
            vehicle_id=state.vehicle_id,
        )
        selector.update(fb)
        self.assertEqual(selector.Q.shape, (27, 4))

    def test_kmeans_cluster_selector(self) -> None:
        selector = KMeansClusterSelector(k_clusters=2)
        s1 = make_dummy_state(vehicle_id="v1", pos=(100.0, 100.0), step=1)
        s2 = make_dummy_state(vehicle_id="v2", pos=(900.0, 900.0), step=1)
        c1 = selector.choose_rsu(s1)
        c2 = selector.choose_rsu(s2)
        self.assertIn(c1, ["rsu_1", "rsu_2"])
        self.assertIn(c2, ["rsu_1", "rsu_2"])

    def test_dataset_logger(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_path = Path(tmp_dir) / "test_dataset.csv"
            logger = DatasetLogger(log_path)
            state = make_dummy_state()
            fb = DecisionFeedback(
                state=state,
                action_rsu="rsu_1",
                latency_s=2.0,
                deadline_met=True,
                reward=1.0,
                step=state.step,
                vehicle_id=state.vehicle_id,
            )
            logger.log(fb)
            logger.flush()
            self.assertTrue(log_path.exists())
            content = log_path.read_text(encoding="utf-8")
            self.assertIn("v1", content)
            self.assertIn("rsu_1", content)

    def test_offline_training_on_baseline_run(self) -> None:
        baseline_dir = Path(__file__).resolve().parents[1] / "results" / "exp_baseline_03"
        if not baseline_dir.exists():
            self.skipTest("Baseline run directory not found")

        records = load_dataset_from_run_dir(baseline_dir)
        self.assertGreater(len(records), 100)

        with tempfile.TemporaryDirectory() as tmp_dir:
            models = train_offline_models(records, output_dir=tmp_dir)
            self.assertIn("ridge", models)
            self.assertIn("random_forest", models)
            self.assertIn("markov", models)
            self.assertIn("decision_tree", models)
            self.assertTrue((Path(tmp_dir) / "ridge_weights.json").exists())
            self.assertTrue((Path(tmp_dir) / "decision_tree_rules.json").exists())

    def test_scheduler_with_all_algorithms(self) -> None:
        registry = RSURegistry(
            rsus=[
                RSUNode("rsu_1", node_id=1, x=0.0, y=0.0, capacity=10),
                RSUNode("rsu_2", node_id=2, x=500.0, y=0.0, capacity=10),
            ]
        )
        algorithms = [
            "greedy",
            "linucb",
            "thompson",
            "ridge",
            "random_forest",
            "markov",
            "decision_tree",
            "q_learning",
            "kmeans",
        ]

        t0 = Telemetry(vehicle_id="veh_test", position=(50.0, 0.0), speed=10.0, step=0, observed_rsu="rsu_1")

        for algo in algorithms:
            tm = TaskManager(task_deadline_s=3.0, compute_time_s=2.0)
            scheduler = Scheduler(registry, tm, overload_threshold=10, algorithm=algo)
            decision = scheduler.schedule(t0, queue_depths={"rsu_1": 1, "rsu_2": 0})
            self.assertIn(decision.execution_rsu, ["rsu_1", "rsu_2"])
            self.assertIsNotNone(decision.action)

    def test_decision_tree_json_serialization_roundtrip(self) -> None:
        selector = DecisionTreeSelector(max_depth=3)
        records = [
            {"norm_distance_to_action": 0.1, "queue_depth_action": 1, "speed": 10, "deadline_slack_steps": 4, "deadline_met": 1},
            {"norm_distance_to_action": 0.9, "queue_depth_action": 18, "speed": 25, "deadline_slack_steps": -2, "deadline_met": 0},
            {"norm_distance_to_action": 0.2, "queue_depth_action": 2, "speed": 8, "deadline_slack_steps": 3, "deadline_met": 1},
            {"norm_distance_to_action": 0.8, "queue_depth_action": 15, "speed": 22, "deadline_slack_steps": -1, "deadline_met": 0},
            {"norm_distance_to_action": 0.15, "queue_depth_action": 2, "speed": 12, "deadline_slack_steps": 2, "deadline_met": 1},
            {"norm_distance_to_action": 0.85, "queue_depth_action": 19, "speed": 28, "deadline_slack_steps": -3, "deadline_met": 0},
        ]
        selector.train_offline(records)
        state_safe = make_dummy_state(queue_depths={"rsu_1": 1, "rsu_2": 2}, deadline_slack=4)
        state_danger = make_dummy_state(queue_depths={"rsu_1": 19, "rsu_2": 1}, deadline_slack=-3)

        mig_safe = selector.should_migrate(state_safe)
        mig_danger = selector.should_migrate(state_danger)

        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = Path(tmp_dir) / "dt_model.json"
            selector.save(model_path)
            self.assertTrue(model_path.exists())

            loaded = DecisionTreeSelector()
            loaded.load(model_path)
            self.assertEqual(loaded.should_migrate(state_safe), mig_safe)
            self.assertEqual(loaded.should_migrate(state_danger), mig_danger)

    def test_random_forest_json_serialization_roundtrip(self) -> None:
        selector = RandomForestSelector(miss_threshold=0.5, n_estimators=10)
        records = [
            {"norm_distance_to_action": 0.1, "queue_depth_action": 1, "speed": 10, "deadline_slack_steps": 4, "deadline_met": 1},
            {"norm_distance_to_action": 0.9, "queue_depth_action": 18, "speed": 25, "deadline_slack_steps": -2, "deadline_met": 0},
            {"norm_distance_to_action": 0.2, "queue_depth_action": 2, "speed": 8, "deadline_slack_steps": 3, "deadline_met": 1},
            {"norm_distance_to_action": 0.8, "queue_depth_action": 15, "speed": 22, "deadline_slack_steps": -1, "deadline_met": 0},
            {"norm_distance_to_action": 0.15, "queue_depth_action": 2, "speed": 12, "deadline_slack_steps": 2, "deadline_met": 1},
            {"norm_distance_to_action": 0.85, "queue_depth_action": 19, "speed": 28, "deadline_slack_steps": -3, "deadline_met": 0},
        ]
        selector.train_offline(records)
        state = make_dummy_state()
        feats = selector._extract_features(state, "rsu_1")
        prob_orig = selector.predict_miss_prob(feats)

        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = Path(tmp_dir) / "rf_model.json"
            selector.save(model_path)
            self.assertTrue(model_path.exists())

            loaded = RandomForestSelector()
            loaded.load(model_path)
            prob_loaded = loaded.predict_miss_prob(feats)
            self.assertAlmostEqual(prob_orig, prob_loaded, places=4)

    def test_kmeans_batch_prepare_step(self) -> None:
        selector = KMeansClusterSelector(k_clusters=2)
        registry = RSURegistry(
            rsus=[
                RSUNode("rsu_1", node_id=1, x=100.0, y=100.0, capacity=10),
                RSUNode("rsu_2", node_id=2, x=900.0, y=900.0, capacity=10),
            ]
        )
        t1 = Telemetry(vehicle_id="v1", position=(110.0, 105.0), speed=10.0, step=5, observed_rsu="rsu_1")
        t2 = Telemetry(vehicle_id="v2", position=(910.0, 895.0), speed=12.0, step=5, observed_rsu="rsu_2")

        selector.prepare_step([t1, t2], registry, {"rsu_1": 0, "rsu_2": 0}, step=5)
        s1 = make_dummy_state(vehicle_id="v1", pos=(110.0, 105.0), step=5)
        s2 = make_dummy_state(vehicle_id="v2", pos=(910.0, 895.0), step=5)
        self.assertEqual(selector.choose_rsu(s1), "rsu_1")
        self.assertEqual(selector.choose_rsu(s2), "rsu_2")

    def test_closed_loop_delayed_feedback(self) -> None:
        registry = RSURegistry(
            rsus=[
                RSUNode("rsu_1", node_id=1, x=0.0, y=0.0, capacity=10),
                RSUNode("rsu_2", node_id=2, x=500.0, y=0.0, capacity=10),
            ]
        )
        tm = TaskManager(task_deadline_s=3.0, compute_time_s=1.0)
        scheduler = Scheduler(registry, tm, algorithm="linucb", delayed_feedback=True)

        t0 = Telemetry(vehicle_id="v_dyn", position=(10.0, 0.0), speed=10.0, step=0, observed_rsu="rsu_1")
        dec = scheduler.schedule(t0, queue_depths={"rsu_1": 0, "rsu_2": 0})
        self.assertEqual(dec.execution_rsu, "rsu_1")

        completed = tm.tick_all(rsu_registry=registry, step_size_s=1.0, current_step=0)
        self.assertEqual(len(completed), 1)
        task = completed[0]
        self.assertEqual(task.completed_step, 0)
        self.assertTrue(task.deadline_met)

        scheduler.on_task_completed(task, completion_step=0)
        self.assertIn("rsu_1", scheduler.selector.A)


if __name__ == "__main__":
    unittest.main()
