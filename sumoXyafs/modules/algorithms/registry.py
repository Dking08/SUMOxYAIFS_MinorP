"""Algorithm factory and catalog for RSU offloading algorithms."""
from __future__ import annotations

from typing import Type

from modules.algorithms.base import RSUSelector
from modules.algorithms.decision_tree import DecisionTreeSelector
from modules.algorithms.greedy import GreedyBaselineSelector
from modules.algorithms.kmeans_cluster import KMeansClusterSelector
from modules.algorithms.linucb import LinUCBSelector
from modules.algorithms.markov_chain import MarkovChainSelector
from modules.algorithms.random_forest import RandomForestSelector
from modules.algorithms.ridge_regression import RidgeRegressionSelector
from modules.algorithms.tabular_q_learning import TabularQLearningSelector
from modules.algorithms.thompson_sampling import ThompsonSamplingSelector

ALGORITHM_CATALOG: dict[str, Type[RSUSelector]] = {
    "greedy": GreedyBaselineSelector,
    "linucb": LinUCBSelector,
    "thompson": ThompsonSamplingSelector,
    "ridge": RidgeRegressionSelector,
    "random_forest": RandomForestSelector,
    "markov": MarkovChainSelector,
    "decision_tree": DecisionTreeSelector,
    "q_learning": TabularQLearningSelector,
    "kmeans": KMeansClusterSelector,
}


def get_algorithm(name: str | RSUSelector | None, **kwargs) -> RSUSelector:
    """Instantiate an RSU selection algorithm by name or return existing instance.

    Supported algorithm names:
      - 'greedy': Greedy baseline (sticky handover + nearest non-overloaded)
      - 'linucb': Online contextual bandit with confidence bonus
      - 'thompson': Online Thompson sampling bandit with Beta posterior
      - 'ridge': Supervised Ridge regression for latency prediction
      - 'random_forest': Supervised Random forest predicting deadline-miss probability
      - 'markov': Statistical Markov chain predicting next RSU from traces
      - 'decision_tree': Supervised Decision tree predicting Migrate vs. Stay
      - 'q_learning': Tabular Q-learning with coarse discretization
      - 'kmeans': Unsupervised K-Means clustering for vehicular grouping
    """
    if name is None:
        return GreedyBaselineSelector(**kwargs)

    if isinstance(name, RSUSelector):
        return name

    key = str(name).strip().lower()
    cls = ALGORITHM_CATALOG.get(key)
    if cls is None:
        valid = ", ".join(sorted(ALGORITHM_CATALOG.keys()))
        raise ValueError(f"Unknown algorithm '{name}'. Available algorithms: {valid}")

    return cls(**kwargs)
