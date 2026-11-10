"""Algorithm 8: K-Means Clustering for vehicular service grouping and RSU pre-placement."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from modules.algorithms.base import RSUSelector
from modules.models import OffloadingState

try:
    from sklearn.cluster import KMeans  # type: ignore
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False


class KMeansClusterSelector(RSUSelector):
    """Unsupervised K-Means clustering algorithm.

    Groups active vehicles by position (x, y), speed, and heading;
    assigns / pre-places services per cluster to the optimal nearby RSU with capacity.
    """

    name: str = "kmeans"

    def __init__(self, k_clusters: int = 5, max_iter: int = 10, **kwargs) -> None:
        super().__init__(**kwargs)
        self.k_clusters = max(1, int(k_clusters))
        self.max_iter = max(1, int(max_iter))
        # Active vehicle registry for the current step: vid -> (x, y, speed, heading)
        self._vehicle_positions: dict[str, tuple[float, float, float, float]] = {}
        self._last_step = -1
        # Cluster assignments: vid -> assigned_rsu_id
        self._assignments: dict[str, str] = {}

    def _cluster_and_assign(self, state: OffloadingState) -> None:
        candidates = state.candidate_rsus or list(state.rsu_distances.keys())
        if not candidates or not self._vehicle_positions:
            return

        vids = list(self._vehicle_positions.keys())
        features = np.array([self._vehicle_positions[vid] for vid in vids], dtype=float)
        # Normalize features: x/5000, y/5000, speed/30, heading/360
        scale = np.array([5000.0, 5000.0, 30.0, 360.0])
        norm_feats = features / scale

        k = min(self.k_clusters, len(vids))
        if k <= 1:
            # Single cluster: assign all to nearest non-overloaded RSU to median vehicle
            med_x = float(np.median(features[:, 0]))
            med_y = float(np.median(features[:, 1]))
            rsu = min(candidates, key=lambda r: state.rsu_distances.get(r, 1000.0))
            for vid in vids:
                self._assignments[vid] = rsu
            return

        # Perform K-Means clustering
        if HAS_SKLEARN:
            kmeans = KMeans(n_clusters=k, max_iter=self.max_iter, random_state=42, n_init="auto")
            labels = kmeans.fit_predict(norm_feats)
            centroids = kmeans.cluster_centers_ * scale
        else:
            # Pure-NumPy K-Means implementation
            np.random.seed(42)
            init_idx = np.random.choice(len(vids), size=k, replace=False)
            centroids = features[init_idx].copy()
            labels = np.zeros(len(vids), dtype=int)

            for _ in range(self.max_iter):
                # Assign to nearest centroid in spatial distance
                dists = np.linalg.norm(features[:, :2, None] - centroids[:, :2].T[None, :, :], axis=1)
                labels = np.argmin(dists, axis=1)
                # Recompute centroids
                new_centroids = np.zeros_like(centroids)
                for ci in range(k):
                    mask = labels == ci
                    new_centroids[ci] = np.mean(features[mask], axis=0) if np.any(mask) else centroids[ci]
                centroids = new_centroids

        # For each cluster centroid, assign the optimal nearby RSU with available capacity
        cluster_rsu_map: dict[int, str] = {}
        for ci in range(k):
            cx, cy = centroids[ci, 0], centroids[ci, 1]
            # Best candidate RSU closest to cluster centroid
            def rsu_dist(r: str) -> float:
                d = state.rsu_distances.get(r, 1000.0)
                q = state.queue_depths.get(r, 0)
                cap = max(1, state.rsu_capacities.get(r, 20))
                # Penalize overloaded RSUs
                penalty = 2000.0 if q >= cap else 0.0
                return d + penalty

            cluster_rsu_map[ci] = min(candidates, key=rsu_dist)

        for vid, label in zip(vids, labels):
            self._assignments[vid] = cluster_rsu_map.get(int(label), state.observed_rsu)

    def choose_rsu(self, state: OffloadingState) -> str:
        candidates = state.candidate_rsus or list(state.rsu_distances.keys())
        if not candidates:
            return state.observed_rsu

        # Register vehicle position and motion for clustering
        pos_x, pos_y = state.position
        self._vehicle_positions[state.vehicle_id] = (pos_x, pos_y, state.speed, state.heading_deg)

        # Trigger clustering when a new step begins
        if state.step != self._last_step:
            self._last_step = state.step
            self._assignments.clear()
            self._cluster_and_assign(state)

        # Retrieve cluster assignment
        assigned = self._assignments.get(state.vehicle_id)
        if assigned and assigned in candidates:
            q = state.queue_depths.get(assigned, 0)
            cap = state.rsu_capacities.get(assigned, 20)
            # If assigned RSU is not saturated, use it
            if q < cap:
                return assigned

        # Fallback to nearest available candidate RSU
        return min(candidates, key=lambda r: state.rsu_distances.get(r, 1000.0))
