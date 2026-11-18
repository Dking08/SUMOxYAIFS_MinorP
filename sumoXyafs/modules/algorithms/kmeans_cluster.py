"""Algorithm 8: K-Means Clustering for vehicular service grouping and RSU pre-placement."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional
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

    def _pure_numpy_kmeans(self, features: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Vectorized pure-NumPy K-Means implementation."""
        np.random.seed(42)
        n_samples = len(features)
        init_idx = np.random.choice(n_samples, size=k, replace=False)
        centroids = features[init_idx].copy()
        labels = np.zeros(n_samples, dtype=int)

        for _ in range(self.max_iter):
            # Compute distances in spatial dimensions (x, y)
            dists = np.linalg.norm(features[:, :2, None] - centroids[:, :2].T[None, :, :], axis=1)
            labels = np.argmin(dists, axis=1)
            new_centroids = np.zeros_like(centroids)
            for ci in range(k):
                mask = labels == ci
                new_centroids[ci] = np.mean(features[mask], axis=0) if np.any(mask) else centroids[ci]
            centroids = new_centroids

        return labels, centroids

    def prepare_step(
        self,
        telemetry_batch: list[Any],
        rsu_registry: Any,
        queue_depths: dict[str, int],
        step: int,
    ) -> None:
        """Batch ingestion and clustering of all active vehicles for the current step."""
        if not telemetry_batch:
            return

        self._last_step = step
        self._assignments.clear()
        self._vehicle_positions = {
            t.vehicle_id: (t.position[0], t.position[1], t.speed, getattr(t, "angle", 0.0))
            for t in telemetry_batch
        }

        if rsu_registry is not None:
            candidates = [r.rsu_id for r in rsu_registry.all_rsus()]
            rsu_capacities = {r.rsu_id: r.capacity for r in rsu_registry.all_rsus()}
            rsu_coords = {r.rsu_id: (r.x, r.y) for r in rsu_registry.all_rsus()}
        else:
            candidates = []
            rsu_capacities = {}
            rsu_coords = {}

        if not candidates:
            return

        vids = list(self._vehicle_positions.keys())
        features = np.array([self._vehicle_positions[vid] for vid in vids], dtype=float)
        scale = np.array([5000.0, 5000.0, 30.0, 360.0])
        norm_feats = features / scale

        k = min(self.k_clusters, len(vids))
        if k <= 1:
            med_x = float(np.median(features[:, 0]))
            med_y = float(np.median(features[:, 1]))
            best_rsu = min(
                candidates,
                key=lambda r: (med_x - rsu_coords.get(r, (0, 0))[0]) ** 2
                + (med_y - rsu_coords.get(r, (0, 0))[1]) ** 2
                + (2000.0 if queue_depths.get(r, 0) >= rsu_capacities.get(r, 20) else 0.0),
            )
            for vid in vids:
                self._assignments[vid] = best_rsu
            return

        if HAS_SKLEARN:
            try:
                kmeans = KMeans(n_clusters=k, max_iter=self.max_iter, random_state=42, n_init="auto")
                labels = kmeans.fit_predict(norm_feats)
                centroids = kmeans.cluster_centers_ * scale
            except Exception:
                labels, centroids = self._pure_numpy_kmeans(features, k)
        else:
            labels, centroids = self._pure_numpy_kmeans(features, k)

        cluster_rsu_map: dict[int, str] = {}
        for ci in range(k):
            cx, cy = centroids[ci, 0], centroids[ci, 1]
            def rsu_dist(r: str) -> float:
                rx, ry = rsu_coords.get(r, (0.0, 0.0))
                d = float(np.hypot(cx - rx, cy - ry))
                q = queue_depths.get(r, 0)
                cap = max(1, rsu_capacities.get(r, 20))
                penalty = 2000.0 if q >= cap else 0.0
                return d + penalty

            cluster_rsu_map[ci] = min(candidates, key=rsu_dist)

        for vid, label in zip(vids, labels):
            self._assignments[vid] = cluster_rsu_map.get(int(label), candidates[0])

    def _cluster_and_assign(self, state: OffloadingState) -> None:
        candidates = state.candidate_rsus or list(state.rsu_distances.keys())
        if not candidates or not self._vehicle_positions:
            return

        vids = list(self._vehicle_positions.keys())
        features = np.array([self._vehicle_positions[vid] for vid in vids], dtype=float)
        scale = np.array([5000.0, 5000.0, 30.0, 360.0])
        norm_feats = features / scale

        k = min(self.k_clusters, len(vids))
        if k <= 1:
            rsu = min(candidates, key=lambda r: state.rsu_distances.get(r, 1000.0))
            for vid in vids:
                self._assignments[vid] = rsu
            return

        if HAS_SKLEARN:
            try:
                kmeans = KMeans(n_clusters=k, max_iter=self.max_iter, random_state=42, n_init="auto")
                labels = kmeans.fit_predict(norm_feats)
                centroids = kmeans.cluster_centers_ * scale
            except Exception:
                labels, centroids = self._pure_numpy_kmeans(features, k)
        else:
            labels, centroids = self._pure_numpy_kmeans(features, k)

        cluster_rsu_map: dict[int, str] = {}
        for ci in range(k):
            cx, cy = centroids[ci, 0], centroids[ci, 1]
            def rsu_dist(r: str) -> float:
                d = state.rsu_distances.get(r, 1000.0)
                q = state.queue_depths.get(r, 0)
                cap = max(1, state.rsu_capacities.get(r, 20))
                penalty = 2000.0 if q >= cap else 0.0
                return d + penalty

            cluster_rsu_map[ci] = min(candidates, key=rsu_dist)

        for vid, label in zip(vids, labels):
            self._assignments[vid] = cluster_rsu_map.get(int(label), state.observed_rsu)

    def choose_rsu(self, state: OffloadingState) -> str:
        candidates = state.candidate_rsus or list(state.rsu_distances.keys())
        if not candidates:
            return state.observed_rsu

        pos_x, pos_y = state.position
        self._vehicle_positions[state.vehicle_id] = (pos_x, pos_y, state.speed, state.heading_deg)

        # If prepare_step wasn't called beforehand, fall back to state-triggered clustering
        if state.step != self._last_step:
            self._last_step = state.step
            self._assignments.clear()
            self._cluster_and_assign(state)

        assigned = self._assignments.get(state.vehicle_id)
        if assigned and assigned in candidates:
            q = state.queue_depths.get(assigned, 0)
            cap = state.rsu_capacities.get(assigned, 20)
            if q < cap:
                return assigned

        return min(candidates, key=lambda r: state.rsu_distances.get(r, 1000.0))
