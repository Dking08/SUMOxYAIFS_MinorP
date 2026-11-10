# CoFog-SUMO: SUMO and YAIFS Integration

This project explores microscopic vehicular mobility and edge/fog computing orchestration by integrating SUMO (Simulation of Urban MObility) with the YAIFS (Yet (not) Another Intelligent Fog Simulator) framework.

[YAIFS GitHub Repository](https://github.com/acsicuib/YAIFS)

[SUMO (Simulation of Urban MObility)](https://github.com/eclipse-sumo/sumo)

---

## ⚡ YAIFS Offloading & Scheduling Algorithms

The platform provides a unified RSU offloading interface (`choose_rsu(state) -> rsu_id`) implementing 8 CPU-only algorithms (plus greedy baseline) runnable with pure NumPy or scikit-learn:

1. **LinUCB** (`linucb`): Online contextual bandit with learned ridge reward estimate and UCB exploration bonus.
2. **Thompson Sampling** (`thompson`): Online bandit maintaining a Beta posterior per RSU with load and distance penalties.
3. **Ridge Regression** (`ridge`): Supervised latency prediction per (vehicle, RSU) choosing the lowest-latency RSU.
4. **Random Forest Classifier** (`random_forest`): Supervised deadline-miss probability predictor filtering high-risk RSUs.
5. **Markov Chain** (`markov`): Statistical transition model predicting next RSU from SUMO traces for proactive handover.
6. **Decision Tree** (`decision_tree`): Supervised trigger rule classifying whether staying causes deadline misses.
7. **Tabular Q-Learning** (`q_learning`): Reinforcement learning using coarse state discretization (load, distance, slack) and a 27x4 Q-table.
8. **K-Means Clustering** (`kmeans`): Unsupervised clustering grouping vehicles by position, heading, and speed, assigning cluster centroids to optimal nearby RSUs.
9. **Greedy Baseline** (`greedy`): Deadline-aware sticky handover + nearest non-overloaded offloading heuristic.

---

## **1. Repository Structure**

This project uses external systems and submodules:

```bash
root/
 ├── SUMO/       ← Microscopic vehicular traffic simulator
 ├── YAIFS/      ← Yet Another Intelligent Fog Simulator (v3.1.0)
 ├── sumoXyafs/  ← Co-simulation bridge, control policies, and ML offloading algorithms
 └── conference_paper/ ← IEEE conference paper and publication artifacts
```

Each submodule is connected to **our private forks**, not the public upstream repos.

---

## **2. Cloning the Project (Important!)**

Because this repo uses submodules, **you must clone with submodules included**.

### **Option A — Recommended (just the code)**

```bash
git clone https://github.com/Dking08/sumo_YAFS_mProject
cd sumo_YAFS_mProject
```

### **Option B — Not-Recommended (one-shot clone with submodules)**

```bash
git clone --recurse-submodules https://github.com/Dking08/sumo_YAFS_mProject
cd sumo_YAFS_mProject
```

### **Option C — If you already cloned normally**

```bash
git submodule update --init --recursive
```

---

## **3. Submodule Setup (Private Forks)**

Each submodule (e.g., `SUMO/`) is itself a git repo pointing to our **private fork**, while also tracking the **original upstream** as a secondary remote.

Inside each submodule, we maintain:

* `origin` → **Our private repository (read/write)**
* `upstream` → Public original repository (read-only, used for updates)

Example:

```bash
cd SUMO
git remote -v
# origin    https://github.com/Dking08/sumo-y-connect.git
# upstream  https://github.com/eclipse/sumo.git
```

This allows us to:

* Pull updates from upstream when needed
* Push our modifications to the private fork
* Keep our changes private and secure

---

## 4. Download
- SUMO - [Download](https://sumo.dlr.de/docs/Downloads.php)

---

## Prof Guide

* Prof. (Dr.) Amulya Ratna Swain
