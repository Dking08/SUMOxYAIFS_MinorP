# Mini Project - SUMO and YAFS integration

This mini project aims to explore the integration of Fog Computing concepts with the YAFS (Yet Another Fog Simulator) framework. The goal is to simulate and analyze the performance of applications deployed in a fog computing environment using YAFS.

[YAFS GitHub Repository](https://github.com/acsicuib/YAFS)
[SUMO (Simulation of Urban MObility)](https://github.com/eclipse-sumo/sumo)

## **1. Repository Structure**

This project uses **multiple submodules** to integrate external systems:

```bash
root/
 ├── SUMO/   ← Git submodule (our modified version of the SUMO)
 ├── YAFS/   ← Git submodule (our modified version of YAFS)
 └── /   ← Main code
```

Each submodule is connected to **our private forks**, not the public upstream repos.

---

## **2. Cloning the Project (Important!)**

Because this repo uses submodules, **you must clone with submodules included**.

### **Option A — Recommended (one-shot clone with submodules)**

```bash
git clone --recurse-submodules https://github.com/Dking08/sumo_YAFS_mProject
cd sumo_YAFS_mProject
```

### **Option B — If you already cloned normally**

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

## Prof Guide

* Prof. (Dr.) Amulya Ratna Swain
