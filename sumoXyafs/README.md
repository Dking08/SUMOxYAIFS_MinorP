To run:
```python realtime.py --start-sumo --gui --sumo-cfg sumo/delhi/osm.sumocfg --steps 500 --step-delay 0.5 --rsu-config rsu_config.json```

Each run now writes artifacts to a dedicated timestamped folder under `sumoXyafs/results/`:
- `results/run_YYYYmmdd_HHMMSS/raw/` -> YAFS runtime traces (`yafs_runtime.csv`, `yafs_runtime_link.csv`)
- `results/run_YYYYmmdd_HHMMSS/analysis/` -> step/decision/telemetry CSVs for analysis
- `results/run_YYYYmmdd_HHMMSS/plots/` -> generated figures (named with numeric prefixes)

You can provide a custom run folder name:
```python realtime.py --start-sumo --gui --sumo-cfg sumo/delhi/osm.sumocfg --run-name exp_baseline```

Auto-plot generation is enabled by default and currently produces feasible graphs:
- `01_mobility_vehicle_position_vs_time.png`
- `02_mobility_speed_density_vs_region.png`
- `03_network_v2i_distance_vs_link_latency.png`
- `04_perf_latency_vs_load.png`
- `05_perf_throughput_vs_density.png`
- `06_perf_utilization_vs_allocation.png`

For plots `03/04/05`, the pipeline now auto-falls back when YAFS runtime traces are empty/zero:
- `03`: uses scheduler-estimated network latency vs distance
- `04`: uses estimated latency vs load
- `05`: uses completed tasks vs density

When fallback mode is used, a note is rendered directly inside the figure.
To get full YAFS communication-performance conclusions (non-fallback), tune run conditions so YAFS emits event/link rows (e.g., non-trivial topology load, longer runs, and message flow that results in measurable transport/service timeline events).

Disable auto-plot generation when needed:
```python realtime.py --start-sumo --gui --sumo-cfg sumo/delhi/osm.sumocfg --no-plots```

Regenerate plots for an existing run directory:
```python -m analysis.generate_run_plots --run-dir results/run_YYYYmmdd_HHMMSS```

If sumo is running,
```sumo -c sumo/delhi/osm.sumocfg --remote-port 8873 --start```
```python realtime.py --port 8873 --steps 500 --rsu-config rsu_config.json```

GUI:
```sumo-gui -c sumo/delhi/osm.sumocfg --remote-port 8873 --start```
```python realtime.py --port 8873 --steps 500 --step-delay 0.5 --rsu-config rsu_config.json```

To enable a live runtime dashboard (terminal logs remain unchanged):
```python realtime.py --start-sumo --gui --sumo-cfg sumo/delhi/fog_test.sumocfg --steps 500 --step-delay 0.5 --rsu-config rsu_config.json --dashboard --dashboard-rows 25```

Quick dashboard/policy notes: in the map panel, RSU circles are static anchors (node_id `1` green-ish, `2` olive, others blue), 
vehicle dots are `red` when command is `slow_down`, `yellow` when execution RSU differs from observed RSU (migrate/offload path, with dashed link), and `cyan` when local/normal; each vehicle line points to its current execution RSU (fallback observed RSU). 
The top cards show step/vehicles/tasks/commands/scheduler/queues, and hover gives details. 
Reroute rule is: reroute if action is `migrate` or `offload`, or if speed is above reroute threshold (`--reroute-cooldown` only prevents spam per vehicle); 
deadline rule is: new task gets `deadline_step = current_step + ceil(task_deadline_s)`, and sticky handover is kept only when `estimated_finish_step <= deadline_step`, where estimate is roughly `current_step + ceil(queue_depth + remaining_compute_s + network_latency_s)`, otherwise migrate. 
Current network latency model is uncapped linear with distance and optional error scaling: `distance_latency_s = base + (max - base) * (distance_to_execution_rsu / norm)` and `distance_latency_s += distance_error * distance_latency_s`.
Overload rule is: an RSU is overloaded when queue depth reaches `min(overload_threshold, rsu_capacity)`, then offload target is nearest non-overloaded RSU by Euclidean distance. 
Additional args which can be used: `--rsu-config`, `--task-deadline`, `--overload-queue-threshold`, `--distance-latency-base`, `--distance-latency-max`, `--distance-latency-norm`, `--distance-error`, `--adaptive-latency-eta`, `--adaptive-latency-alpha`, `--adaptive-latency-weight`, `--reroute-cooldown`, `--rsu-capacity-default`, `--dashboard`, `--dashboard-rows` (compute time is currently fixed in code at 2s/task).

Adaptive ETA mode (`--adaptive-latency-eta`) keeps the existing distance-based latency term and adds an EWMA-smoothed measured YAFS link latency contribution:
- `adaptive_link_latency_s <- (1-alpha)*prev + alpha*measured`
- `network_latency_s <- distance_latency_s + adaptive_link_latency_s * weight`
- This is opt-in and disabled by default to preserve legacy behavior.

YAFS runtime traces are now written per run under `sumoXyafs/results/` as:
- `yafs_runtime.csv` (event/service timeline: includes `time_emit`, `time_reception`, `service`, `TOPO.src`, `TOPO.dst`)
- `yafs_runtime_link.csv` (link-level transfer latency)

Every 20 steps, the bridge prints a compact `[YAFS]` line with incremental averages from newly emitted trace rows (`e2e`, `service`, `link`) and `[YAFS-RSU]` lines for top RSUs by message count.

Generate an initial RSU config from the SUMO net:
```python tools/generate_rsu_config.py --net-file sumo/delhi/osm.net.xml.gz --count 9 --output rsu_config.json```

Notes for me (and...):
A sensor module is like an entrypoint to the yafs, so currently done is, sumo simulates the vehicles, the vehicles telemetry (which is generated by sumo), gets to a sensor module and yafs takes in the sensor data. Now, we are needed to do some sort of processing as per requirement by taking the data that we have got from the sensor module(s) in yafs and based on the computed results, communicate with other nodes, and then send back the updates to the sumo to modify the behaviour of the vehicle as required.

This is now a hybrid layer-2 bridge: telemetry capture and SUMO command apply are bridge-side, while per-step scheduling/control orchestration is centralized in `yafs_core.MinimalFogSim.process_step`.

Just keeping a note here, to present what "can-be-done", however not necessarily "to-be-done":
- Collision Avoidance (Fog computes collision risk, and sends braking commands to the vehicle)
- Dynamic Routing (Fog+Cloud computes the best route and...)
- Traffic light opt. - self-explanatory (however, this would require to export traffic-signal metrics from OSM, not sure if the current setup would work, not tested.)
- switch to ETA based reroute policy,
- current deadline estimation is queue_delay + compute_delay, however, a better service-time model should be implemented,
- current migration status: scheduling/task/control plus observed-RSU assignment moved into YAFS core adapter; Sensor/FogProcessor deployment is now RSU-node aware in YAFS placement; adaptive ETA can use measured YAFS link latency and per-RSU YAFS runtime attribution is available while SUMO I/O and visualization stay bridge-side,
- observability migration note: core YAFS trace extraction is now in place, next can be dashboard cards and per-RSU/per-task trace attribution.
