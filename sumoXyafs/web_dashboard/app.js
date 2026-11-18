// ============================================================================
// SUMO + YAFS Fog Computing Sandbox - 3D Hierarchical Orchestrator & Micro-Sim
// ============================================================================

// Main 3D / 2D Canvas State
let canvas, ctx;
let simState = null;

// View Modes: "3d" (Hierarchical Stacked Layers) vs "2d" (Geographic Map)
let viewMode = "3d";

// 3D Perspective Camera State
let viewAngle = -0.32;       // Rotation around Z axis (radians)
let pitch = 0.44;            // Perspective tilt compression
let layerSpacing = 210;      // Vertical gap between planes in pixels
let scale3d = 0.11;          // 3D World Scale
let panX3d = 0;              // 3D Pan X offset
let panY3d = 100;            // 3D Pan Y offset

// 2D Camera State
let scale2d = 0.16;
let panX2d = 40;
let panY2d = 20;

let isPanning = false;
let isRotating = false;
let panStart = { x: 0, y: 0 };

// Animation Loop State
let animFrameId = null;
let animClock = 0;

// Sandbox Modes: "run" vs "build"
let currentMode = "run";
let pendingBuildPos = null;

// Drag & Drop Task Reassignment State
let isDragging = false;
let draggedVehicleId = null;
let dragScreenPos = { x: 0, y: 0 };
let hoverTargetRsuId = null;
let activeHoverRsu = null;
let activeHoverVehicle = null;

// World & Geographic Dimensions
const WORLD_WIDTH = 4200;
const WORLD_HEIGHT = 9200;
const CENTER_X = WORLD_WIDTH / 2;
const CENTER_Y = WORLD_HEIGHT / 2;

// ============================================================================
// Left Panel: SUMO Micro-Simulator Engine (Delhi Network)
// ============================================================================
let sumoCanvas, sumoCtx;
let sumoRoadsData = null;
let sumoRoadsBounds = null;
let sumoScale = 0.05;
let sumoPanX = 10;
let sumoPanY = 10;
let sumoIsPanning = false;
let sumoPanStart = { x: 0, y: 0 };
let showSumoTrails = true;
let sumoVehicleTrails = {}; // Map of vehId -> [{x, y, alpha, angle}]
let selectedVehicleId = null;

// ----------------------------------------------------------------------------
// Initialization
// ----------------------------------------------------------------------------
document.addEventListener("DOMContentLoaded", () => {
  // Main Canvas
  canvas = document.getElementById("simCanvas");
  ctx = canvas.getContext("2d");

  // SUMO Mini Canvas
  sumoCanvas = document.getElementById("sumoCanvas");
  sumoCtx = sumoCanvas.getContext("2d");

  resizeCanvases();
  window.addEventListener("resize", resizeCanvases);

  setupEventListeners();
  setupPaneResizers();
  loadSumoRoadNetwork();
  startStatePolling();
  startAnimationLoop();
});

function onPaneResized() {
  if (canvas && canvas.parentElement) {
    const r1 = canvas.parentElement.getBoundingClientRect();
    canvas.width = r1.width;
    canvas.height = r1.height;
  }
  if (sumoCanvas && sumoCanvas.parentElement) {
    const r2 = sumoCanvas.parentElement.getBoundingClientRect();
    sumoCanvas.width = r2.width;
    sumoCanvas.height = r2.height;
  }
}

function resizeCanvases() {
  if (canvas && canvas.parentElement) {
    const r1 = canvas.parentElement.getBoundingClientRect();
    canvas.width = r1.width;
    canvas.height = r1.height;
  }
  if (sumoCanvas && sumoCanvas.parentElement) {
    const r2 = sumoCanvas.parentElement.getBoundingClientRect();
    sumoCanvas.width = r2.width;
    sumoCanvas.height = r2.height;
    if (sumoRoadsBounds) fitSumoMap();
  }
  if (viewMode === "3d") {
    zoomFit3D();
  } else {
    zoomFit2D();
  }
}

function zoomFit3D() {
  scale3d = Math.min(canvas.width / 5200, canvas.height / 6400) * 0.95;
  scale3d = Math.max(0.08, scale3d);
  viewAngle = -0.32;
  pitch = 0.44;
  layerSpacing = Math.min(260, Math.max(160, canvas.height * 0.28));
  panX3d = 0;
  panY3d = canvas.height * 0.12;
}

function zoomFit2D() {
  const sx = canvas.width / 4200;
  const sy = canvas.height / 9400;
  scale2d = Math.max(0.10, Math.min(sx, sy) * 0.92);
  panX2d = 30;
  panY2d = 20;
}

// ----------------------------------------------------------------------------
// 3D Coordinate Projections (Hierarchical Stacked Layers: 0, 1, 2)
// ----------------------------------------------------------------------------
function project3D(wx, wy, z_layer) {
  const dx = (wx - CENTER_X) * scale3d;
  const dy = (wy - CENTER_Y) * scale3d;

  const cosA = Math.cos(viewAngle);
  const sinA = Math.sin(viewAngle);

  // Isometric ground plane rotation
  const rx = dx * cosA - dy * sinA;
  const ry = (dx * sinA + dy * cosA) * pitch;

  // Vertical tier shelf elevation
  const rz = z_layer * layerSpacing;

  const screenX = canvas.width / 2 + rx + panX3d;
  const screenY = canvas.height / 2 + ry - rz + panY3d;

  return { x: screenX, y: screenY };
}

function unproject3D(screenX, screenY, z_layer) {
  const rz = z_layer * layerSpacing;
  const rx = screenX - (canvas.width / 2 + panX3d);
  const ry = (screenY - (canvas.height / 2 + panY3d - rz)) / pitch;

  const cosA = Math.cos(viewAngle);
  const sinA = Math.sin(viewAngle);

  const dx = rx * cosA + ry * sinA;
  const dy = -rx * sinA + ry * cosA;

  const wx = dx / scale3d + CENTER_X;
  const wy = dy / scale3d + CENTER_Y;

  return { x: wx, y: wy };
}

function worldToScreen(wx, wy, z_layer = 0) {
  if (viewMode === "3d") {
    return project3D(wx, wy, z_layer);
  }
  return { x: wx * scale2d + panX2d, y: wy * scale2d + panY2d };
}

function screenToWorld(sx, sy, z_layer = 0) {
  if (viewMode === "3d") {
    return unproject3D(sx, sy, z_layer);
  }
  return { x: (sx - panX2d) / scale2d, y: (sy - panY2d) / scale2d };
}

// ----------------------------------------------------------------------------
// Event Listeners & Interaction Controls
// ----------------------------------------------------------------------------
function setupEventListeners() {
  // Mode Buttons
  const modeRunBtn = document.getElementById("modeRunBtn");
  const modeBuildBtn = document.getElementById("modeBuildBtn");
  modeRunBtn.addEventListener("click", () => {
    currentMode = "run";
    modeRunBtn.classList.add("active-mode");
    modeBuildBtn.classList.remove("active-mode");
  });
  modeBuildBtn.addEventListener("click", () => {
    currentMode = "build";
    modeBuildBtn.classList.add("active-mode");
    modeRunBtn.classList.remove("active-mode");
  });

  // View Switcher: 3D vs 2D
  const view3dBtn = document.getElementById("view3dBtn");
  const view2dBtn = document.getElementById("view2dBtn");
  const perspectiveControls = document.getElementById("perspectiveControls");

  view3dBtn.addEventListener("click", () => {
    viewMode = "3d";
    view3dBtn.classList.add("active-mode");
    view3dBtn.classList.remove("btn-secondary");
    view2dBtn.classList.remove("active-mode");
    view2dBtn.classList.add("btn-secondary");
    perspectiveControls.style.display = "flex";
    zoomFit3D();
  });

  view2dBtn.addEventListener("click", () => {
    viewMode = "2d";
    view2dBtn.classList.add("active-mode");
    view2dBtn.classList.remove("btn-secondary");
    view3dBtn.classList.remove("active-mode");
    view3dBtn.classList.add("btn-secondary");
    perspectiveControls.style.display = "none";
    zoomFit2D();
  });

  // Simulation Controls
  document.getElementById("toggleAutoBtn").addEventListener("click", () => {
    fetch("/api/toggle_auto", { method: "POST" })
      .then(res => res.json())
      .then(data => {
        document.getElementById("toggleAutoBtn").textContent = data.auto_running ? "Pause" : "Resume";
      });
  });

  document.getElementById("stepBtn").addEventListener("click", () => {
    fetch("/api/step", { method: "POST" });
  });

  // 3D Perspective Controls
  document.getElementById("tiltUpBtn").addEventListener("click", () => {
    pitch = Math.min(0.85, pitch + 0.05);
  });
  document.getElementById("tiltDownBtn").addEventListener("click", () => {
    pitch = Math.max(0.20, pitch - 0.05);
  });
  document.getElementById("rotateLeftBtn").addEventListener("click", () => {
    viewAngle -= 0.15;
  });
  document.getElementById("rotateRightBtn").addEventListener("click", () => {
    viewAngle += 0.15;
  });
  document.getElementById("spacingUpBtn").addEventListener("click", () => {
    layerSpacing = Math.min(450, layerSpacing + 30);
  });
  document.getElementById("spacingDownBtn").addEventListener("click", () => {
    layerSpacing = Math.max(80, layerSpacing - 30);
  });
  document.getElementById("zoomInBtn").addEventListener("click", () => {
    if (viewMode === "3d") scale3d *= 1.25;
    else scale2d *= 1.25;
  });
  document.getElementById("zoomOutBtn").addEventListener("click", () => {
    if (viewMode === "3d") scale3d /= 1.25;
    else scale2d /= 1.25;
  });
  document.getElementById("zoomFitBtn").addEventListener("click", () => {
    if (viewMode === "3d") zoomFit3D();
    else zoomFit2D();
  });

  // Modal Dialogs
  document.getElementById("cancelRsuBtn").addEventListener("click", hideRsuModal);
  document.getElementById("confirmRsuBtn").addEventListener("click", confirmRsuBuild);

  // Main Canvas Mouse Controls
  canvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    const zoomFactor = e.deltaY < 0 ? 1.12 : 0.89;
    if (viewMode === "3d") {
      scale3d *= zoomFactor;
    } else {
      const mouseX = e.clientX - canvas.getBoundingClientRect().left;
      const mouseY = e.clientY - canvas.getBoundingClientRect().top;
      panX2d = mouseX - (mouseX - panX2d) * zoomFactor;
      panY2d = mouseY - (mouseY - panY2d) * zoomFactor;
      scale2d *= zoomFactor;
    }
  }, { passive: false });

  canvas.addEventListener("mousedown", onMouseDown);
  canvas.addEventListener("mousemove", onMouseMove);
  canvas.addEventListener("mouseup", onMouseUp);
  canvas.addEventListener("contextmenu", (e) => e.preventDefault());

  // SUMO Panel Toggle
  const sumoPanel = document.getElementById("sumoSidePanel");
  const toggleSumoBtn = document.getElementById("toggleSumoPanelBtn");
  toggleSumoBtn.addEventListener("click", () => {
    sumoPanel.classList.toggle("collapsed");
    toggleSumoBtn.textContent = sumoPanel.classList.contains("collapsed") ? "▶" : "◀";
    setTimeout(resizeCanvases, 260);
  });

  // SUMO Canvas Controls
  document.getElementById("sumoFitBtn").addEventListener("click", fitSumoMap);
  const trailsBtn = document.getElementById("sumoTrailsBtn");
  trailsBtn.addEventListener("click", () => {
    showSumoTrails = !showSumoTrails;
    trailsBtn.classList.toggle("active-mode", showSumoTrails);
  });

  sumoCanvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    const factor = e.deltaY < 0 ? 1.15 : 0.87;
    const rect = sumoCanvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    sumoPanX = mx - (mx - sumoPanX) * factor;
    sumoPanY = my - (my - sumoPanY) * factor;
    sumoScale *= factor;
  }, { passive: false });

  sumoCanvas.addEventListener("mousedown", (e) => {
    sumoIsPanning = true;
    sumoPanStart = { x: e.clientX - sumoPanX, y: e.clientY - sumoPanY };
  });

  window.addEventListener("mousemove", (e) => {
    if (sumoIsPanning) {
      sumoPanX = e.clientX - sumoPanStart.x;
      sumoPanY = e.clientY - sumoPanStart.y;
    }
  });

  window.addEventListener("mouseup", () => {
    sumoIsPanning = false;
  });
}

// ----------------------------------------------------------------------------
// Interactive Draggers / Panel Resizers
// ----------------------------------------------------------------------------
function setupPaneResizers() {
  const sumoPanel = document.getElementById("sumoSidePanel");
  const sidebar = document.getElementById("sidebarRight");
  const resizerLeft = document.getElementById("resizerLeft");
  const resizerRight = document.getElementById("resizerRight");

  if (!resizerLeft || !resizerRight) return;

  // Left resizer: adjusts SUMO micro-simulator width
  let isResizingLeft = false;
  let startXLeft = 0;
  let startWidthLeft = 0;

  resizerLeft.addEventListener("mousedown", (e) => {
    if (sumoPanel.classList.contains("collapsed")) return;
    isResizingLeft = true;
    startXLeft = e.clientX;
    startWidthLeft = sumoPanel.getBoundingClientRect().width;
    resizerLeft.classList.add("resizing");
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";
  });

  // Right resizer: adjusts Analytics sidebar width
  let isResizingRight = false;
  let startXRight = 0;
  let startWidthRight = 0;

  resizerRight.addEventListener("mousedown", (e) => {
    isResizingRight = true;
    startXRight = e.clientX;
    startWidthRight = sidebar.getBoundingClientRect().width;
    resizerRight.classList.add("resizing");
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";
  });

  window.addEventListener("mousemove", (e) => {
    if (isResizingLeft) {
      const dx = e.clientX - startXLeft;
      const minW = 200;
      const maxW = Math.max(minW, window.innerWidth * 0.45);
      const newWidth = Math.max(minW, Math.min(maxW, startWidthLeft + dx));
      sumoPanel.style.width = `${newWidth}px`;
      sumoPanel.style.minWidth = `${newWidth}px`;
      onPaneResized();
    }
    if (isResizingRight) {
      const dx = startXRight - e.clientX;
      const minW = 240;
      const maxW = Math.max(minW, window.innerWidth * 0.45);
      const newWidth = Math.max(minW, Math.min(maxW, startWidthRight + dx));
      sidebar.style.width = `${newWidth}px`;
      sidebar.style.minWidth = `${newWidth}px`;
      onPaneResized();
    }
  });

  window.addEventListener("mouseup", () => {
    if (isResizingLeft) {
      isResizingLeft = false;
      resizerLeft.classList.remove("resizing");
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
      onPaneResized();
    }
    if (isResizingRight) {
      isResizingRight = false;
      resizerRight.classList.remove("resizing");
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
      onPaneResized();
    }
  });
}

// ----------------------------------------------------------------------------
// Load SUMO Delhi Road Network Geometry
// ----------------------------------------------------------------------------
function loadSumoRoadNetwork() {
  fetch("/delhi_roads.json")
    .then(res => res.json())
    .then(data => {
      sumoRoadsData = data.roads;
      sumoRoadsBounds = data.bounds;
      fitSumoMap();
      renderSumoMap();
    })
    .catch(err => {
      console.warn("Could not load /delhi_roads.json:", err);
    });
}

function fitSumoMap() {
  if (!sumoRoadsBounds || !sumoCanvas) return;
  const bw = sumoRoadsBounds.max_x - sumoRoadsBounds.min_x;
  const bh = sumoRoadsBounds.max_y - sumoRoadsBounds.min_y;
  const sx = (sumoCanvas.width - 24) / bw;
  const sy = (sumoCanvas.height - 24) / bh;
  sumoScale = Math.min(sx, sy);
  sumoPanX = (sumoCanvas.width - bw * sumoScale) / 2 - sumoRoadsBounds.min_x * sumoScale;
  sumoPanY = (sumoCanvas.height - bh * sumoScale) / 2 + sumoRoadsBounds.max_y * sumoScale;
}

function sumoWorldToScreen(wx, wy) {
  // SUMO cartesian coordinates: Y increases upwards
  const sx = wx * sumoScale + sumoPanX;
  const sy = sumoPanY - wy * sumoScale;
  return { x: sx, y: sy };
}

// ----------------------------------------------------------------------------
// Mouse Interaction Logic (Drag & Drop, Hover HUD, Spawning)
// ----------------------------------------------------------------------------
function onMouseDown(e) {
  const rect = canvas.getBoundingClientRect();
  const mouseX = e.clientX - rect.left;
  const mouseY = e.clientY - rect.top;

  if (!simState) return;

  // Middle Click or Shift-Click: Camera Panning
  if (e.button === 1 || (e.shiftKey && e.button === 0)) {
    isPanning = true;
    panStart = {
      x: mouseX - (viewMode === "3d" ? panX3d : panX2d),
      y: mouseY - (viewMode === "3d" ? panY3d : panY2d)
    };
    return;
  }

  // Right Click: 3D Camera Rotation
  if (e.button === 2 && viewMode === "3d") {
    isRotating = true;
    panStart = { x: mouseX, y: mouseY, initialAngle: viewAngle, initialPitch: pitch };
    return;
  }

  // Build Mode: Click on Fog Plane (Layer 1) to Place or Delete RSU
  if (currentMode === "build") {
    if (simState.rsus) {
      for (const rsu of simState.rsus) {
        if (rsu.rsu_id === "cloud_node") continue;
        const sp = worldToScreen(rsu.x, rsu.y, 1);
        if (Math.hypot(mouseX - sp.x, mouseY - sp.y) < 32) {
          if (confirm(`Delete RSU Node ${rsu.rsu_id}?`)) {
            fetch("/api/delete_rsu", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ rsu_id: rsu.rsu_id })
            });
          }
          return;
        }
      }
    }
    const worldPos = screenToWorld(mouseX, mouseY, 1);
    showRsuModal(worldPos.x, worldPos.y);
    return;
  }

  // Run Mode: Check for Vehicle Drag-and-Drop on Layer 0
  if (simState.vehicles) {
    for (const v of simState.vehicles) {
      const sp = worldToScreen(v.x, v.y, 0);
      if (Math.hypot(mouseX - sp.x, mouseY - sp.y) < 28) {
        isDragging = true;
        draggedVehicleId = v.vehicle_id;
        selectedVehicleId = v.vehicle_id;
        dragScreenPos = { x: mouseX, y: mouseY };
        return;
      }
    }
  }

  // Run Mode: Tap on Physical Ground (Layer 0) to Spawn Vehicle
  const worldPos = screenToWorld(mouseX, mouseY, 0);
  const cat = document.getElementById("taskCatSelect").value;
  fetch("/api/spawn_vehicle", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ x: worldPos.x, y: worldPos.y, category: cat })
  });
}

function onMouseMove(e) {
  const rect = canvas.getBoundingClientRect();
  const mouseX = e.clientX - rect.left;
  const mouseY = e.clientY - rect.top;

  if (isPanning) {
    if (viewMode === "3d") {
      panX3d = mouseX - panStart.x;
      panY3d = mouseY - panStart.y;
    } else {
      panX2d = mouseX - panStart.x;
      panY2d = mouseY - panStart.y;
    }
    return;
  }

  if (isRotating && viewMode === "3d") {
    const dx = mouseX - panStart.x;
    const dy = mouseY - panStart.y;
    viewAngle = panStart.initialAngle + dx * 0.006;
    pitch = Math.max(0.20, Math.min(0.85, panStart.initialPitch + dy * 0.003));
    return;
  }

  // Hover detection across nodes
  activeHoverRsu = null;
  hoverTargetRsuId = null;
  activeHoverVehicle = null;

  const tooltip = document.getElementById("rsuHoverTooltip");

  // Check RSU hover on Layer 1 (Fog) or Layer 2 (Cloud)
  if (simState && simState.rsus) {
    for (const rsu of simState.rsus) {
      const z_lvl = rsu.rsu_id === "cloud_node" ? 2 : 1;
      const sp = worldToScreen(rsu.x, rsu.y, z_lvl);
      if (Math.hypot(mouseX - sp.x, mouseY - sp.y) < 32) {
        activeHoverRsu = rsu;
        hoverTargetRsuId = rsu.rsu_id;
        break;
      }
    }
  }

  // Check Vehicle hover on Layer 0
  if (!activeHoverRsu && simState && simState.vehicles) {
    for (const v of simState.vehicles) {
      const vp = worldToScreen(v.x, v.y, 0);
      if (Math.hypot(mouseX - vp.x, mouseY - vp.y) < 24) {
        activeHoverVehicle = v;
        break;
      }
    }
  }

  // Update Floating HUD Tooltip
  if (activeHoverRsu) {
    const z_lvl = activeHoverRsu.rsu_id === "cloud_node" ? 2 : 1;
    const sp = worldToScreen(activeHoverRsu.x, activeHoverRsu.y, z_lvl);
    tooltip.classList.remove("hidden");
    tooltip.style.left = `${Math.min(canvas.width - 310, Math.max(10, sp.x + 24))}px`;
    tooltip.style.top = `${Math.min(canvas.height - 240, Math.max(10, sp.y - 40))}px`;

    document.getElementById("ttRsuId").textContent = activeHoverRsu.rsu_id.toUpperCase();
    document.getElementById("ttHwBadge").textContent = (activeHoverRsu.hardware_profile || "AGX ORIN").toUpperCase();

    const maxGflops = activeHoverRsu.gflops || 20000;
    const cpuPct = activeHoverRsu.cpu_load_pct || 0;
    document.getElementById("ttGflopsVal").textContent = `${Math.round((cpuPct / 100) * maxGflops).toLocaleString()} / ${maxGflops.toLocaleString()} GF`;
    document.getElementById("ttCpuBar").style.width = `${cpuPct}%`;
    document.getElementById("ttRamVal").textContent = `${((activeHoverRsu.ram_used_mb || 0) / 1024).toFixed(1)} GB / ${(activeHoverRsu.ram_gb || 16).toFixed(0)} GB`;
    document.getElementById("ttQueueVal").textContent = `${activeHoverRsu.queue_depth || 0} queued tasks`;
    document.getElementById("ttLinksVal").textContent = `${activeHoverRsu.connected_links || 0} active V2X links`;
    document.getElementById("ttProtoVal").textContent = (activeHoverRsu.protocol || "c-v2x").toUpperCase();
  } else {
    tooltip.classList.add("hidden");
  }

  if (isDragging) {
    dragScreenPos = { x: mouseX, y: mouseY };
  }
}

function onMouseUp(e) {
  if (isPanning) {
    isPanning = false;
    return;
  }
  if (isRotating) {
    isRotating = false;
    return;
  }

  if (!isDragging) return;

  const rect = canvas.getBoundingClientRect();
  const mouseX = e.clientX - rect.left;
  const mouseY = e.clientY - rect.top;

  // Check drop onto any RSU on Fog Layer (Layer 1) or Cloud (Layer 2)
  if (simState && simState.rsus) {
    for (const rsu of simState.rsus) {
      const z_lvl = rsu.rsu_id === "cloud_node" ? 2 : 1;
      const sp = worldToScreen(rsu.x, rsu.y, z_lvl);
      if (Math.hypot(mouseX - sp.x, mouseY - sp.y) < 48) {
        fetch("/api/reassign_task", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ vehicle_id: draggedVehicleId, target_rsu: rsu.rsu_id })
        });
        break;
      }
    }
  }

  isDragging = false;
  draggedVehicleId = null;
  hoverTargetRsuId = null;
}

function showRsuModal(worldX, worldY) {
  pendingBuildPos = { x: worldX, y: worldY };
  document.getElementById("modalRsuId").value = `rsu_${(simState ? simState.rsus.length : 1)}`;
  document.getElementById("rsuModal").classList.remove("hidden");
}

function hideRsuModal() {
  document.getElementById("rsuModal").classList.add("hidden");
  pendingBuildPos = null;
}

function confirmRsuBuild() {
  if (!pendingBuildPos) return;
  const rsuId = document.getElementById("modalRsuId").value || "rsu_custom";
  const hw = document.getElementById("modalHwSelect").value;
  const v2x = document.getElementById("modalProtoSelect").value;

  fetch("/api/add_rsu", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      rsu_id: rsuId,
      x: pendingBuildPos.x,
      y: pendingBuildPos.y,
      hardware_profile: hw,
      v2x_protocol: v2x,
      capacity: 20
    })
  }).then(() => hideRsuModal());
}

// ----------------------------------------------------------------------------
// State Polling & Continuous Render Loop
// ----------------------------------------------------------------------------
function startStatePolling() {
  setInterval(() => {
    fetch("/api/state")
      .then(res => res.json())
      .then(data => {
        simState = data;
        updateUI();
      })
      .catch(() => {});
  }, 250);
}

function startAnimationLoop() {
  function loop() {
    animClock += 0.035;
    renderMainCanvas();
    renderSumoMap();
    animFrameId = requestAnimationFrame(loop);
  }
  animFrameId = requestAnimationFrame(loop);
}

// ----------------------------------------------------------------------------
// Sidebar & Badges UI Updates
// ----------------------------------------------------------------------------
function updateUI() {
  if (!simState) return;

  document.getElementById("stepBadge").textContent = `Step: ${simState.step}`;
  if (document.getElementById("completedBadge")) {
    document.getElementById("completedBadge").textContent = `Tasks Done: ${simState.completed_tasks || 0}`;
  }
  const algoBadge = document.getElementById("algoBadge");
  if (algoBadge && simState.algorithm) {
    algoBadge.textContent = `Algo: ${simState.algorithm.toUpperCase()}`;
  }

  const toggleBtn = document.getElementById("toggleAutoBtn");
  if (toggleBtn && simState.is_paused !== undefined) {
    toggleBtn.textContent = simState.is_paused ? "Resume" : "Pause";
  }

  const sumoBadge = document.getElementById("sumoBadge");
  const sumoStatusPill = document.getElementById("sumoStatusPill");
  if (simState.sumo_connected) {
    if (sumoBadge) {
      sumoBadge.textContent = "SUMO: Active TraCI";
      sumoBadge.className = "badge badge-success";
    }
    if (sumoStatusPill) {
      sumoStatusPill.textContent = "TraCI Synced";
      sumoStatusPill.className = "status-pill status-active";
    }
  } else {
    if (sumoBadge) {
      sumoBadge.textContent = "SUMO: Standalone";
      sumoBadge.className = "badge";
    }
    if (sumoStatusPill) {
      sumoStatusPill.textContent = "Standalone Engine";
      sumoStatusPill.className = "status-pill status-standalone";
    }
  }

  // SUMO Mini-Simulator Stats
  const vCount = simState.vehicles ? simState.vehicles.length : 0;
  document.getElementById("sumoVehCount").textContent = vCount;
  document.getElementById("sumoStepVal").textContent = simState.step;

  let totalSpeed = 0;
  if (simState.vehicles) {
    simState.vehicles.forEach(v => totalSpeed += (v.speed || 0));
  }
  const avgSpeedKmh = vCount > 0 ? (totalSpeed / vCount * 3.6).toFixed(1) : "0.0";
  document.getElementById("sumoAvgSpeed").textContent = `${avgSpeedKmh} km/h`;

  // RSU Queue Sliders
  const slidersContainer = document.getElementById("rsuSlidersContainer");
  const filteredRsus = simState.rsus ? simState.rsus.filter(r => r.rsu_id !== "cloud_node") : [];
  if (slidersContainer.children.length !== filteredRsus.length) {
    slidersContainer.innerHTML = "";
    filteredRsus.forEach(rsu => {
      const row = document.createElement("div");
      row.className = "slider-row";
      row.innerHTML = `
        <div class="slider-header">
          <span class="rsu-name">${rsu.rsu_id.toUpperCase()} (${rsu.hardware_profile})</span>
          <span class="rsu-val" id="val_${rsu.rsu_id}">Queue: ${rsu.queue_depth}</span>
        </div>
        <input type="range" min="0" max="25" value="${rsu.queue_depth}" id="slide_${rsu.rsu_id}">
      `;
      slidersContainer.appendChild(row);

      const slider = row.querySelector("input");
      slider.addEventListener("input", (e) => {
        const val = parseInt(e.target.value);
        document.getElementById(`val_${rsu.rsu_id}`).textContent = `Queue: ${val}`;
        fetch("/api/update_rsu_load", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ rsu_id: rsu.rsu_id, queue_depth: val })
        });
      });
    });
  } else {
    filteredRsus.forEach(rsu => {
      const slider = document.getElementById(`slide_${rsu.rsu_id}`);
      const valSpan = document.getElementById(`val_${rsu.rsu_id}`);
      if (slider && document.activeElement !== slider) {
        slider.value = rsu.queue_depth;
      }
      if (valSpan) {
        valSpan.textContent = `Queue: ${rsu.queue_depth}`;
      }
    });
  }

  // 4-Tier Distribution Bars
  const totalVeh = Math.max(1, simState.vehicles ? simState.vehicles.length : 1);
  const tc = simState.tier_counts || { V2V: 0, V2I: 0, FOG_MIGRATE: 0, CLOUD: 0 };
  updateTierBar("v2vBar", "v2vCount", tc.V2V || 0, totalVeh);
  updateTierBar("v2iBar", "v2iCount", tc.V2I || 0, totalVeh);
  updateTierBar("f2fBar", "f2fCount", tc.FOG_MIGRATE || 0, totalVeh);
  updateTierBar("cloudBar", "cloudCount", tc.CLOUD || 0, totalVeh);

  // Real-Time Physics & Decisions Table
  const tbody = document.getElementById("decisionsTableBody");
  if (simState.decisions) {
    tbody.innerHTML = simState.decisions.map(d => `
      <tr>
        <td><b>${d.vehicle_id}</b></td>
        <td>${d.observed_rsu} ➔ <span style="color:var(--accent-cyan)">${d.execution_rsu}</span></td>
        <td><span class="badge" style="font-size:10px">${d.tier}</span></td>
        <td>${(d.sinr || 0).toFixed(1)}</td>
        <td>${(d.data_rate_mbps || 0).toFixed(1)}</td>
        <td><span style="color:${d.status.includes('FINISHED') ? '#10b981' : '#f59e0b'}; font-weight:bold">${d.status || d.reason}</span></td>
      </tr>
    `).join("");
  }
}

function updateTierBar(barId, countId, count, total) {
  const el = document.getElementById(countId);
  if (el) el.textContent = count;
  const pct = Math.round((count / total) * 100);
  const bar = document.getElementById(barId);
  if (bar) bar.style.width = `${pct}%`;
}

// ============================================================================
// MAIN CANVAS: 3D HIERARCHICAL LAYERS & 2D MAP RENDERING
// ============================================================================
function renderMainCanvas() {
  if (!ctx || !simState) return;

  // Background
  ctx.fillStyle = "#070a0e";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  if (viewMode === "3d") {
    render3DHierarchy();
  } else {
    render2DGeographic();
  }
}

// ----------------------------------------------------------------------------
// 3D Hierarchical Architecture: Layer 0 (V2V) -> Layer 1 (Fog) -> Layer 2 (Cloud)
// ----------------------------------------------------------------------------
function render3DHierarchy() {
  // 1. Draw Layer Planes (Plates) from bottom to top
  // Layer 0: Physical Vehicles & V2V OBU (Z = 0)
  drawLayerPlate(0, {
    stroke: "rgba(16, 185, 129, 0.45)",
    fill: "rgba(16, 185, 129, 0.03)",
    gridColor: "rgba(16, 185, 129, 0.07)",
    title: "TIER 1: PHYSICAL VEHICLES & V2V OBU PLANE (Z=0)",
    color: "#10b981"
  });

  // Layer 1: Fog RSU Edge Mesh (Z = 1)
  drawLayerPlate(1, {
    stroke: "rgba(6, 182, 212, 0.55)",
    fill: "rgba(6, 182, 212, 0.04)",
    gridColor: "rgba(6, 182, 212, 0.08)",
    title: "TIER 2 & 3: FOG RSU EDGE MESH PLANE (Z=1)",
    color: "#06b6d4"
  });

  // Layer 2: Cloud Datacenter & Orchestration (Z = 2)
  drawCloudPlate(2, {
    stroke: "rgba(139, 92, 246, 0.65)",
    fill: "rgba(139, 92, 246, 0.05)",
    title: "TIER 4: CLOUD DATA CENTER & CENTRAL ORCHESTRATION (Z=2)",
    color: "#8b5cf6"
  });

  // 2. Vertical Alignment Guides: Drop Lines from Fog RSUs to Ground Footprint
  drawRsuGroundAnchors();

  // 3. Inter-RSU Fiber Backhaul Mesh on Fog Plane (Layer 1)
  drawInterRsuBackhaulMesh();

  // 4. Vertical Offloading Beams (V2I, Fog Migration, Cloud Column)
  drawOffloadingBeams();

  // 5. Draw Layer 0 (Physical Vehicles & V2V Halos)
  drawLayer0Vehicles();

  // 6. Draw Layer 1 (Fog RSU Edge Nodes with Crystal Clear Specs)
  drawLayer1Rsus();

  // 7. Draw Layer 2 (Cloud Datacenter Apex Node)
  drawLayer2CloudNode();

  // 8. Active Drag-and-Drop Task Reassignment Beam
  drawDragReassignmentBeam();
}

function drawLayerPlate(z_layer, style) {
  const p00 = worldToScreen(0, 0, z_layer);
  const p10 = worldToScreen(WORLD_WIDTH, 0, z_layer);
  const p11 = worldToScreen(WORLD_WIDTH, WORLD_HEIGHT, z_layer);
  const p01 = worldToScreen(0, WORLD_HEIGHT, z_layer);

  // Filled Glass Plate
  ctx.fillStyle = style.fill;
  ctx.beginPath();
  ctx.moveTo(p00.x, p00.y);
  ctx.lineTo(p10.x, p10.y);
  ctx.lineTo(p11.x, p11.y);
  ctx.lineTo(p01.x, p01.y);
  ctx.closePath();
  ctx.fill();

  // Glowing Plate Perimeter Border
  ctx.strokeStyle = style.stroke;
  ctx.lineWidth = 1.5;
  ctx.stroke();

  // Grid Lines on Plate
  ctx.strokeStyle = style.gridColor;
  ctx.lineWidth = 1;
  const gridStep = 800;
  for (let x = 0; x <= WORLD_WIDTH; x += gridStep) {
    const a = worldToScreen(x, 0, z_layer);
    const b = worldToScreen(x, WORLD_HEIGHT, z_layer);
    ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
  }
  for (let y = 0; y <= WORLD_HEIGHT; y += gridStep) {
    const a = worldToScreen(0, y, z_layer);
    const b = worldToScreen(WORLD_WIDTH, y, z_layer);
    ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
  }

  // Layer Badge Banner
  ctx.save();
  ctx.fillStyle = style.color;
  ctx.font = "bold 11px JetBrains Mono";
  ctx.textAlign = "left";
  ctx.fillText(style.title, p00.x + 10, p00.y - 8);
  ctx.restore();
}

function drawCloudPlate(z_layer, style) {
  // Cloud canopy diamond floating at top
  const padX = 600;
  const padY = 1200;
  const p00 = worldToScreen(padX, padY, z_layer);
  const p10 = worldToScreen(WORLD_WIDTH - padX, padY, z_layer);
  const p11 = worldToScreen(WORLD_WIDTH - padX, WORLD_HEIGHT - padY, z_layer);
  const p01 = worldToScreen(padX, WORLD_HEIGHT - padY, z_layer);

  ctx.fillStyle = style.fill;
  ctx.beginPath();
  ctx.moveTo(p00.x, p00.y);
  ctx.lineTo(p10.x, p10.y);
  ctx.lineTo(p11.x, p11.y);
  ctx.lineTo(p01.x, p01.y);
  ctx.closePath();
  ctx.fill();

  ctx.strokeStyle = style.stroke;
  ctx.lineWidth = 2;
  ctx.stroke();

  // Title
  ctx.fillStyle = style.color;
  ctx.font = "bold 11px JetBrains Mono";
  ctx.textAlign = "left";
  ctx.fillText(style.title, p00.x + 10, p00.y - 8);
}

// Vertical Guides from RSUs to Ground Footprints
function drawRsuGroundAnchors() {
  if (!simState || !simState.rsus) return;

  simState.rsus.forEach(rsu => {
    if (rsu.rsu_id === "cloud_node") return;
    const topPt = worldToScreen(rsu.x, rsu.y, 1);
    const gndPt = worldToScreen(rsu.x, rsu.y, 0);
    const isVrsu = Boolean(rsu.is_vehicular);

    // Dashed Pillar Line
    ctx.strokeStyle = isVrsu ? "rgba(245, 158, 11, 0.45)" : "rgba(6, 182, 212, 0.28)";
    ctx.lineWidth = isVrsu ? 1.6 : 1.2;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(topPt.x, topPt.y);
    ctx.lineTo(gndPt.x, gndPt.y);
    ctx.stroke();
    ctx.setLineDash([]);

    // Ground Shadow Footprint Ellipse
    const covM = rsu.coverage_radius || (isVrsu ? 350 : 500);
    const r = Math.max(12, covM * scale3d);
    ctx.fillStyle = isVrsu ? "rgba(245, 158, 11, 0.08)" : "rgba(6, 182, 212, 0.05)";
    ctx.strokeStyle = isVrsu ? "rgba(245, 158, 11, 0.35)" : "rgba(6, 182, 212, 0.22)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.ellipse(gndPt.x, gndPt.y, r, r * pitch, viewAngle, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();

    // Ground Center Dot
    ctx.fillStyle = isVrsu ? "rgba(245, 158, 11, 0.85)" : "rgba(6, 182, 212, 0.5)";
    ctx.beginPath();
    ctx.arc(gndPt.x, gndPt.y, isVrsu ? 4 : 3, 0, Math.PI * 2);
    ctx.fill();
  });
}

// Inter-RSU Backhaul Mesh on Fog Plane
function drawInterRsuBackhaulMesh() {
  if (!simState || !simState.rsus) return;
  const rsuMap = {};
  simState.rsus.forEach(r => rsuMap[r.rsu_id] = r);

  const drawnPairs = new Set();

  simState.rsus.forEach(rsu => {
    if (rsu.rsu_id === "cloud_node") return;
    const neighbors = rsu.neighbors || [];
    const p1 = worldToScreen(rsu.x, rsu.y, 1);

    neighbors.forEach(nbId => {
      const nb = rsuMap[nbId];
      if (!nb) return;
      const pairKey = [rsu.rsu_id, nbId].sort().join("<->");
      if (drawnPairs.has(pairKey)) return;
      drawnPairs.add(pairKey);

      const p2 = worldToScreen(nb.x, nb.y, 1);

      // Fiber backhaul line
      ctx.strokeStyle = "rgba(6, 182, 212, 0.35)";
      ctx.lineWidth = 2.0;
      ctx.beginPath();
      ctx.moveTo(p1.x, p1.y);
      ctx.lineTo(p2.x, p2.y);
      ctx.stroke();

      // Flowing data light pulse along fiber
      const pulseT = (animClock * 0.8 + (rsu.x + nb.x) * 0.001) % 1.0;
      const px = p1.x + (p2.x - p1.x) * pulseT;
      const py = p1.y + (p2.y - p1.y) * pulseT;
      ctx.fillStyle = "#06b6d4";
      ctx.beginPath();
      ctx.arc(px, py, 3.5, 0, Math.PI * 2);
      ctx.fill();
    });
  });
}

// Vertical 3D Offloading Beams (Vehicles to Tiers)
function drawOffloadingBeams() {
  if (!simState || !simState.vehicles || !simState.rsus) return;
  const rsuMap = {};
  simState.rsus.forEach(r => rsuMap[r.rsu_id] = r);

  simState.vehicles.forEach(v => {
    const vp = worldToScreen(v.x, v.y, 0); // Vehicle is on Layer 0

    if (v.tier === "V2V" || v.execution_rsu === "local_obu") {
      // Tier 1: Local OBU / V2V - Concentric emerald wave on Layer 0
      const pulseR = 12 + (animClock * 18) % 22;
      ctx.strokeStyle = `rgba(16, 185, 129, ${Math.max(0, 1 - pulseR / 34)})`;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.ellipse(vp.x, vp.y, pulseR, pulseR * pitch, viewAngle, 0, Math.PI * 2);
      ctx.stroke();
      return;
    }

    if (v.tier === "CLOUD" || v.execution_rsu === "cloud_node") {
      // Tier 4: Vertical Purple Beam to Cloud on Layer 2
      const cloud = rsuMap["cloud_node"];
      const cp = cloud ? worldToScreen(cloud.x, cloud.y, 2) : worldToScreen(CENTER_X, 800, 2);

      // Glowing vertical column
      const grad = ctx.createLinearGradient(vp.x, vp.y, cp.x, cp.y);
      grad.addColorStop(0, "rgba(16, 185, 129, 0.8)");
      grad.addColorStop(0.5, "rgba(139, 92, 246, 0.9)");
      grad.addColorStop(1, "rgba(192, 132, 252, 1.0)");

      ctx.strokeStyle = grad;
      ctx.lineWidth = 3.0;
      ctx.beginPath();
      ctx.moveTo(vp.x, vp.y);
      ctx.lineTo(cp.x, cp.y);
      ctx.stroke();

      // Traveling energy packet
      const t = (animClock * 1.5 + v.x * 0.002) % 1.0;
      const pktX = vp.x + (cp.x - vp.x) * t;
      const pktY = vp.y + (cp.y - vp.y) * t;
      ctx.fillStyle = "#c084fc";
      ctx.shadowColor = "#8b5cf6";
      ctx.shadowBlur = 8;
      ctx.beginPath(); ctx.arc(pktX, pktY, 4.5, 0, Math.PI * 2); ctx.fill();
      ctx.shadowBlur = 0;
      return;
    }

    // Tier 2 (V2I), vRSU or Tier 3 (Fog Migration)
    const targetRsu = rsuMap[v.execution_rsu] || rsuMap[v.observed_rsu];
    if (targetRsu) {
      const rp = worldToScreen(targetRsu.x, targetRsu.y, 1);
      const isVrsu = Boolean(targetRsu.is_vehicular) || v.tier === "vRSU" || String(v.execution_rsu).startsWith("vrsu");
      const isMigrate = v.tier === "FOG_MIGRATE";

      // Beam color: Amber/Gold for vRSU, Amber for Fog Migration, Cyan for V2I
      const beamColor = isVrsu ? "#fbbf24" : (isMigrate ? "#f59e0b" : "#06b6d4");
      ctx.strokeStyle = beamColor;
      ctx.lineWidth = isVrsu ? 2.8 : (isMigrate ? 2.5 : 2.0);
      if (isMigrate) ctx.setLineDash([6, 4]);
      if (isVrsu) ctx.setLineDash([4, 2]);

      ctx.beginPath();
      ctx.moveTo(vp.x, vp.y);
      ctx.lineTo(rp.x, rp.y);
      ctx.stroke();
      ctx.setLineDash([]);

      // Traveling data packet
      const t = (animClock * 1.2 + v.y * 0.001) % 1.0;
      const px = vp.x + (rp.x - vp.x) * t;
      const py = vp.y + (rp.y - vp.y) * t;
      ctx.fillStyle = isMigrate ? "#f59e0b" : "#06b6d4";
      ctx.beginPath(); ctx.arc(px, py, 3.5, 0, Math.PI * 2); ctx.fill();
    }
  });
}

// Layer 0: Physical Vehicles on Road Plane
function drawLayer0Vehicles() {
  if (!simState || !simState.vehicles) return;

  simState.vehicles.forEach(v => {
    const vp = worldToScreen(v.x, v.y, 0);
    const isSelected = selectedVehicleId === v.vehicle_id;
    const isSafety = v.category === "safety";
    const baseColor = isSafety ? "#ef4444" : (v.category === "global" ? "#8b5cf6" : "#10b981");

    // Vehicle Core Node
    ctx.fillStyle = baseColor;
    ctx.beginPath();
    ctx.arc(vp.x, vp.y, 8, 0, Math.PI * 2);
    ctx.fill();

    // Directional Heading Indicator
    const headingRad = ((v.angle || 90.0) - 90) * (Math.PI / 180);
    const arrowLen = 14;
    const ax = vp.x + Math.cos(headingRad) * arrowLen;
    const ay = vp.y + Math.sin(headingRad) * arrowLen;
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 2.0;
    ctx.beginPath(); ctx.moveTo(vp.x, vp.y); ctx.lineTo(ax, ay); ctx.stroke();

    // Outer Compute Progress Ring
    const prog = v.progress || 0.0;
    ctx.strokeStyle = baseColor;
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    ctx.arc(vp.x, vp.y, 13, -Math.PI / 2, -Math.PI / 2 + prog * Math.PI * 2);
    ctx.stroke();

    // Vehicle Label & Progress Pill
    ctx.fillStyle = "#ffffff";
    ctx.font = "bold 10px Inter";
    ctx.textAlign = "center";
    ctx.fillText(`${v.vehicle_id} (${Math.round(prog * 100)}%)`, vp.x, vp.y - 18);

    // OBU Spec Tag
    ctx.fillStyle = "rgba(255, 255, 255, 0.65)";
    ctx.font = "8px JetBrains Mono";
    ctx.fillText(`[${v.tier || 'V2I'}] OBU NANO`, vp.x, vp.y + 20);

    if (isSelected) {
      ctx.strokeStyle = "#f59e0b";
      ctx.lineWidth = 2.5;
      ctx.beginPath(); ctx.arc(vp.x, vp.y, 18, 0, Math.PI * 2); ctx.stroke();
    }
  });
}

// Layer 1: Fog RSU Edge Nodes with Crystal Clear Hardware Specs
function drawLayer1Rsus() {
  if (!simState || !simState.rsus) return;

  simState.rsus.forEach(rsu => {
    if (rsu.rsu_id === "cloud_node") return;
    const sp = worldToScreen(rsu.x, rsu.y, 1);
    const isHovered = hoverTargetRsuId === rsu.rsu_id;
    const isOverloaded = rsu.queue_depth > 2;
    const isVrsu = Boolean(rsu.is_vehicular);

    // Coverage Circle on Fog Plane
    const covM = rsu.coverage_radius || (isVrsu ? 350 : 550);
    const covRadius = Math.max(24, covM * scale3d);
    ctx.strokeStyle = isHovered ? "rgba(245, 158, 11, 0.7)" : (isVrsu ? "rgba(245, 158, 11, 0.35)" : "rgba(6, 182, 212, 0.22)");
    ctx.fillStyle = isHovered ? "rgba(245, 158, 11, 0.12)" : (isVrsu ? "rgba(245, 158, 11, 0.05)" : "rgba(6, 182, 212, 0.03)");
    ctx.lineWidth = isHovered ? 2.5 : (isVrsu ? 1.5 : 1);
    ctx.beginPath();
    ctx.ellipse(sp.x, sp.y, covRadius, covRadius * pitch, viewAngle, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();

    // RSU Node Base Body
    const nodeColor = isVrsu ? (isOverloaded ? "#ea580c" : "#f59e0b") : (isOverloaded ? "#f59e0b" : "#06b6d4");
    ctx.fillStyle = nodeColor;
    ctx.beginPath();
    ctx.arc(sp.x, sp.y, isVrsu ? 15 : 14, 0, Math.PI * 2);
    ctx.fill();

    // Pulse wave for moving vRSU
    if (isVrsu) {
      const vwave = 15 + (animClock * 14) % 18;
      ctx.strokeStyle = `rgba(245, 158, 11, ${Math.max(0, 0.9 - vwave / 33)})`;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.arc(sp.x, sp.y, vwave, 0, Math.PI * 2);
      ctx.stroke();
    }

    // CPU Utilization Circular Dial Gauge
    const loadPct = (rsu.cpu_load_pct || 0) / 100;
    ctx.strokeStyle = loadPct > 0.8 ? "#ef4444" : (loadPct > 0.4 ? "#f59e0b" : "#10b981");
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.arc(sp.x, sp.y, 19, -Math.PI / 2, -Math.PI / 2 + loadPct * Math.PI * 2);
    ctx.stroke();

    // Hover Highlight Halo
    if (isHovered) {
      ctx.strokeStyle = "#f59e0b";
      ctx.lineWidth = 3;
      ctx.beginPath(); ctx.arc(sp.x, sp.y, 25, 0, Math.PI * 2); ctx.stroke();
    }

    // Node Title & Hardware Badge
    ctx.fillStyle = isVrsu ? "#fbbf24" : "#ffffff";
    ctx.font = isVrsu ? "bold 13px Inter" : "bold 12px Inter";
    ctx.textAlign = "center";
    const title = isVrsu ? `[vRSU] ${rsu.rsu_id.toUpperCase()}` : rsu.rsu_id.toUpperCase();
    ctx.fillText(title, sp.x, sp.y - 28);

    // Hardware Spec Label
    const hwName = isVrsu ? "vRSU ORIN" : (rsu.hardware_profile === "jetson_agx_orin" ? "AGX ORIN" : "SIEMENS RS22X");
    ctx.fillStyle = isOverloaded ? "#f59e0b" : (isVrsu ? "#fbbf24" : "#06b6d4");
    ctx.font = "bold 10px JetBrains Mono";
    ctx.fillText(`${hwName} (${(rsu.gflops || 3200).toLocaleString()} GF)`, sp.x, sp.y + 32);

    // Protocol & Queue Depth Badges
    ctx.fillStyle = "rgba(255, 255, 255, 0.75)";
    ctx.font = "9px JetBrains Mono";
    ctx.fillText(`[${(rsu.protocol || 'c-v2x').toUpperCase()}] • Q: ${rsu.queue_depth}`, sp.x, sp.y + 44);
  });
}

// Layer 2: Cloud Datacenter Apex Node
function drawLayer2CloudNode() {
  if (!simState || !simState.rsus) return;
  const cloud = simState.rsus.find(r => r.rsu_id === "cloud_node");
  const cp = cloud ? worldToScreen(cloud.x, cloud.y, 2) : worldToScreen(CENTER_X, 800, 2);
  const isHovered = hoverTargetRsuId === "cloud_node";

  // Concentric Orbital Cloud Rings
  const wave1 = 26 + (animClock * 12) % 20;
  ctx.strokeStyle = `rgba(139, 92, 246, ${Math.max(0, 0.8 - wave1 / 46)})`;
  ctx.lineWidth = 2;
  ctx.beginPath(); ctx.arc(cp.x, cp.y, wave1, 0, Math.PI * 2); ctx.stroke();

  // Cloud Center Body
  ctx.fillStyle = isHovered ? "#a855f7" : "#8b5cf6";
  ctx.beginPath();
  ctx.arc(cp.x, cp.y, 20, 0, Math.PI * 2);
  ctx.fill();

  // Cloud Center Text
  ctx.fillStyle = "#ffffff";
  ctx.font = "bold 9px JetBrains Mono";
  ctx.textAlign = "center";
  ctx.fillText("CLOUD", cp.x, cp.y + 3);

  // Cloud Title & Specs
  ctx.fillStyle = "#ffffff";
  ctx.font = "bold 13px Inter";
  ctx.fillText("CLOUD DATA CENTER", cp.x, cp.y - 28);

  ctx.fillStyle = "#c084fc";
  ctx.font = "bold 10px JetBrains Mono";
  ctx.fillText("1,000,000 GFLOPS • 512GB RAM", cp.x, cp.y + 32);

  ctx.fillStyle = "rgba(255, 255, 255, 0.75)";
  ctx.font = "9px JetBrains Mono";
  ctx.fillText(`ACTIVE LINKS: ${cloud ? cloud.connected_links : 0} • WAN 5G`, cp.x, cp.y + 44);
}

// Active Drag-to-Reassign Target Beam
function drawDragReassignmentBeam() {
  if (!isDragging || !draggedVehicleId || !simState || !simState.vehicles) return;
  const v = simState.vehicles.find(item => item.vehicle_id === draggedVehicleId);
  if (!v) return;

  const vp = worldToScreen(v.x, v.y, 0);

  // Vehicle ring
  ctx.strokeStyle = "#f59e0b";
  ctx.lineWidth = 3;
  ctx.beginPath(); ctx.arc(vp.x, vp.y, 20, 0, Math.PI * 2); ctx.stroke();

  // Connecting dashed line to mouse
  ctx.strokeStyle = "#f59e0b";
  ctx.lineWidth = 3;
  ctx.setLineDash([6, 4]);
  ctx.beginPath();
  ctx.moveTo(vp.x, vp.y);
  ctx.lineTo(dragScreenPos.x, dragScreenPos.y);
  ctx.stroke();
  ctx.setLineDash([]);

  // Label
  ctx.fillStyle = "#f59e0b";
  ctx.font = "bold 12px Inter";
  ctx.textAlign = "center";
  ctx.fillText("Drop onto target RSU or Cloud", dragScreenPos.x, dragScreenPos.y - 14);
}

// ----------------------------------------------------------------------------
// 2D Geographic Map Render Mode (Classic Fallback)
// ----------------------------------------------------------------------------
function render2DGeographic() {
  // Grid Lines
  ctx.strokeStyle = "rgba(255, 255, 255, 0.04)";
  ctx.lineWidth = 1;
  const gridStep = 500;
  for (let x = 0; x < WORLD_WIDTH; x += gridStep) {
    const sp = worldToScreen(x, 0);
    ctx.beginPath(); ctx.moveTo(sp.x, 0); ctx.lineTo(sp.x, canvas.height); ctx.stroke();
  }
  for (let y = 0; y < WORLD_HEIGHT; y += gridStep) {
    const sp = worldToScreen(0, y);
    ctx.beginPath(); ctx.moveTo(0, sp.y); ctx.lineTo(canvas.width, sp.y); ctx.stroke();
  }

  // RSUs
  if (simState.rsus) {
    simState.rsus.forEach(rsu => {
      const sp = worldToScreen(rsu.x, rsu.y);
      const isHovered = hoverTargetRsuId === rsu.rsu_id;
      const rsuRadius = Math.max(12, 16 * scale2d);

      if (rsu.rsu_id === "cloud_node") {
        ctx.fillStyle = isHovered ? "#a855f7" : "#8b5cf6";
        ctx.beginPath(); ctx.arc(sp.x, sp.y, rsuRadius + 6, 0, Math.PI * 2); ctx.fill();
        ctx.fillStyle = "#ffffff";
        ctx.font = "bold 11px JetBrains Mono"; ctx.textAlign = "center";
        ctx.fillText("CLOUD", sp.x, sp.y + 4);
      } else {
        const isVrsu = Boolean(rsu.is_vehicular);
        const covM = rsu.coverage_radius || (isVrsu ? 350 : 600);
        const radius = Math.max(isVrsu ? 40 : 60, covM * scale2d);
        ctx.strokeStyle = isHovered ? "rgba(245, 158, 11, 0.6)" : (isVrsu ? "rgba(245, 158, 11, 0.35)" : "rgba(6, 182, 212, 0.15)");
        ctx.fillStyle = isHovered ? "rgba(245, 158, 11, 0.12)" : (isVrsu ? "rgba(245, 158, 11, 0.06)" : "rgba(6, 182, 212, 0.03)");
        ctx.lineWidth = isHovered ? 3 : (isVrsu ? 2 : 1.5);
        ctx.beginPath(); ctx.arc(sp.x, sp.y, radius, 0, Math.PI * 2); ctx.fill(); ctx.stroke();

        ctx.fillStyle = isVrsu ? (rsu.queue_depth > 2 ? "#ea580c" : "#f59e0b") : (rsu.queue_depth > 2 ? "#f59e0b" : "#06b6d4");
        ctx.beginPath(); ctx.arc(sp.x, sp.y, rsuRadius, 0, Math.PI * 2); ctx.fill();

        ctx.fillStyle = isVrsu ? "#fbbf24" : "#ffffff";
        ctx.font = isVrsu ? "bold 12px Inter" : "bold 12px Inter"; ctx.textAlign = "center";
        const title = isVrsu ? `[vRSU] ${rsu.rsu_id.toUpperCase()}` : rsu.rsu_id.toUpperCase();
        ctx.fillText(title, sp.x, sp.y - rsuRadius - 6);
      }
    });
  }

  // Transmission Links & Vehicles
  if (simState.vehicles && simState.rsus) {
    const rsuMap = {};
    simState.rsus.forEach(r => rsuMap[r.rsu_id] = r);

    simState.vehicles.forEach(v => {
      const vp = worldToScreen(v.x, v.y);
      const vehRadius = Math.max(9, 12 * scale2d);

      if (v.tier !== "V2V" && v.execution_rsu !== "local_obu") {
        const targetRsu = rsuMap[v.execution_rsu];
        if (targetRsu) {
          const tp = worldToScreen(targetRsu.x, targetRsu.y);
          const isVrsu = Boolean(targetRsu.is_vehicular) || v.tier === "vRSU" || String(v.execution_rsu).startsWith("vrsu");
          ctx.strokeStyle = v.tier === "CLOUD" ? "#8b5cf6" : (isVrsu ? "#fbbf24" : (v.tier === "FOG_MIGRATE" ? "#f59e0b" : "#06b6d4"));
          ctx.lineWidth = isVrsu ? 3.0 : 2.5;
          ctx.beginPath(); ctx.moveTo(vp.x, vp.y); ctx.lineTo(tp.x, tp.y); ctx.stroke();
        }
      }

      ctx.fillStyle = v.category === "safety" ? "#ef4444" : "#10b981";
      ctx.beginPath(); ctx.arc(vp.x, vp.y, vehRadius, 0, Math.PI * 2); ctx.fill();

      ctx.fillStyle = "#ffffff";
      ctx.font = "bold 11px Inter"; ctx.textAlign = "center";
      ctx.fillText(v.vehicle_id, vp.x, vp.y - vehRadius - 6);
    });
  }
}

// ============================================================================
// LEFT PANEL: SUMO MICRO-SIMULATOR (DELHI ROAD NETWORK RENDERING)
// ============================================================================
function renderSumoMap() {
  if (!sumoCtx || !sumoCanvas) return;

  // Background
  sumoCtx.fillStyle = "#06090d";
  sumoCtx.fillRect(0, 0, sumoCanvas.width, sumoCanvas.height);

  // 1. Draw Delhi Road Network Geometry
  if (sumoRoadsData && sumoRoadsData.length > 0) {
    sumoCtx.strokeStyle = "rgba(100, 116, 139, 0.45)";
    sumoCtx.lineWidth = 1.0;
    sumoCtx.beginPath();

    for (let i = 0; i < sumoRoadsData.length; i++) {
      const road = sumoRoadsData[i];
      if (road.length < 2) continue;
      const p0 = sumoWorldToScreen(road[0][0], road[0][1]);
      sumoCtx.moveTo(p0.x, p0.y);
      for (let j = 1; j < road.length; j++) {
        const pj = sumoWorldToScreen(road[j][0], road[j][1]);
        sumoCtx.lineTo(pj.x, pj.y);
      }
    }
    sumoCtx.stroke();
  } else {
    sumoCtx.fillStyle = "rgba(255, 255, 255, 0.3)";
    sumoCtx.font = "11px Inter";
    sumoCtx.textAlign = "center";
    sumoCtx.fillText("Loading Delhi Road Network...", sumoCanvas.width / 2, sumoCanvas.height / 2);
  }

  // 2. Draw RSUs on the Delhi Map
  if (simState && simState.rsus) {
    simState.rsus.forEach(rsu => {
      if (rsu.rsu_id === "cloud_node") return;
      const sp = sumoWorldToScreen(rsu.x, rsu.y);
      const isVrsu = Boolean(rsu.is_vehicular);

      // Coverage Ring
      const covM = rsu.coverage_radius || (isVrsu ? 350 : 600);
      const covR = Math.max(6, covM * sumoScale);
      sumoCtx.strokeStyle = isVrsu ? "rgba(245, 158, 11, 0.45)" : "rgba(6, 182, 212, 0.25)";
      sumoCtx.fillStyle = isVrsu ? "rgba(245, 158, 11, 0.10)" : "rgba(6, 182, 212, 0.05)";
      sumoCtx.lineWidth = isVrsu ? 1.5 : 1;
      sumoCtx.beginPath(); sumoCtx.arc(sp.x, sp.y, covR, 0, Math.PI * 2); sumoCtx.fill(); sumoCtx.stroke();

      // RSU Node Point
      sumoCtx.fillStyle = isVrsu ? "#f59e0b" : "#06b6d4";
      sumoCtx.beginPath(); sumoCtx.arc(sp.x, sp.y, isVrsu ? 5 : 4, 0, Math.PI * 2); sumoCtx.fill();
      sumoCtx.beginPath();
      sumoCtx.arc(sp.x, sp.y, 4, 0, Math.PI * 2);
      sumoCtx.fill();
    });
  }

  // 3. Motion Trails for Real Vehicles
  if (showSumoTrails && simState && simState.vehicles) {
    simState.vehicles.forEach(v => {
      if (!sumoVehicleTrails[v.vehicle_id]) {
        sumoVehicleTrails[v.vehicle_id] = [];
      }
      const history = sumoVehicleTrails[v.vehicle_id];
      history.push({ x: v.x, y: v.y, time: animClock });
      if (history.length > 12) history.shift();

      if (history.length > 1) {
        sumoCtx.strokeStyle = v.category === "safety" ? "rgba(239, 68, 68, 0.35)" : "rgba(16, 185, 129, 0.35)";
        sumoCtx.lineWidth = 2.0;
        sumoCtx.beginPath();
        const p0 = sumoWorldToScreen(history[0].x, history[0].y);
        sumoCtx.moveTo(p0.x, p0.y);
        for (let k = 1; k < history.length; k++) {
          const pk = sumoWorldToScreen(history[k].x, history[k].y);
          sumoCtx.lineTo(pk.x, pk.y);
        }
        sumoCtx.stroke();
      }
    });
  }

  // 4. Draw Vehicles on the Real Road Network
  if (simState && simState.vehicles) {
    simState.vehicles.forEach(v => {
      const sp = sumoWorldToScreen(v.x, v.y);
      const isSelected = selectedVehicleId === v.vehicle_id;
      const isSafety = v.category === "safety";

      // Vehicle Dot with Heading Arrow
      sumoCtx.fillStyle = isSafety ? "#ef4444" : (v.category === "global" ? "#8b5cf6" : "#10b981");
      sumoCtx.beginPath();
      sumoCtx.arc(sp.x, sp.y, 5, 0, Math.PI * 2);
      sumoCtx.fill();

      // Heading Chevron
      const angRad = ((v.angle || 0) - 90) * (Math.PI / 180);
      const hx = sp.x + Math.cos(angRad) * 8;
      const hy = sp.y - Math.sin(angRad) * 8; // Y inverted in screen coords
      sumoCtx.strokeStyle = "#ffffff";
      sumoCtx.lineWidth = 1.8;
      sumoCtx.beginPath(); sumoCtx.moveTo(sp.x, sp.y); sumoCtx.lineTo(hx, hy); sumoCtx.stroke();

      if (isSelected) {
        sumoCtx.strokeStyle = "#f59e0b";
        sumoCtx.lineWidth = 2;
        sumoCtx.beginPath(); sumoCtx.arc(sp.x, sp.y, 9, 0, Math.PI * 2); sumoCtx.stroke();
      }
    });
  }
}
