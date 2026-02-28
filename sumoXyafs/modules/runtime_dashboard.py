"""Minimal Tkinter dashboard for live fog-control observability."""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class _Node:
    rsu_id: str
    x: float
    y: float
    node_id: int


class _HoverTip:
    def __init__(self, tk, widget, text_provider: Callable[[], str]):
        self.tk = tk
        self.widget = widget
        self.text_provider = text_provider
        self.tip = None
        self.widget.bind("<Enter>", self._show)
        self.widget.bind("<Leave>", self._hide)

    def _show(self, _event=None):
        if self.tip is not None:
            return
        text = self.text_provider().strip()
        if not text:
            return
        self.tip = self.tk.Toplevel(self.widget)
        self.tip.wm_overrideredirect(True)
        x = self.widget.winfo_pointerx() + 12
        y = self.widget.winfo_pointery() + 12
        self.tip.wm_geometry(f"+{x}+{y}")
        label = self.tk.Label(
            self.tip,
            text=text,
            justify="left",
            background="#111111",
            foreground="#f0f0f0",
            relief="solid",
            borderwidth=1,
            padx=8,
            pady=6,
            font=("Consolas", 9),
        )
        label.pack()

    def _hide(self, _event=None):
        if self.tip is None:
            return
        try:
            self.tip.destroy()
        except Exception:
            pass
        self.tip = None


class RuntimeDashboard:
    def __init__(self, enabled: bool = False, max_rows: int = 20):
        self.enabled = bool(enabled)
        self.max_rows = max(5, int(max_rows))
        self.available = False

        self._sched_events: deque[str] = deque(maxlen=self.max_rows)
        self._cmd_events: deque[str] = deque(maxlen=self.max_rows)
        self._cmd_counts = Counter()
        self._sched_counts = Counter()
        self._step = 0
        self._steps_total = 0
        self._vehicles = 0
        self._active_tasks = 0
        self._queue_depths: dict[str, int] = {}
        self._last_cmds = 0

        self._nodes: dict[str, _Node] = {}
        self._node_items: dict[str, dict[str, int]] = {}
        self._vehicle_items: dict[str, dict[str, int]] = {}
        self._vehicle_state: dict[str, dict] = {}
        self._seen_this_step: set[str] = set()
        self._bounds: Optional[tuple[float, float, float, float]] = None
        self._show_labels = False

        if not self.enabled:
            return

        try:
            import tkinter as tk
        except Exception as exc:
            print(f"[DASH] Tkinter unavailable. Continuing without dashboard. ({exc})", flush=True)
            self.enabled = False
            return

        self.tk = tk
        try:
            self.root = tk.Tk()
        except Exception as exc:
            print(f"[DASH] Could not open dashboard window. Continuing without dashboard. ({exc})", flush=True)
            self.enabled = False
            return

        self.available = True
        self.root.title("SUMO-YAFS Runtime Dashboard")
        self.root.geometry("1220x760")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._build_layout()
        self._render()
        self._pump_events()

    def _build_layout(self) -> None:
        tk = self.tk
        self.root.configure(bg="#161a1d")

        self.step_var = tk.StringVar(value="0 / 0")
        self.veh_var = tk.StringVar(value="0")
        self.task_var = tk.StringVar(value="0")
        self.cmd_var = tk.StringVar(value="0")
        self.sched_var = tk.StringVar(value="offload=0 migrate=0")
        self.queue_var = tk.StringVar(value="-")

        top = tk.Frame(self.root, bg="#161a1d")
        top.pack(fill="x", padx=10, pady=10)

        self._make_card(
            parent=top,
            title="Step",
            text_var=self.step_var,
            tip=lambda: f"Current simulation step and configured limit.\nNow: {self._step}\nLimit: {self._steps_total}",
        ).pack(side="left", padx=4, fill="x", expand=True)
        self._make_card(
            parent=top,
            title="Vehicles",
            text_var=self.veh_var,
            tip=lambda: f"Live vehicles seen in this step: {self._vehicles}",
        ).pack(side="left", padx=4, fill="x", expand=True)
        self._make_card(
            parent=top,
            title="Active Tasks",
            text_var=self.task_var,
            tip=lambda: f"Active task count from TaskManager queue depths: {self._active_tasks}",
        ).pack(side="left", padx=4, fill="x", expand=True)
        self._make_card(
            parent=top,
            title="Cmds / Step",
            text_var=self.cmd_var,
            tip=self._tip_commands,
        ).pack(side="left", padx=4, fill="x", expand=True)
        self._make_card(
            parent=top,
            title="Scheduler",
            text_var=self.sched_var,
            tip=self._tip_scheduler,
        ).pack(side="left", padx=4, fill="x", expand=True)
        self._make_card(
            parent=top,
            title="Queue Depths",
            text_var=self.queue_var,
            tip=self._tip_queues,
        ).pack(side="left", padx=4, fill="x", expand=True)

        body = tk.Frame(self.root, bg="#161a1d")
        body.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        left = tk.LabelFrame(
            body,
            text="Fixed RSU / Vehicle Layout",
            bg="#0f1315",
            fg="#d9e1e8",
            font=("Segoe UI", 10, "bold"),
            labelanchor="n",
        )
        left.pack(side="left", fill="both", expand=True, padx=(0, 6))

        self.canvas = tk.Canvas(left, bg="#0b0f12", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=8, pady=8)
        self.canvas.bind("<Configure>", lambda _e: self._redraw_map())

        right = tk.Frame(body, bg="#161a1d")
        right.pack(side="left", fill="both", padx=(6, 0))

        sched_box_frame = tk.LabelFrame(
            right,
            text="Recent Scheduler Events",
            bg="#0f1315",
            fg="#d9e1e8",
            font=("Segoe UI", 10, "bold"),
        )
        sched_box_frame.pack(fill="both", expand=True, pady=(0, 6))
        self.sched_box = tk.Listbox(
            sched_box_frame,
            width=58,
            height=16,
            bg="#10171a",
            fg="#d2dde5",
            font=("Consolas", 9),
            selectbackground="#274c77",
        )
        self.sched_box.pack(fill="both", expand=True, padx=6, pady=6)

        cmd_box_frame = tk.LabelFrame(
            right,
            text="Recent Applied Commands",
            bg="#0f1315",
            fg="#d9e1e8",
            font=("Segoe UI", 10, "bold"),
        )
        cmd_box_frame.pack(fill="both", expand=True)
        self.cmd_box = tk.Listbox(
            cmd_box_frame,
            width=58,
            height=16,
            bg="#10171a",
            fg="#d2dde5",
            font=("Consolas", 9),
            selectbackground="#274c77",
        )
        self.cmd_box.pack(fill="both", expand=True, padx=6, pady=6)

    def _make_card(self, parent, title: str, text_var, tip: Callable[[], str]):
        tk = self.tk
        card = tk.Frame(parent, bg="#1f2a30", padx=8, pady=6, relief="ridge", bd=1)
        tk.Label(card, text=title, bg="#1f2a30", fg="#9fb3c8", font=("Segoe UI", 9, "bold")).pack(anchor="w")
        tk.Label(card, textvariable=text_var, bg="#1f2a30", fg="#f7fbff", font=("Consolas", 11, "bold")).pack(anchor="w")
        _HoverTip(tk, card, tip)
        return card

    def configure_rsus(self, rsu_nodes: list) -> None:
        if not self.enabled:
            return
        self._nodes = {
            n.rsu_id: _Node(rsu_id=n.rsu_id, x=float(n.x), y=float(n.y), node_id=int(n.node_id))
            for n in rsu_nodes
        }
        self._build_bounds()
        self._redraw_map()

    def begin_step(self, step: int, steps_total: int) -> None:
        if not self.enabled:
            return
        self._step = int(step)
        self._steps_total = int(steps_total)
        self._seen_this_step.clear()
        self._last_cmds = 0

    def update_vehicle(
        self,
        vehicle_id: str,
        position: tuple[float, float],
        observed_rsu: str,
        execution_rsu: str,
        speed: float,
        action: str,
        reroute: bool,
    ) -> None:
        if not self.enabled:
            return
        self._seen_this_step.add(vehicle_id)
        self._vehicle_state[vehicle_id] = {
            "position": (float(position[0]), float(position[1])),
            "observed_rsu": observed_rsu,
            "execution_rsu": execution_rsu,
            "speed": float(speed),
            "action": action,
            "reroute": bool(reroute),
        }
        self._draw_or_update_vehicle(vehicle_id)

    def finish_step(self, live_vehicle_ids: set[str]) -> None:
        if not self.enabled:
            return
        stale = [vid for vid in self._vehicle_items if vid not in live_vehicle_ids]
        for vid in stale:
            items = self._vehicle_items.pop(vid)
            for item in items.values():
                try:
                    self.canvas.delete(item)
                except Exception:
                    pass
            self._vehicle_state.pop(vid, None)

    def record_scheduler(self, vehicle_id: str, action: str, reason: str, src_rsu: str, exec_rsu: str) -> None:
        if not self.enabled:
            return
        self._sched_counts[action] += 1
        msg = f"veh={vehicle_id} action={action} reason={reason} obs={src_rsu} exec={exec_rsu}"
        self._sched_events.append(msg)
        self._render_lists()

    def record_command(self, cmd: dict) -> None:
        if not self.enabled:
            return
        action = str(cmd.get("action", "unknown"))
        reroute = bool(cmd.get("reroute", False))
        speed = float(cmd.get("speed", 0.0))
        vid = str(cmd.get("id", ""))
        rsu = str(cmd.get("source_rsu", "-"))
        reason = str(cmd.get("reason", ""))

        self._cmd_counts[action] += 1
        if reroute:
            self._cmd_counts["reroute"] += 1
        self._last_cmds += 1

        msg = f"veh={vid} action={action} speed={speed:.2f} reroute={reroute} rsu={rsu} reason={reason}"
        self._cmd_events.append(msg)
        self._render_lists()

    def set_snapshot(
        self,
        step: int,
        steps_total: int,
        vehicles: int,
        active_tasks: int,
        queue_depths: dict[str, int],
    ) -> None:
        if not self.enabled:
            return
        self._step = int(step)
        self._steps_total = int(steps_total)
        self._vehicles = int(vehicles)
        self._active_tasks = int(active_tasks)
        self._queue_depths = dict(sorted(queue_depths.items(), key=lambda x: x[0]))
        self._render()
        self._pump_events()

    def close(self) -> None:
        if not self.enabled:
            return
        self._on_close()

    def _on_close(self) -> None:
        self.available = False
        self.enabled = False
        try:
            self.root.destroy()
        except Exception:
            pass

    def _pump_events(self) -> None:
        if not self.available:
            return
        try:
            self.root.update_idletasks()
            self.root.update()
        except Exception:
            self.available = False
            self.enabled = False

    def _build_bounds(self) -> None:
        if not self._nodes:
            self._bounds = (0.0, 0.0, 1.0, 1.0)
            return
        xs = [n.x for n in self._nodes.values()]
        ys = [n.y for n in self._nodes.values()]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        pad_x = max(50.0, (max_x - min_x) * 0.08)
        pad_y = max(50.0, (max_y - min_y) * 0.08)
        self._bounds = (min_x - pad_x, min_y - pad_y, max_x + pad_x, max_y + pad_y)

    def _scale(self, x: float, y: float) -> tuple[float, float]:
        w = max(200.0, float(self.canvas.winfo_width()))
        h = max(200.0, float(self.canvas.winfo_height()))
        if self._bounds is None:
            self._build_bounds()
        min_x, min_y, max_x, max_y = self._bounds
        dx = max(1.0, max_x - min_x)
        dy = max(1.0, max_y - min_y)
        px = ((x - min_x) / dx) * (w - 24.0) + 12.0
        py = h - (((y - min_y) / dy) * (h - 24.0) + 12.0)
        return px, py

    def _redraw_map(self) -> None:
        if not self.enabled:
            return
        self.canvas.delete("rsu")
        self._node_items.clear()
        for node in sorted(self._nodes.values(), key=lambda x: x.rsu_id):
            cx, cy = self._scale(node.x, node.y)
            color = "#2d6a4f" if node.node_id == 1 else ("#3a5a40" if node.node_id == 2 else "#355070")
            oval = self.canvas.create_oval(cx - 12, cy - 12, cx + 12, cy + 12, fill=color, outline="#9ec5ab", width=2, tags=("rsu",))
            label = self.canvas.create_text(cx, cy - 18, text=node.rsu_id, fill="#d8f3dc", font=("Consolas", 9, "bold"), tags=("rsu",))
            queue = self.canvas.create_text(cx, cy + 18, text="q:0", fill="#b7c7d6", font=("Consolas", 8), tags=("rsu",))
            self._node_items[node.rsu_id] = {"oval": oval, "label": label, "queue": queue}

        for vid in list(self._vehicle_state.keys()):
            self._draw_or_update_vehicle(vid, redraw=True)
        self._update_queue_labels()

    def _draw_or_update_vehicle(self, vehicle_id: str, redraw: bool = False) -> None:
        if vehicle_id not in self._vehicle_state:
            return
        s = self._vehicle_state[vehicle_id]
        x, y = s["position"]
        cx, cy = self._scale(float(x), float(y))

        rsu_id = s["execution_rsu"] if s["execution_rsu"] in self._nodes else s["observed_rsu"]
        rsu_anchor = self._nodes.get(rsu_id)
        if rsu_anchor is not None:
            rx, ry = self._scale(rsu_anchor.x, rsu_anchor.y)
        else:
            rx, ry = cx, cy

        if s["action"] == "slow_down":
            color = "#ff6b6b"
            link_color = "#d1495b"
            link_dash = ()
        elif s["execution_rsu"] != s["observed_rsu"]:
            color = "#ffd166"
            link_color = "#f4a261"
            link_dash = (4, 2)
        else:
            color = "#4cc9f0"
            link_color = "#6c8ebf"
            link_dash = ()

        label_text = vehicle_id if self._show_labels else ""
        if len(label_text) > 14:
            label_text = label_text[:14]

        items = self._vehicle_items.get(vehicle_id)
        if items is None:
            line = self.canvas.create_line(cx, cy, rx, ry, fill=link_color, width=1, dash=link_dash)
            dot = self.canvas.create_oval(cx - 4, cy - 4, cx + 4, cy + 4, fill=color, outline="#e5e5e5", width=1.0)
            txt = self.canvas.create_text(cx, cy - 10, text=label_text, fill="#e8edf2", font=("Consolas", 7))
            self._vehicle_items[vehicle_id] = {"line": line, "dot": dot, "txt": txt}
        else:
            line, dot, txt = items["line"], items["dot"], items["txt"]
            self.canvas.coords(line, cx, cy, rx, ry)
            self.canvas.itemconfig(line, fill=link_color, dash=link_dash)
            self.canvas.coords(dot, cx - 4, cy - 4, cx + 4, cy + 4)
            self.canvas.itemconfig(dot, fill=color)
            self.canvas.coords(txt, cx, cy - 10)
            self.canvas.itemconfig(txt, text=label_text)

        if redraw:
            self.canvas.tag_lower(self._vehicle_items[vehicle_id]["line"])
            self.canvas.tag_raise(self._vehicle_items[vehicle_id]["dot"])
            self.canvas.tag_raise(self._vehicle_items[vehicle_id]["txt"])

    def _render(self) -> None:
        if not self.available:
            return
        self.step_var.set(f"{self._step} / {self._steps_total}")
        self.veh_var.set(str(self._vehicles))
        self.task_var.set(str(self._active_tasks))
        self.cmd_var.set(str(self._last_cmds))
        self.sched_var.set(
            f"offload={self._sched_counts.get('offload', 0)}  migrate={self._sched_counts.get('migrate', 0)}"
        )
        if self._queue_depths:
            top = "  ".join(f"{k}:{v}" for k, v in list(self._queue_depths.items())[:4])
        else:
            top = "-"
        self.queue_var.set(top)
        self._update_queue_labels()
        self._render_lists()

    def _render_lists(self) -> None:
        if not self.available:
            return
        self._set_listbox(self.sched_box, self._sched_events)
        self._set_listbox(self.cmd_box, self._cmd_events)

    def _update_queue_labels(self) -> None:
        if not self.available:
            return
        for rsu_id, items in self._node_items.items():
            depth = self._queue_depths.get(rsu_id, 0)
            self.canvas.itemconfig(items["queue"], text=f"q:{depth}")

    def _tip_commands(self) -> str:
        return (
            "Command counters\n"
            f"keep_speed: {self._cmd_counts.get('keep_speed', 0)}\n"
            f"slow_down: {self._cmd_counts.get('slow_down', 0)}\n"
            f"reroute: {self._cmd_counts.get('reroute', 0)}\n"
            f"last-step-applied: {self._last_cmds}"
        )

    def _tip_scheduler(self) -> str:
        return (
            "Scheduler counters\n"
            f"offload: {self._sched_counts.get('offload', 0)}\n"
            f"migrate: {self._sched_counts.get('migrate', 0)}\n"
            f"sticky: {self._sched_counts.get('sticky', 0)}"
        )

    def _tip_queues(self) -> str:
        if not self._queue_depths:
            return "No queue data yet."
        rows = [f"{k}: {v}" for k, v in self._queue_depths.items()]
        return "Current queue depths\n" + "\n".join(rows)

    @staticmethod
    def _set_listbox(listbox, rows: deque[str]) -> None:
        listbox.delete(0, "end")
        for item in rows:
            listbox.insert("end", item)
