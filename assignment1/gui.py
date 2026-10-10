#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Resizable tkinter game; AI processes keep the interface responsive.

Click intersections to play. Ctrl+N: new game; Ctrl+Z: undo; Esc: pause AI-AI.
The coordinate labels and exported records use the protocol's zero-based axes.
"""

import json
import math
import copy
from pathlib import Path
import sys
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from arena import AIProcess
from gomoku_game import GomokuGame, validate_move

BASE = Path(__file__).resolve().parent
BG, PANEL, TEXT, MUTED, ACCENT = "#10191f", "#1b2831", "#edf3f5", "#9aadb8", "#75dfb5"
MODES = {"人人对战": "human-human", "人机对战": "human-ai", "机机对战": "ai-ai"}


class GomokuApp:
    def __init__(self, root, args, on_home=None):
        self.root, self.args = root, args
        self.on_home = on_home
        root.title("AEGIS · 广义五子棋")
        root.minsize(900, 680)
        root.configure(bg=BG)
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.closed = False
        self.workers = {}
        self.ai_paths = {}
        self.records = []
        self.paused = False
        self.pause_after_move = False
        self.hover = None
        self.animation_tick = 0
        self.next_ai_at = 0.0
        self.game = GomokuGame(args.board, args.win)
        self.mode = tk.StringVar(value=next(label for label, mode in MODES.items() if mode == args.mode))
        self.n_var = tk.StringVar(value=str(args.board))
        self.k_var = tk.StringVar(value=str(args.win))
        self.time_var = tk.StringVar(value=str(args.time))
        self.human_var = tk.StringVar(value="黑棋先手" if args.human_color == "black" else "白棋后手")
        self.path_vars = {1: tk.StringVar(value=str(args.ai_black)),
                          2: tk.StringVar(value=str(args.ai_white or BASE / ("random_ai.py" if args.mode == "ai-ai" else "gomoku_ai.py")))}
        self.status = tk.StringVar(value="")
        self.detail = tk.StringVar(value="")
        self.black_info = tk.StringVar(value="黑棋 · X")
        self.white_info = tk.StringVar(value="白棋 · O")
        self.stats_var = tk.StringVar(value="")
        self._style()
        self._layout()
        root.bind("<Control-n>", lambda e: self.new_game())
        root.bind("<Control-z>", lambda e: self.undo())
        root.bind("<Escape>", lambda e: self.toggle_pause())
        self.new_game()
        self.tick_id = self.root.after(10, self._tick)

    def _style(self):
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("TNotebook", background=PANEL, borderwidth=0)
        style.configure("TNotebook.Tab", background="#263a46", foreground=MUTED, padding=(10, 8), font=("Microsoft YaHei UI", 10))
        style.map("TNotebook.Tab", background=[("selected", PANEL)], foreground=[("selected", ACCENT)])
        style.configure("TCombobox", fieldbackground=PANEL, background=PANEL,
                        foreground=TEXT, arrowcolor=ACCENT, bordercolor="#34465e", padding=6)
        style.map("TCombobox", fieldbackground=[("readonly", PANEL)], foreground=[("readonly", TEXT)])
        style.configure("TSpinbox", fieldbackground=PANEL, background=PANEL,
                        foreground=TEXT, arrowcolor=ACCENT, bordercolor="#34465e", padding=5)
        self.root.option_add("*TCombobox*Listbox.background", PANEL)
        self.root.option_add("*TCombobox*Listbox.foreground", TEXT)
        self.root.option_add("*TCombobox*Listbox.selectBackground", "#315c62")

    def _label(self, parent, text="", variable=None, size=11, color=TEXT, bold=False):
        return tk.Label(parent, text=text, textvariable=variable, bg=parent["bg"], fg=color,
                        font=("Microsoft YaHei UI", size, "bold" if bold else "normal"), anchor="w")

    def _button(self, parent, text, command, primary=False):
        return tk.Button(parent, text=text, command=command, bg=ACCENT if primary else "#2a3a50",
                         fg=BG if primary else TEXT, activebackground="#7ae9d2" if primary else "#3a506c",
                         activeforeground=BG if primary else TEXT, relief="flat", bd=0, cursor="hand2",
                         padx=12, pady=9, font=("Microsoft YaHei UI", 10, "bold"))

    def _layout(self):
        header = tk.Frame(self.root, bg=BG)
        header.pack(fill="x", padx=26, pady=(20, 14))
        self._label(header, "AEGIS", size=23, bold=True).pack(side="left")
        self._label(header, "广义五子棋", size=11, color=MUTED).pack(side="left", padx=16, pady=(6, 0))
        self._button(header, "返回模式选择", self.return_home).pack(side="right")
        body = tk.Frame(self.root, bg=BG)
        body.pack(fill="both", expand=True, padx=24, pady=(0, 20))
        sidebar = tk.Frame(body, bg=PANEL, width=284)
        sidebar.pack(side="right", fill="y", padx=(20, 0))
        sidebar.pack_propagate(False)
        self._label(sidebar, "对局控制", size=13, bold=True).pack(fill="x", padx=18, pady=(16, 12))
        tabs = ttk.Notebook(sidebar)
        tabs.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        settings = tk.Frame(tabs, bg=PANEL)
        record_tab = tk.Frame(tabs, bg=PANEL)
        tabs.add(settings, text="  设置  ")
        tabs.add(record_tab, text="  棋谱  ")
        fields = tk.Frame(settings, bg=PANEL)
        fields.pack(fill="x", padx=12, pady=12)
        self._label(fields, "对战模式", color=MUTED, size=9).pack(fill="x")
        mode_box = ttk.Combobox(fields, textvariable=self.mode, values=list(MODES), state="readonly")
        mode_box.pack(fill="x", pady=(4, 10))
        mode_box.bind("<<ComboboxSelected>>", lambda e: self._mode_fields())
        row = tk.Frame(fields, bg=PANEL)
        row.pack(fill="x")
        for text, variable, start, end in (("棋盘边长 N", self.n_var, 5, 15), ("获胜连珠 K", self.k_var, 3, 15)):
            col = tk.Frame(row, bg=PANEL)
            col.pack(side="left", fill="x", expand=True, padx=(0, 6))
            self._label(col, text, color=MUTED, size=9).pack(fill="x")
            ttk.Spinbox(col, from_=start, to=end, textvariable=variable, width=7).pack(fill="x", pady=(4, 10))
        option_group = tk.Frame(fields, bg=PANEL)
        option_group.pack(fill="x")
        self.human_row = tk.Frame(option_group, bg=PANEL)
        self._label(self.human_row, "我的执棋", color=MUTED, size=9).pack(fill="x")
        ttk.Combobox(self.human_row, textvariable=self.human_var, values=("黑棋先手", "白棋后手"), state="readonly").pack(fill="x", pady=(4, 10))
        self.ai_fields = tk.Frame(option_group, bg=PANEL)
        self._label(self.ai_fields, "AI 每步思考上限 / 秒", color=MUTED, size=9).pack(fill="x")
        ttk.Spinbox(self.ai_fields, from_=0.1, to=60, increment=0.5, textvariable=self.time_var).pack(fill="x", pady=(4, 10))
        for side in (1, 2):
            self._label(self.ai_fields, "黑棋 AI" if side == 1 else "白棋 AI", color=MUTED, size=9).pack(fill="x")
            entry_row = tk.Frame(self.ai_fields, bg=PANEL)
            entry_row.pack(fill="x", pady=(4, 8))
            var = tk.StringVar(value=self._ai_label(self.path_vars[side].get()))
            box = ttk.Combobox(entry_row, textvariable=var, values=("AEGIS 3", "随机 AI", "自定义 AI…"), state="readonly", width=12)
            box.pack(side="left", fill="x", expand=True)
            box.bind("<<ComboboxSelected>>", lambda e, p=side, v=var: self._select_ai(p, v))
            self._button(entry_row, "浏览", lambda p=side, v=var: self._browse_ai(p, v)).pack(side="right", padx=(5, 0))
        self._mode_fields()
        self._button(fields, "应用设置 · 开始新局", self.new_game, primary=True).pack(fill="x", pady=(12, 8))
        actions = tk.Frame(fields, bg=PANEL)
        actions.pack(fill="x")
        self.undo_button = self._button(actions, "悔棋", self.undo)
        self.undo_button.pack(side="left", expand=True, fill="x", padx=(0, 4))
        self.pause_button = self._button(actions, "暂停", self.toggle_pause)
        self.pause_button.pack(side="left", expand=True, fill="x", padx=(4, 0))
        self._label(fields, "设置将在新局生效", size=9, color=MUTED).pack(fill="x", pady=(10, 0))
        self._label(fields, "Ctrl+N 重开  ·  Ctrl+Z 悔棋", size=8, color=MUTED).pack(fill="x", pady=(4, 0))
        self._label(record_tab, "落子坐标从 0 开始编号", size=9, color=MUTED).pack(fill="x", padx=12, pady=12)
        history_frame = tk.Frame(record_tab, bg=PANEL)
        history_frame.pack(fill="both", expand=True, padx=12)
        self.history_box = tk.Text(history_frame, bg="#111e27", fg=TEXT, relief="flat", bd=0,
                                   font=("Consolas", 10), width=18, state="disabled")
        self.history_box.pack(side="left", fill="both", expand=True)
        scrollbar = ttk.Scrollbar(history_frame, command=self.history_box.yview)
        scrollbar.pack(side="right", fill="y")
        self.history_box.configure(yscrollcommand=scrollbar.set)
        self._button(record_tab, "保存棋谱", self.save_record).pack(fill="x", padx=12, pady=12)
        play_area = tk.Frame(body, bg=BG)
        play_area.pack(side="left", fill="both", expand=True)
        self._label(play_area, variable=self.status, size=17, bold=True).pack(fill="x", padx=6, pady=(2, 5))
        self._label(play_area, variable=self.detail, size=9, color=MUTED).pack(fill="x", padx=6, pady=(0, 12))
        self.canvas = tk.Canvas(play_area, bg=BG, highlightthickness=0, cursor="crosshair")
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda e: self.draw())
        self.canvas.bind("<Button-1>", self.click)
        self.canvas.bind("<Motion>", self.motion)
        self.canvas.bind("<Leave>", lambda e: self._set_hover(None))
        players = tk.Frame(play_area, bg=BG)
        players.pack(fill="x", padx=6, pady=(10, 4))
        self._label(players, variable=self.black_info, size=10).pack(side="left")
        self._label(players, variable=self.white_info, size=10).pack(side="right")
        self._label(play_area, variable=self.stats_var, size=9, color=ACCENT).pack(fill="x", padx=6)

    def _mode_fields(self):
        self.human_row.pack_forget()
        self.ai_fields.pack_forget()
        if self.mode.get() == "人机对战":
            self.human_row.pack(fill="x")
        if self.mode.get() != "人人对战":
            self.ai_fields.pack(fill="x")

    @staticmethod
    def _ai_label(path):
        name = Path(path).name
        return {"gomoku_ai.py": "AEGIS 3", "random_ai.py": "随机 AI"}.get(name, "自定义 AI…")

    def _select_ai(self, side, variable):
        if variable.get() == "自定义 AI…":
            self._browse_ai(side, variable)
        else:
            self.path_vars[side].set(str(BASE / ("random_ai.py" if variable.get() == "随机 AI" else "gomoku_ai.py")))

    def _browse_ai(self, side, variable):
        self.choose_ai(side)
        variable.set(self._ai_label(self.path_vars[side].get()))

    def return_home(self):
        self.closed = True
        self._close_workers()
        self.root.after_cancel(self.tick_id)
        for key in ("<Control-n>", "<Control-z>", "<Escape>"):
            self.root.unbind(key)
        for widget in self.root.winfo_children():
            widget.destroy()
        if self.on_home:
            self.on_home()

    def choose_ai(self, side):
        path = filedialog.askopenfilename(title="选择符合 GomokuAI 协议的 Python 文件",
                                           filetypes=[("Python AI", "*.py")], initialdir=BASE)
        if path:
            self.path_vars[side].set(path)

    def _close_workers(self):
        for worker in self.workers.values():
            worker.close()
        self.workers.clear()

    def _start_workers(self):
        self._close_workers()
        for side, path in self.ai_paths.items():
            try:
                self.workers[side] = AIProcess(path, side, self.game.board_size,
                                               self.game.win_length, self.args.seed + side)
            except Exception as exc:
                self.game.forfeit(side, "AI 启动失败: {}".format(exc))
                break

    def new_game(self):
        try:
            n, k, seconds = int(self.n_var.get()), int(self.k_var.get()), float(self.time_var.get())
            if not 5 <= n <= 15 or not 3 <= k <= n:
                raise ValueError("棋盘边长需为 5 至 15，且 3 ≤ K ≤ N")
            if not math.isfinite(seconds) or not 0.05 <= seconds <= 60:
                raise ValueError("AI 时限需为 0.05 至 60 秒")
            mode = MODES[self.mode.get()]
            human = 1 if self.human_var.get() == "黑棋先手" else 2
            sides = () if mode == "human-human" else (3 - human,) if mode == "human-ai" else (1, 2)
            paths = {side: Path(self.path_vars[side].get()).expanduser().resolve() for side in sides}
            for path in paths.values():
                if not path.is_file() or path.suffix.lower() != ".py":
                    raise ValueError("AI 文件不存在或不是 .py: {}".format(path))
        except (ValueError, KeyError) as exc:
            messagebox.showerror("设置无效", str(exc))
            return
        self.active_mode, self.seconds, self.ai_paths = mode, seconds, paths
        self.args.board, self.args.win, self.args.time = n, k, seconds
        self.args.mode = mode
        self.args.human_color = "black" if human == 1 else "white"
        self.args.ai_black = Path(self.path_vars[1].get())
        self.args.ai_white = Path(self.path_vars[2].get())
        self.game = GomokuGame(n, k)
        self.records, self.hover = [], None
        self.paused = self.pause_after_move = False
        self.pause_button.configure(text="暂停", state="normal" if mode == "ai-ai" else "disabled")
        self.next_ai_at = 0.0
        self.stats_var.set("绿色圆点标记最后一手 · 横、竖或斜线连珠均可获胜")
        self._start_workers()
        self._refresh_history()
        self._refresh_status()
        self.draw()

    def _geometry(self):
        width, height = self.canvas.winfo_width(), self.canvas.winfo_height()
        size = max(40, min(width - 8, height - 8))
        x0, y0 = (width - size) / 2, (height - size) / 2
        margin = max(23, size * 0.055)
        spacing = (size - margin * 2) / (self.game.board_size - 1)
        return x0, y0, size, margin, spacing

    def _point(self, row, col):
        x0, y0, _, margin, spacing = self._geometry()
        return x0 + margin + col * spacing, y0 + margin + row * spacing

    def _cell_at(self, event):
        x0, y0, _, margin, spacing = self._geometry()
        if spacing <= 0:
            return None
        col = int(math.floor((event.x - x0 - margin) / spacing + 0.5))
        row = int(math.floor((event.y - y0 - margin) / spacing + 0.5))
        if not (0 <= row < self.game.board_size and 0 <= col < self.game.board_size):
            return None
        x, y = self._point(row, col)
        if abs(x - event.x) > spacing * 0.46 or abs(y - event.y) > spacing * 0.46:
            return None
        return row, col

    def _can_play(self):
        return not self.game.finished and self.game.to_move not in self.ai_paths

    def click(self, event):
        if not self._can_play():
            return
        cell = self._cell_at(event)
        if cell is None or validate_move(cell, self.game.board):
            return
        side = self.game.to_move
        self.game.play(cell)
        self.records.append({"player": side, "move": cell, "elapsed": 0.0})
        self.hover = None
        self._refresh_history()
        self._refresh_status()
        self.draw()

    def _set_hover(self, cell):
        if cell != self.hover:
            self.hover = cell
            self._draw_hover()

    def motion(self, event):
        cell = self._cell_at(event) if self._can_play() else None
        if cell and self.game.board[cell[0]][cell[1]]:
            cell = None
        self._set_hover(cell)

    def _draw_hover(self):
        self.canvas.delete("hover")
        if self.hover and self._can_play():
            x, y = self._point(*self.hover)
            radius = self._geometry()[4] * 0.37
            color = "#27313e" if self.game.to_move == 1 else "#fff8ed"
            self.canvas.create_oval(x - radius, y - radius, x + radius, y + radius,
                                    outline=color, width=2, dash=(3, 3), tags="hover")

    def draw(self):
        cv = self.canvas
        cv.delete("all")
        x0, y0, size, margin, spacing = self._geometry()
        n = self.game.board_size
        # Quiet vertical wood gradient, drawn entirely with standard tkinter.
        for i in range(40):
            fraction = i / 39
            color = "#{:02x}{:02x}{:02x}".format(int(225 - 13 * fraction), int(195 - 22 * fraction), int(150 - 25 * fraction))
            cv.create_rectangle(x0, y0 + size * i / 40, x0 + size,
                                 y0 + size * (i + 1) / 40 + 1, fill=color, outline=color)
        startx, starty = x0 + margin, y0 + margin
        endx, endy = x0 + size - margin, y0 + size - margin
        for index in range(n):
            x, y = startx + index * spacing, starty + index * spacing
            cv.create_line(x, starty, x, endy, fill="#796546", width=1)
            cv.create_line(startx, y, endx, y, fill="#796546", width=1)
            if n <= 25 or index % 2 == 0:
                font_size = max(6, min(10, int(spacing * 0.27)))
                cv.create_text(x, y0 + margin * 0.40, text=str(index), fill="#665338", font=("Segoe UI", font_size))
                cv.create_text(x0 + margin * 0.40, y, text=str(index), fill="#665338", font=("Segoe UI", font_size))
        stars = {(n // 2, n // 2)} if n % 2 else set()
        if n >= 11:
            a, b = max(2, n // 4), n - max(2, n // 4) - 1
            stars |= {(a, a), (a, b), (b, a), (b, b)}
        for row, col in stars:
            x, y = self._point(row, col)
            radius = max(2, spacing * 0.07)
            cv.create_oval(x - radius, y - radius, x + radius, y + radius, fill="#725c3f", outline="")
        radius = spacing * 0.435
        for row in range(n):
            for col in range(n):
                side = self.game.board[row][col]
                if not side:
                    continue
                x, y = self._point(row, col)
                cv.create_oval(x - radius + 1.8, y - radius + 2.5, x + radius + 1.8,
                                y + radius + 2.5, fill="#947957", outline="")
                base = "#111722" if side == 1 else "#eee9e0"
                edge = "#0b111b" if side == 1 else "#c4bbaa"
                cv.create_oval(x - radius, y - radius, x + radius, y + radius, fill=base, outline=edge, width=1)
                # Small offset discs give the stones soft, readable highlights.
                colors = ("#202a3a", "#283448", "#324159") if side == 1 else ("#f4f0e8", "#faf7f0", "#fffdf8")
                for j, color in enumerate(colors):
                    r = radius * (0.72 - 0.15 * j)
                    dx, dy = -radius * 0.13, -radius * 0.18
                    cv.create_oval(x + dx - r, y + dy - r, x + dx + r,
                                    y + dy + r, fill=color, outline="")
        if self.game.win_line:
            first, last = self.game.win_line[0], self.game.win_line[-1]
            x1, y1 = self._point(*first)
            x2, y2 = self._point(*last)
            cv.create_line(x1, y1, x2, y2, fill="#f3a936", width=max(3, spacing * 0.08), capstyle="round")
        last = self.game.last_move()
        if last:
            x, y = self._point(*last)
            r = max(2.5, spacing * 0.09)
            cv.create_oval(x - r, y - r, x + r, y + r, fill="#12a88b", outline="#88ffe3", width=1)
        self._draw_hover()

    def _refresh_history(self):
        self.history_box.configure(state="normal")
        self.history_box.delete("1.0", "end")
        for index, move in enumerate(self.game.history, 1):
            line = "{:3d} {} ({:2d},{:2d})".format(index, "黑" if move.player == 1 else "白", move.row, move.col)
            if move.elapsed:
                line += " {:.2f}s".format(move.elapsed)
            self.history_box.insert("end", line + "\n")
        self.history_box.see("end")
        self.history_box.configure(state="disabled")

    def _refresh_status(self):
        game = self.game
        if game.finished:
            self.status.set("和棋" if game.draw else "{}获胜".format("黑棋" if game.winner == 1 else "白棋"))
            self.detail.set("{} · 共 {} 手".format(self._reason_text(game.reason), len(game.history)))
        elif self.paused:
            self.status.set("对局已暂停")
            self.detail.set("点击继续，或按 Esc 恢复")
        else:
            side = game.to_move
            thinking = side in self.ai_paths
            self.status.set("{}{}".format("黑棋" if side == 1 else "白棋", "正在思考" if thinking else "落子"))
            self.detail.set("第 {} 手 · {} × {} 棋盘 · {} 连珠获胜 · {}".format(
                len(game.history) + 1, game.board_size, game.board_size, game.win_length,
                "AI 时限 {:.2f} 秒".format(self.seconds) if thinking else "点击交叉点落子"))
        for side, variable in ((1, self.black_info), (2, self.white_info)):
            worker = self.workers.get(side)
            variable.set("{} · {}".format("黑棋 X" if side == 1 else "白棋 O", worker.name if worker else "玩家"))

    @staticmethod
    def _reason_text(reason):
        if reason.endswith("-in-a-row"):
            return reason.split("-", 1)[0] + " 连珠达成"
        return {"board full": "棋盘已满", "timeout": "AI 超时，判负", "initialization timeout": "AI 启动超时"}.get(reason, reason)

    def undo(self):
        if not self.game.history:
            return
        self._close_workers()
        count = 1
        if self.active_mode == "human-ai" and self.game.history[-1].player in self.ai_paths:
            count = 2
        self.game.undo(count)
        self.records = self.records[:len(self.game.history)]
        if self.active_mode == "ai-ai":
            self.paused = True
            self.pause_button.configure(text="继续")
        self.pause_after_move = False
        self.hover = None
        self._start_workers()
        self._refresh_history()
        self._refresh_status()
        self.draw()

    def toggle_pause(self):
        if self.active_mode != "ai-ai" or self.game.finished:
            return
        if any(worker.pending for worker in self.workers.values()):
            self.pause_after_move = not self.pause_after_move
            self.pause_button.configure(text="已预约暂停" if self.pause_after_move else "暂停")
            return
        self.paused = not self.paused
        self.pause_button.configure(text="继续" if self.paused else "暂停")
        self._refresh_status()

    def save_record(self):
        path = filedialog.asksaveasfilename(title="保存棋谱", defaultextension=".json",
                                            initialfile="gomoku_record.json", filetypes=[("JSON 棋谱", "*.json")])
        if not path:
            return
        data = {"board_size": self.game.board_size, "win_length": self.game.win_length,
                "time_limit": self.seconds, "mode": self.active_mode,
                "winner": self.game.winner, "reason": self.game.reason, "records": self.records}
        try:
            Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError as exc:
            messagebox.showerror("保存失败", str(exc))

    def _tick(self):
        if self.closed:
            return
        self.animation_tick += 1
        if not self.game.finished:
            ready = True
            for side, worker in self.workers.items():
                is_ready, error = worker.poll_ready()
                if error:
                    self.game.forfeit(side, error)
                    self._refresh_status()
                    break
                ready = ready and is_ready
            if ready and not self.game.finished:
                side = self.game.to_move
                worker = self.workers.get(side)
                if worker and worker.pending:
                    result = worker.poll()
                    if result:
                        move, elapsed, error, stats = result
                        error = error or validate_move(move, self.game.board)
                        if error:
                            self.game.forfeit(side, error)
                        else:
                            self.game.play(move, elapsed)
                            self.records.append({"player": side, "move": move, "elapsed": elapsed, "stats": stats})
                            self.stats_var.set("AI 用时 {:.3f}s  ·  搜索深度 {}  ·  节点 {:,}  ·  {}".format(
                                elapsed, stats.get("depth", "—"), stats.get("nodes", 0), worker.name))
                            self._refresh_history()
                            self.draw()
                        self.next_ai_at = time.perf_counter() + 0.25
                        if self.pause_after_move:
                            self.paused, self.pause_after_move = True, False
                            self.pause_button.configure(text="继续")
                        self._refresh_status()
                elif worker and not self.paused and time.perf_counter() >= self.next_ai_at:
                    try:
                        worker.request(self.game.board, self.game.last_move(3 - side), self.seconds)
                    except Exception as exc:
                        self.game.forfeit(side, "AI 通信失败: {}".format(exc))
                    self.hover = None
                    self._draw_hover()
                    self._refresh_status()
        if self.game.finished and self.workers:
            self._close_workers()
        self.tick_id = self.root.after(15, self._tick)

    def close(self):
        self.closed = True
        self._close_workers()
        self.root.destroy()


class HomeScreen:
    """Mode selection and match setup, sharing one native window with the game."""

    def __init__(self, root, args):
        self.root, self.args = root, args
        self.frame = tk.Frame(root, bg=BG)
        self.frame.pack(fill="both", expand=True)
        root.title("AEGIS 五子棋 · 选择玩法")
        root.protocol("WM_DELETE_WINDOW", root.destroy)
        self.mode = tk.StringVar(value=args.mode)
        self.n = tk.StringVar(value=str(args.board))
        self.k = tk.StringVar(value=str(args.win))
        self.seconds = tk.StringVar(value=str(args.time))
        self.human = tk.StringVar(value="黑棋先手" if args.human_color == "black" else "白棋后手")
        self.paths = {1: Path(args.ai_black), 2: Path(args.ai_white or BASE / "gomoku_ai.py")}
        if args.mode == "ai-ai" and args.ai_white is None:
            self.paths[2] = BASE / "random_ai.py"
        self.cards = {}
        self.ai_vars = {}
        self._style()
        self._layout()
        self._select_mode(args.mode)
        self.n.trace_add("write", self._resize_options)
        root.bind("<Return>", lambda e: self.start())

    def _style(self):
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("TCombobox", fieldbackground="#233540", background="#233540", foreground=TEXT,
                        arrowcolor=ACCENT, bordercolor="#344955", padding=7, font=("Microsoft YaHei UI", 10))
        style.map("TCombobox", fieldbackground=[("readonly", "#233540")], foreground=[("readonly", TEXT)])
        style.configure("TSpinbox", fieldbackground="#233540", background="#233540", foreground=TEXT,
                        arrowcolor=ACCENT, bordercolor="#344955", padding=7, font=("Microsoft YaHei UI", 10))
        self.root.option_add("*TCombobox*Listbox.background", PANEL)
        self.root.option_add("*TCombobox*Listbox.foreground", TEXT)
        self.root.option_add("*TCombobox*Listbox.selectBackground", "#356858")

    @staticmethod
    def label(parent, text, size=10, color=TEXT, bold=False, **kwargs):
        return tk.Label(parent, text=text, bg=parent["bg"], fg=color,
                        font=("Microsoft YaHei UI", size, "bold" if bold else "normal"), anchor="w", **kwargs)

    @staticmethod
    def button(parent, text, command, primary=False):
        return tk.Button(parent, text=text, command=command, bg=ACCENT if primary else "#293e49",
                         fg=BG if primary else TEXT, activebackground="#98eacd" if primary else "#3b535e",
                         activeforeground=BG if primary else TEXT, relief="flat", bd=0, cursor="hand2",
                         padx=14, pady=10, font=("Microsoft YaHei UI", 10, "bold"))

    def _layout(self):
        header = tk.Frame(self.frame, bg=BG)
        header.pack(fill="x", padx=34, pady=(25, 20))
        self.label(header, "AEGIS", 24, bold=True).pack(side="left")
        self.label(header, "广义五子棋", 11, MUTED).pack(side="left", padx=16, pady=(8, 0))
        self.label(header, "离线运行  /  本地对局", 9, ACCENT).pack(side="right", pady=(8, 0))
        hero = tk.Frame(self.frame, bg=BG)
        hero.pack(fill="x", padx=34, pady=(2, 20))
        self.label(hero, "先选一种玩法。", 27, bold=True).pack(fill="x")
        self.label(hero, "和朋友过招，挑战 AEGIS，或观看两个 AI 的较量。", 11, MUTED).pack(fill="x", pady=(9, 0))
        cards = tk.Frame(self.frame, bg=BG)
        cards.pack(fill="x", padx=34)
        choices = (("human-human", "01  /  LOCAL", "人人对战", "两位玩家轮流落子", "同屏对弈 · 黑棋先手"),
                   ("human-ai", "02  /  CHALLENGE", "人机对战", "与 AEGIS AI 过招", "自由选择执黑或执白"),
                   ("ai-ai", "03  /  SPECTATE", "机机对战", "观看 AI 自动对弈", "可暂停 · 可加载自定义 AI"))
        for i, (mode, eyebrow, title, description, note) in enumerate(choices):
            cards.columnconfigure(i, weight=1, uniform="mode")
            card = tk.Frame(cards, bg=PANEL, highlightthickness=2, highlightbackground="#2b3f4a", cursor="hand2")
            card.grid(row=0, column=i, sticky="nsew", padx=(0 if i == 0 else 8, 0 if i == 2 else 8))
            inner = tk.Frame(card, bg=PANEL, cursor="hand2")
            inner.pack(fill="both", expand=True, padx=18, pady=17)
            self.label(inner, eyebrow, 8, ACCENT).pack(fill="x")
            self.label(inner, title, 18, bold=True).pack(fill="x", pady=(12, 7))
            self.label(inner, description, 10, TEXT).pack(fill="x")
            self.label(inner, note, 9, MUTED).pack(fill="x", pady=(6, 12))
            selected = self.label(inner, "选择此模式", 9, MUTED)
            selected.pack(fill="x")
            self.cards[mode] = (card, selected)
            self._bind_card(card, mode)
        panels = tk.Frame(self.frame, bg=BG)
        panels.pack(fill="both", expand=True, padx=34, pady=(22, 18))
        panels.columnconfigure(0, weight=1, uniform="settings")
        panels.columnconfigure(1, weight=1, uniform="settings")
        panels.rowconfigure(0, weight=1)
        board = tk.Frame(panels, bg=PANEL)
        board.grid(row=0, column=0, sticky="nsew", padx=(0, 9))
        match = tk.Frame(panels, bg=PANEL)
        match.grid(row=0, column=1, sticky="nsew", padx=(9, 0))
        b = tk.Frame(board, bg=PANEL)
        b.pack(fill="both", expand=True, padx=20, pady=18)
        self.label(b, "棋盘与规则", 13, bold=True).pack(fill="x")
        presets = tk.Frame(b, bg=PANEL)
        presets.pack(fill="x", pady=(14, 12))
        for label, n, k in (("小棋盘 5 / 3", 5, 3), ("快局 9 / 4", 9, 4), ("标准 15 / 5", 15, 5)):
            self.button(presets, label, lambda a=n, z=k: self._preset(a, z)).pack(side="left", expand=True, fill="x", padx=(0, 5))
        values = tk.Frame(b, bg=PANEL)
        values.pack(fill="x")
        for text, var, low, high in (("棋盘边长 N", self.n, 5, 15), ("获胜连珠 K", self.k, 3, 15)):
            col = tk.Frame(values, bg=PANEL)
            col.pack(side="left", fill="x", expand=True, padx=(0, 10))
            self.label(col, text, 9, MUTED).pack(fill="x")
            spin = ttk.Spinbox(col, from_=low, to=high, textvariable=var, width=8)
            spin.pack(fill="x", pady=(5, 0))
            if var is self.k:
                self.k_spin = spin
        self.label(b, "横、竖或斜线连成 K 子及以上即获胜。", 9, MUTED).pack(fill="x", pady=(16, 4))
        self.label(b, "黑棋先行 · 无禁手 · 棋盘下满则和棋", 9, MUTED).pack(fill="x")
        m = tk.Frame(match, bg=PANEL)
        m.pack(fill="both", expand=True, padx=20, pady=18)
        self.label(m, "对局选项", 13, bold=True).pack(fill="x", pady=(0, 14))
        self.human_options = tk.Frame(m, bg=PANEL)
        self.label(self.human_options, "我的执棋", 9, MUTED).pack(side="left", padx=(0, 14))
        ttk.Combobox(self.human_options, textvariable=self.human, values=("黑棋先手", "白棋后手"), state="readonly", width=12).pack(side="left", fill="x", expand=True)
        self.ai_options = tk.Frame(m, bg=PANEL)
        for side in (1, 2):
            row = tk.Frame(self.ai_options, bg=PANEL)
            row.pack(fill="x", pady=(0, 7))
            self.label(row, "黑棋 AI" if side == 1 else "白棋 AI", 9, MUTED, width=8).pack(side="left")
            var = tk.StringVar(value=GomokuApp._ai_label(self.paths[side]))
            self.ai_vars[side] = var
            box = ttk.Combobox(row, textvariable=var, values=("AEGIS 3", "随机 AI", "自定义 AI…"), state="readonly", width=14)
            box.pack(side="left", fill="x", expand=True)
            box.bind("<<ComboboxSelected>>", lambda e, s=side: self._choose_ai(s))
        row = tk.Frame(self.ai_options, bg=PANEL)
        row.pack(fill="x")
        self.label(row, "思考上限 / 秒", 9, MUTED, width=13).pack(side="left")
        ttk.Spinbox(row, from_=0.1, to=60, increment=0.5, textvariable=self.seconds, width=8).pack(side="left", fill="x", expand=True)
        self.mode_note = self.label(m, "", 9, MUTED, wraplength=390, justify="left")
        self.mode_note.pack(fill="x", side="bottom", pady=(12, 0))
        footer = tk.Frame(self.frame, bg=BG)
        footer.pack(fill="x", padx=34, pady=(0, 26))
        self.label(footer, "落子、悔棋、保存棋谱，都在棋盘界面。", 9, MUTED).pack(side="left")
        self.start_button = self.button(footer, "开始对局  →", self.start, primary=True)
        self.start_button.configure(width=21, pady=13)
        self.start_button.pack(side="right")

    def _bind_card(self, widget, mode):
        widget.bind("<Button-1>", lambda e: self._select_mode(mode))
        for child in widget.winfo_children():
            self._bind_card(child, mode)

    def _select_mode(self, mode):
        self.mode.set(mode)
        for key, (card, label) in self.cards.items():
            card.configure(highlightbackground=ACCENT if key == mode else "#2b3f4a")
            label.configure(text="已选择  ✓" if key == mode else "选择此模式", fg=ACCENT if key == mode else MUTED)
        self.human_options.pack_forget()
        self.ai_options.pack_forget()
        if mode == "human-ai":
            self.human_options.pack(fill="x", pady=(0, 10))
        if mode != "human-human":
            self.ai_options.pack(fill="x")
        notes = {"human-human": "两位玩家共用一块棋盘。轮到谁，谁点击交叉点落子。",
                 "human-ai": "只会启用与你执棋相反的一方 AI。思考时间越长，AI 搜索通常越充分。",
                 "ai-ai": "两个 AI 自动交替落子。进入棋盘后，可随时预约暂停或继续。"}
        self.mode_note.configure(text=notes[mode])

    def _preset(self, n, k):
        self.n.set(str(n))
        self.k.set(str(k))

    def _resize_options(self, *_):
        try:
            n = int(self.n.get())
            self.k_spin.configure(to=n)
            if int(self.k.get()) > n >= 3:
                self.k.set(str(n))
        except ValueError:
            pass

    def _choose_ai(self, side):
        choice = self.ai_vars[side].get()
        if choice == "自定义 AI…":
            path = filedialog.askopenfilename(title="选择实现 GomokuAI 协议的 AI 文件", filetypes=[("Python AI", "*.py")])
            if path:
                self.paths[side] = Path(path)
            self.ai_vars[side].set(GomokuApp._ai_label(self.paths[side]))
        else:
            self.paths[side] = BASE / ("random_ai.py" if choice == "随机 AI" else "gomoku_ai.py")

    def start(self):
        try:
            n, k, seconds = int(self.n.get()), int(self.k.get()), float(self.seconds.get())
            if not 5 <= n <= 15 or not 3 <= k <= n:
                raise ValueError("棋盘边长 N 需为 5 至 15，连珠数需满足 3 ≤ K ≤ N。")
            if not math.isfinite(seconds) or not 0.05 <= seconds <= 60:
                raise ValueError("AI 思考上限需为 0.05 至 60 秒。")
            mode = self.mode.get()
            human = 1 if self.human.get() == "黑棋先手" else 2
            sides = () if mode == "human-human" else (3 - human,) if mode == "human-ai" else (1, 2)
            for side in sides:
                if not self.paths[side].is_file():
                    raise ValueError("AI 文件不存在，请重新选择。")
        except ValueError as exc:
            messagebox.showerror("请检查对局设置", str(exc), parent=self.root)
            return
        args = copy.copy(self.args)
        args.board, args.win, args.time, args.mode = n, k, seconds, mode
        args.human_color = "black" if human == 1 else "white"
        args.ai_black, args.ai_white = self.paths[1], self.paths[2]
        self.root.unbind("<Return>")
        self.frame.destroy()
        self.app = GomokuApp(self.root, args, on_home=lambda: HomeScreen(self.root, args))


def create_root():
    # Initialize the frozen Tcl filesystem before Tk loads bundled resources.
    # Querying its working directory primes path resolution in this runtime.
    if sys.platform == "win32" and getattr(sys, "frozen", False):
        import ctypes
        import os
        for name in ("TCL_LIBRARY", "TK_LIBRARY"):
            if name in os.environ:
                os.environ[name] = Path(os.environ[name]).as_posix()
        tcl = ctypes.CDLL(str(Path(sys._MEIPASS) / "tcl86t.dll"))
        tcl.Tcl_FindExecutable.argtypes = [ctypes.c_char_p]
        tcl.Tcl_FindExecutable.restype = None
        tcl.Tcl_FindExecutable(sys.executable.encode("utf-8"))
        tcl.Tcl_CreateInterp.restype = ctypes.c_void_p
        tcl.Tcl_Eval.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        tcl.Tcl_Init.argtypes = [ctypes.c_void_p]
        tcl.Tcl_GetStringResult.argtypes = [ctypes.c_void_p]
        tcl.Tcl_GetStringResult.restype = ctypes.c_char_p
        tcl.Tcl_DeleteInterp.argtypes = [ctypes.c_void_p]
        tcl.Tcl_DeleteInterp.restype = None
        interp = tcl.Tcl_CreateInterp()
        try:
            tcl.Tcl_Eval(interp, b"pwd")
            if tcl.Tcl_Init(interp):
                raise tk.TclError(tcl.Tcl_GetStringResult(interp).decode("utf-8", errors="replace"))
        finally:
            tcl.Tcl_DeleteInterp(interp)
    return tk.Tk()


def launch(args):
    root = create_root()
    root.configure(bg=BG)
    width = min(1180, root.winfo_screenwidth() - 90)
    height = min(860, root.winfo_screenheight() - 110)
    root.geometry("{}x{}+{}+{}".format(width, height,
                  (root.winfo_screenwidth() - width) // 2,
                  (root.winfo_screenheight() - height) // 2 - 20))
    root.minsize(900, 680)
    try:
        root.iconbitmap(str(BASE / "aegis.ico"))
    except tk.TclError:
        pass
    HomeScreen(root, args)
    root.mainloop()
