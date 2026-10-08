"""Tkinter 图形界面。所有搜索在线程中运行，Tk 操作仅在主线程执行。"""
from __future__ import annotations

import queue
import threading
import tkinter as tk
from concurrent.futures import CancelledError, ThreadPoolExecutor
from dataclasses import replace
from tkinter import messagebox, ttk

from .core import BLACK, WHITE, EMPTY, Game, Move, player_name
from .search import ALGORITHM_NAMES, SearchCancelled, SearchConfig, SearchResult, choose_move

LABELS = {"human": "玩家", **ALGORITHM_NAMES}
KEYS = {value: key for key, value in LABELS.items()}
BG, PANEL, INK, MUTED, GREEN = "#f4f3ee", "#ffffff", "#20372d", "#63736b", "#256d50"


def result_summary(algorithm: str, result: SearchResult) -> str:
    if algorithm == "mcts":
        return (f"模拟 {result.iterations} 次 · 树节点 {result.nodes}\n"
                f"平均收益 {result.value:+.3f} · 用时 {result.elapsed:.3f} 秒")
    suffix = f" · 剪枝 {result.cutoffs} 次" if algorithm == "alphabeta" else ""
    return (f"深度 {result.depth} · 节点 {result.nodes}{suffix}\n"
            f"估值 {result.value:+.0f} · 用时 {result.elapsed:.3f} 秒")


class ReversiApp:
    def __init__(self, root: tk.Tk, black: str = "human", white: str = "alphabeta",
                 config: SearchConfig | None = None):
        self.root = root
        self.game = Game()
        self.config = config or SearchConfig()
        self.players = {BLACK: black, WHITE: white}
        self.generation = 0
        self.paused = False
        self.thinking = False
        self.closed = False
        self.cancel_event: threading.Event | None = None
        self.future = None
        self.ai_after = None
        self.poll_after = None
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="reversi-search")
        self.results: queue.Queue = queue.Queue()
        self.log_blocks: list[str] = []
        self.board_geometry = (0.0, 0.0, 1.0)
        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self._render()
        self._schedule_ai()
        self.poll_after = self.root.after(50, self._poll_results)

    def _build_ui(self):
        root = self.root
        root.title("黑白棋 · 三种搜索算法")
        root.geometry("1080x790")
        root.minsize(930, 720)
        root.configure(bg=BG)
        style = ttk.Style(root)
        style.theme_use("clam")
        style.configure("TFrame", background=BG)
        style.configure("Panel.TFrame", background=PANEL)
        style.configure("TLabel", background=BG, foreground=INK, font=("Microsoft YaHei UI", 10))
        style.configure("Panel.TLabel", background=PANEL)
        style.configure("Muted.TLabel", background=BG, foreground=MUTED)
        style.configure("TButton", font=("Microsoft YaHei UI", 10), padding=(8, 8))
        style.configure("Accent.TButton", foreground="white", background=GREEN)
        style.map("Accent.TButton", background=[("active", "#1b563e")])
        style.configure("TCheckbutton", background=PANEL, font=("Microsoft YaHei UI", 10))
        style.configure("TCombobox", padding=5, font=("Microsoft YaHei UI", 10))

        header = ttk.Frame(root, padding=(24, 16, 24, 10))
        header.pack(fill="x")
        ttk.Label(header, text="黑白棋", font=("Microsoft YaHei UI", 25, "bold")).pack(side="left")
        ttk.Label(header, text="REVERSI  /  搜索算法实验", style="Muted.TLabel").pack(side="left", padx=18, pady=(12, 0))

        body = ttk.Frame(root, padding=(20, 0, 20, 16))
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=0)
        body.rowconfigure(0, weight=1)
        board_panel = ttk.Frame(body)
        board_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 18))
        scores = ttk.Frame(board_panel)
        scores.pack(fill="x", pady=(0, 6))
        self.black_score = ttk.Label(scores, font=("Microsoft YaHei UI", 18, "bold"))
        self.black_score.pack(side="left", padx=(12, 26))
        self.white_score = ttk.Label(scores, font=("Microsoft YaHei UI", 18, "bold"))
        self.white_score.pack(side="left")
        self.empty_score = ttk.Label(scores, style="Muted.TLabel")
        self.empty_score.pack(side="right", padx=12)
        self.canvas = tk.Canvas(board_panel, bg=BG, highlightthickness=0, width=610, height=610)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda event: self._render_board())
        self.canvas.bind("<Button-1>", self._on_click)
        self.status_var = tk.StringVar()
        ttk.Label(board_panel, textvariable=self.status_var, font=("Microsoft YaHei UI", 12, "bold"),
                  wraplength=600).pack(fill="x", pady=(8, 4))
        self.hint_var = tk.StringVar()
        ttk.Label(board_panel, textvariable=self.hint_var, style="Muted.TLabel", wraplength=600).pack(fill="x")

        sidebar = ttk.Frame(body, style="Panel.TFrame", padding=18, width=310)
        sidebar.grid(row=0, column=1, sticky="nsew")
        sidebar.grid_propagate(False)
        sidebar.columnconfigure(0, weight=1)
        sidebar.rowconfigure(13, weight=1)
        ttk.Label(sidebar, text="对局设置", style="Panel.TLabel", font=("Microsoft YaHei UI", 14, "bold")).grid(row=0, sticky="w", pady=(0, 12))
        self.black_var = tk.StringVar(value=LABELS[self.players[BLACK]])
        self.white_var = tk.StringVar(value=LABELS[self.players[WHITE]])
        for row, text, variable in ((1, "黑方 · 先手", self.black_var), (3, "白方 · 后手", self.white_var)):
            ttk.Label(sidebar, text=text, style="Panel.TLabel").grid(row=row, sticky="w", pady=(0, 3))
            ttk.Combobox(sidebar, values=list(LABELS.values()), textvariable=variable,
                         state="readonly", width=23).grid(row=row + 1, sticky="ew", pady=(0, 10))
        settings = ttk.Frame(sidebar, style="Panel.TFrame")
        settings.grid(row=5, sticky="ew", pady=(0, 8))
        ttk.Label(settings, text="搜索深度", style="Panel.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(settings, text="MCTS 模拟次数", style="Panel.TLabel").grid(row=0, column=1, sticky="w", padx=(14, 0))
        self.depth_var = tk.StringVar(value=str(self.config.depth))
        self.iterations_var = tk.StringVar(value=str(self.config.iterations))
        ttk.Spinbox(settings, from_=1, to=5, width=8, textvariable=self.depth_var).grid(row=1, column=0, sticky="ew", pady=5)
        ttk.Spinbox(settings, from_=10, to=10000, increment=100, width=12,
                    textvariable=self.iterations_var).grid(row=1, column=1, sticky="ew", padx=(14, 0), pady=5)
        self.show_hints = tk.BooleanVar(value=True)
        ttk.Checkbutton(sidebar, text="显示合法落子位置", variable=self.show_hints,
                        command=self._render_board).grid(row=6, sticky="w", pady=(0, 8))
        ttk.Button(sidebar, text="开始 / 重新开始", style="Accent.TButton", command=self.new_game).grid(row=7, sticky="ew")
        buttons = ttk.Frame(sidebar, style="Panel.TFrame")
        buttons.grid(row=8, sticky="ew", pady=8)
        buttons.columnconfigure((0, 1), weight=1)
        self.pause_button = ttk.Button(buttons, text="暂停对局", command=self.toggle_pause)
        self.pause_button.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self.undo_button = ttk.Button(buttons, text="悔棋", command=self.undo)
        self.undo_button.grid(row=0, column=1, sticky="ew", padx=(4, 0))
        ttk.Label(sidebar, text="更改设置后，点击“开始 / 重新开始”。\n深度用于前两种算法；模拟次数用于 MCTS。",
                  style="Panel.TLabel", foreground=MUTED, font=("Microsoft YaHei UI", 8), wraplength=278).grid(row=9, sticky="w", pady=(0, 12))
        ttk.Separator(sidebar).grid(row=10, sticky="ew", pady=(0, 10))
        self.stats_var = tk.StringVar(value="搜索统计会在 AI 落子后显示。")
        ttk.Label(sidebar, textvariable=self.stats_var, style="Panel.TLabel", wraplength=275,
                  font=("Microsoft YaHei UI", 9)).grid(row=11, sticky="w", pady=(0, 12))
        ttk.Label(sidebar, text="棋谱", style="Panel.TLabel", font=("Microsoft YaHei UI", 11, "bold")).grid(row=12, sticky="w", pady=(0, 6))
        log_frame = ttk.Frame(sidebar, style="Panel.TFrame")
        log_frame.grid(row=13, sticky="nsew")
        self.log = tk.Text(log_frame, width=28, height=7, wrap="word", bg="#f7f8f5", fg=INK,
                           relief="flat", padx=9, pady=8, font=("Microsoft YaHei UI", 9), state="disabled")
        scroll = ttk.Scrollbar(log_frame, command=self.log.yview)
        self.log.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.log.pack(side="left", fill="both", expand=True)

    def _render_board(self):
        canvas = self.canvas
        canvas.delete("all")
        width, height = max(canvas.winfo_width(), 100), max(canvas.winfo_height(), 100)
        size = max(32, min(width - 52, height - 52))
        cell = size / 8
        x0, y0 = (width - size) / 2 + 8, (height - size) / 2 + 8
        self.board_geometry = (x0, y0, cell)
        canvas.create_rectangle(x0 - 5, y0 - 5, x0 + size + 5, y0 + size + 5, fill="#173e2c", outline="")
        legal = set(self.game.board.legal_moves(self.game.turn)) if not self.game.over and self.show_hints.get() else set()
        last_move = self.game.history[-1].move if self.game.history else None
        for i in range(8):
            canvas.create_text(x0 + (i + .5) * cell, y0 - 19, text=chr(65 + i), fill=MUTED, font=("Segoe UI", 10, "bold"))
            canvas.create_text(x0 - 20, y0 + (i + .5) * cell, text=str(i + 1), fill=MUTED, font=("Segoe UI", 10, "bold"))
        for row in range(8):
            for col in range(8):
                x, y = x0 + col * cell, y0 + row * cell
                canvas.create_rectangle(x, y, x + cell, y + cell, fill="#337b59" if (row + col) % 2 else "#307454",
                                        outline="#245e43", width=1)
                move = Move(row, col)
                piece = self.game.board.cells[move.index]
                margin = cell * .12
                if piece:
                    canvas.create_oval(x + margin + 1, y + margin + 3, x + cell - margin + 1, y + cell - margin + 3,
                                       fill="#20513b", outline="")
                    canvas.create_oval(x + margin, y + margin, x + cell - margin, y + cell - margin,
                                       fill="#202526" if piece == BLACK else "#f5f3e9",
                                       outline="#111717" if piece == BLACK else "#deddd4", width=2)
                    if move == last_move:
                        radius = max(3, cell * .045)
                        canvas.create_oval(x + cell / 2 - radius, y + cell / 2 - radius,
                                           x + cell / 2 + radius, y + cell / 2 + radius, fill="#e3b653", outline="")
                elif move in legal:
                    radius = cell * .075
                    canvas.create_oval(x + cell / 2 - radius, y + cell / 2 - radius,
                                       x + cell / 2 + radius, y + cell / 2 + radius, fill="#a3c9a8", outline="")

    def _render(self):
        board = self.game.board
        self.black_score.configure(text=f"● 黑方  {board.count(BLACK):02d}")
        self.white_score.configure(text=f"○ 白方  {board.count(WHITE):02d}")
        self.empty_score.configure(text=f"空格 {board.count(EMPTY)}")
        if self.game.over:
            winner = board.winner()
            self.status_var.set("对局结束 · 平局" if winner == EMPTY else f"对局结束 · {player_name(winner)}获胜")
            self.hint_var.set(f"最终比分：黑 {board.count(BLACK)} : 白 {board.count(WHITE)}。可重新开始或悔棋。")
        elif self.paused:
            self.status_var.set("对局已暂停")
            self.hint_var.set("点击“继续对局”后继续。")
        else:
            actor = LABELS[self.players[self.game.turn]]
            self.status_var.set(f"{player_name(self.game.turn)}正在思考 · {actor}" if self.thinking
                                else f"轮到{player_name(self.game.turn)} · {actor}")
            prefix = f"{player_name(self.game.last_pass)}无合法落子，已自动跳过。" if self.game.last_pass else ""
            action = "点击浅色圆点落子。" if self.players[self.game.turn] == "human" else "AI 将自动落子，可随时暂停。"
            self.hint_var.set(prefix + action + "金色点标记上一步。")
        self.pause_button.configure(text="继续对局" if self.paused else "暂停对局",
                                    state="disabled" if self.game.over else "normal")
        self.undo_button.configure(state="normal" if self.game.history else "disabled")
        self._render_board()
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.insert("end", "\n\n".join(self.log_blocks) if self.log_blocks else "黑方先行。\n请选择双方棋手，开始对局。")
        self.log.configure(state="disabled")
        self.log.see("end")

    def _on_click(self, event):
        if self.game.over or self.paused or self.players[self.game.turn] != "human":
            return
        x0, y0, cell = self.board_geometry
        col, row = int((event.x - x0) // cell), int((event.y - y0) // cell)
        if not (0 <= row < 8 and 0 <= col < 8):
            return
        move = Move(row, col)
        if move not in self.game.board.legal_moves(self.game.turn):
            self.hint_var.set("该位置不能夹住对手棋子。请选择合法落子位置。")
            return
        self._play(move)

    def _play(self, move: Move, result: SearchResult | None = None):
        actor = self.players[self.game.turn]
        ply = self.game.play(move)
        text = f"{len(self.game.history):02d}. {player_name(ply.player)} {ply.move} · 翻转 {ply.flipped} 子"
        if result is not None:
            summary = result_summary(actor, result)
            text += f"\n{LABELS[actor]}\n{summary}"
            self.stats_var.set(f"{player_name(ply.player)} · {LABELS[actor]}\n{summary}")
        if ply.passed is not None:
            text += f"\n{player_name(ply.passed)}无合法落子，跳过。"
        self.log_blocks.append(text)
        self._render()
        self._schedule_ai()

    def _cancel_search(self):
        self.generation += 1
        if self.ai_after is not None:
            self.root.after_cancel(self.ai_after)
            self.ai_after = None
        if self.cancel_event is not None:
            self.cancel_event.set()
        if self.future is not None:
            self.future.cancel()
        self.future = None
        self.thinking = False

    def _schedule_ai(self):
        if self.closed or self.paused or self.game.over or self.players[self.game.turn] == "human":
            return
        if self.future is not None or self.ai_after is not None:
            return
        generation = self.generation
        self.ai_after = self.root.after(180, lambda: self._start_ai(generation))

    def _start_ai(self, generation):
        self.ai_after = None
        if generation != self.generation or self.paused or self.game.over or self.closed:
            return
        board, turn = self.game.board, self.game.turn
        algorithm = self.players[turn]
        if algorithm == "human":
            return
        self.cancel_event = threading.Event()
        config = self.config
        if config.seed is not None:
            config = replace(config, seed=config.seed + len(self.game.history))
        self.thinking = True
        self._render()
        self.future = self.executor.submit(choose_move, algorithm, board, turn, config, self.cancel_event)

        def finished(future):
            try:
                result, error = future.result(), None
            except (CancelledError, SearchCancelled):
                result, error = None, SearchCancelled()
            except Exception as exc:
                result, error = None, exc
            self.results.put((generation, board, turn, result, error))

        self.future.add_done_callback(finished)

    def _poll_results(self):
        if self.closed:
            return
        while True:
            try:
                generation, board, turn, result, error = self.results.get_nowait()
            except queue.Empty:
                break
            # 忽略重新开始、悔棋或暂停前提交的过期搜索结果。
            if generation != self.generation:
                continue
            self.future = None
            self.thinking = False
            if error is not None:
                if not isinstance(error, SearchCancelled):
                    self.paused = True
                    messagebox.showerror("搜索失败", str(error), parent=self.root)
                self._render()
                continue
            if board != self.game.board or turn != self.game.turn or self.paused:
                continue
            if result.move is None:
                self.paused = True
                self._render()
                self.hint_var.set("算法未返回合法落子。请重新开始对局。")
                continue
            self._play(result.move, result)
        self.poll_after = self.root.after(50, self._poll_results)

    def new_game(self):
        try:
            depth, iterations = int(self.depth_var.get()), int(self.iterations_var.get())
            if not 1 <= depth <= 5 or not 10 <= iterations <= 10000:
                raise ValueError()
        except ValueError:
            messagebox.showerror("设置无效", "搜索深度请输入 1—5；MCTS 模拟次数请输入 10—10000。", parent=self.root)
            return
        self._cancel_search()
        self.config = replace(self.config, depth=depth, iterations=iterations)
        self.players = {BLACK: KEYS[self.black_var.get()], WHITE: KEYS[self.white_var.get()]}
        self.game = Game()
        self.log_blocks.clear()
        self.paused = False
        self.stats_var.set("搜索统计会在 AI 落子后显示。")
        self._render()
        self._schedule_ai()

    def toggle_pause(self):
        if self.game.over:
            return
        self._cancel_search()
        self.paused = not self.paused
        self._render()
        self._schedule_ai()

    def undo(self):
        if not self.game.history:
            return
        self._cancel_search()
        self.game.undo()
        self.log_blocks.pop()
        if "human" in self.players.values():
            # 人机对战回退至玩家的上一落子机会，包含自动跳过的情形。
            while self.game.history and self.players[self.game.turn] != "human":
                self.game.undo()
                self.log_blocks.pop()
            self.paused = False
        else:
            self.paused = True
        self.stats_var.set("已悔棋；下次 AI 落子后更新统计。")
        self._render()
        self._schedule_ai()

    def close(self):
        if self.closed:
            return
        self.closed = True
        self._cancel_search()
        if self.poll_after is not None:
            self.root.after_cancel(self.poll_after)
        self.executor.shutdown(wait=False, cancel_futures=True)
        self.root.destroy()


def launch(black: str = "human", white: str = "alphabeta", config: SearchConfig | None = None):
    root = tk.Tk()
    ReversiApp(root, black, white, config)
    root.mainloop()
