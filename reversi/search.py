"""独立实现 Minimax、Alpha-Beta 和 UCT 蒙特卡洛树搜索。"""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from threading import Event

from .core import BLACK, WHITE, EMPTY, Board, Move, RAYS

ALGORITHM_NAMES = {
    "minimax": "最小最大搜索",
    "alphabeta": "Alpha-Beta 剪枝搜索",
    "mcts": "蒙特卡洛树搜索",
}
POSITION_WEIGHTS = (
    120, -25, 20, 5, 5, 20, -25, 120,
    -25, -45, -5, -5, -5, -5, -45, -25,
    20, -5, 15, 3, 3, 15, -5, 20,
    5, -5, 3, 3, 3, 3, -5, 5,
    5, -5, 3, 3, 3, 3, -5, 5,
    20, -5, 15, 3, 3, 15, -5, 20,
    -25, -45, -5, -5, -5, -5, -45, -25,
    120, -25, 20, 5, 5, 20, -25, 120,
)


class SearchCancelled(Exception):
    """重新开始、悔棋或关闭窗口时停止后台搜索。"""


@dataclass(frozen=True, slots=True)
class SearchConfig:
    depth: int = 3
    iterations: int = 600
    exploration: float = math.sqrt(2.0)
    seed: int | None = None

    def __post_init__(self):
        if self.depth < 1:
            raise ValueError("搜索深度至少为 1。")
        if self.iterations < 1:
            raise ValueError("MCTS 模拟次数至少为 1。")
        if not math.isfinite(self.exploration) or self.exploration < 0:
            raise ValueError("UCT 探索常数必须是非负有限数。")


@dataclass(frozen=True, slots=True)
class RootStat:
    move: Move
    visits: int
    mean_reward: float


@dataclass(frozen=True, slots=True)
class SearchResult:
    move: Move | None
    value: float
    nodes: int
    elapsed: float
    cutoffs: int = 0
    depth: int = 0
    iterations: int = 0
    root_stats: tuple[RootStat, ...] = ()


@dataclass(slots=True)
class _Context:
    root_player: int
    cancel: Event | None = None
    nodes: int = 0
    cutoffs: int = 0

    def check(self):
        if self.cancel is not None and self.cancel.is_set():
            raise SearchCancelled()


def terminal_value(board: Board, player: int) -> float:
    """终局胜负优先于所有非终局估值；同为获胜时偏好更多净胜子。"""
    difference = board.count(player) - board.count(-player)
    if difference == 0:
        return 0.0
    return (1_000_000.0 if difference > 0 else -1_000_000.0) + 100 * difference


def evaluate(board: Board, player: int) -> float:
    """统一估值：位置、行动力、前沿棋子和随阶段变化的棋子数。"""
    own_moves, other_moves = board.legal_moves(player), board.legal_moves(-player)
    if not own_moves and not other_moves:
        return terminal_value(board, player)
    positional = sum(weight * piece * player for weight, piece in zip(POSITION_WEIGHTS, board.cells))
    mobility = 15 * (len(own_moves) - len(other_moves))
    frontier = 0
    for index, piece in enumerate(board.cells):
        if piece and any(ray and board.cells[ray[0]] == EMPTY for ray in RAYS[index]):
            frontier += piece * player
    piece_weight = 1 if board.count(EMPTY) > 16 else 8
    material = piece_weight * (board.count(player) - board.count(-player))
    return float(positional + mobility - 5 * frontier + material)


def _minimax(board: Board, turn: int, depth: int, ctx: _Context) -> tuple[float, Move | None]:
    ctx.check()
    ctx.nodes += 1
    moves = board.legal_moves(turn)
    if not moves:
        if not board.legal_moves(-turn):
            return terminal_value(board, ctx.root_player), None
        # 跳过回合不消耗落子深度；双方均无合法棋步才终止。
        value, _ = _minimax(board, -turn, depth, ctx)
        return value, None
    if depth == 0:
        return evaluate(board, ctx.root_player), None
    maximizing = turn == ctx.root_player
    best_value = -math.inf if maximizing else math.inf
    best_move = None
    for move in moves:
        value, _ = _minimax(board.play(move, turn), -turn, depth - 1, ctx)
        if (maximizing and value > best_value) or (not maximizing and value < best_value):
            best_value, best_move = value, move
    return best_value, best_move


def minimax(board: Board, player: int, config: SearchConfig | None = None,
            cancel: Event | None = None) -> SearchResult:
    """完整遍历给定深度的博弈树，不做剪枝。"""
    config = config or SearchConfig()
    started = time.perf_counter()
    ctx = _Context(player, cancel)
    value, move = _minimax(board, player, config.depth, ctx)
    return SearchResult(move, value, ctx.nodes, time.perf_counter() - started, depth=config.depth)


def _alpha_beta(board: Board, turn: int, depth: int, alpha: float, beta: float,
                ctx: _Context) -> tuple[float, Move | None]:
    ctx.check()
    ctx.nodes += 1
    moves = board.legal_moves(turn)
    if not moves:
        if not board.legal_moves(-turn):
            return terminal_value(board, ctx.root_player), None
        value, _ = _alpha_beta(board, -turn, depth, alpha, beta, ctx)
        return value, None
    if depth == 0:
        return evaluate(board, ctx.root_player), None
    maximizing = turn == ctx.root_player
    best_value = -math.inf if maximizing else math.inf
    best_move = None
    # 优先搜索角等位置，增加剪枝机会；不改变 Minimax 的最优估值。
    ordered = sorted(moves, key=lambda move: POSITION_WEIGHTS[move.index], reverse=True)
    for move in ordered:
        value, _ = _alpha_beta(board.play(move, turn), -turn, depth - 1, alpha, beta, ctx)
        if (maximizing and value > best_value) or (not maximizing and value < best_value):
            best_value, best_move = value, move
        if maximizing:
            alpha = max(alpha, best_value)
        else:
            beta = min(beta, best_value)
        if alpha >= beta:
            ctx.cutoffs += 1
            break
    return best_value, best_move


def alphabeta(board: Board, player: int, config: SearchConfig | None = None,
              cancel: Event | None = None) -> SearchResult:
    config = config or SearchConfig()
    started = time.perf_counter()
    ctx = _Context(player, cancel)
    value, move = _alpha_beta(board, player, config.depth, -math.inf, math.inf, ctx)
    return SearchResult(move, value, ctx.nodes, time.perf_counter() - started,
                        cutoffs=ctx.cutoffs, depth=config.depth)


@dataclass(slots=True)
class _Node:
    board: Board
    turn: int
    parent: _Node | None = None
    move: Move | None = None
    children: list[_Node] = field(default_factory=list)
    visits: int = 0
    reward: float = 0.0
    untried: list[Move | None] = field(init=False)

    def __post_init__(self):
        self.untried = list(self.board.legal_moves(self.turn))
        if not self.untried and self.board.legal_moves(-self.turn):
            self.untried = [None]  # 显式的强制跳过节点

    def select(self, root_player: int, exploration: float) -> _Node:
        # 奖励统一以根棋手视角保存；对手节点必须偏好根棋手较低的收益。
        sign = 1 if self.turn == root_player else -1
        log_visits = math.log(max(1, self.visits))

        def uct(child: _Node) -> float:
            if child.visits == 0:
                return math.inf
            return sign * child.reward / child.visits + exploration * math.sqrt(log_visits / child.visits)

        return max(self.children, key=uct)


def _rollout(board: Board, turn: int, root_player: int, rng: random.Random, ctx: _Context) -> float:
    consecutive_passes = 0
    while consecutive_passes < 2:
        ctx.check()
        moves = board.legal_moves(turn)
        if moves:
            board = board.play(rng.choice(moves), turn)
            consecutive_passes = 0
        else:
            consecutive_passes += 1
        turn = -turn
    difference = board.count(root_player) - board.count(-root_player)
    return 1.0 if difference > 0 else -1.0 if difference < 0 else 0.0


def mcts(board: Board, player: int, config: SearchConfig | None = None,
         cancel: Event | None = None) -> SearchResult:
    """UCT：选择 → 扩展 → 随机模拟到终局 → 回传；最终选访问次数最多的着法。"""
    config = config or SearchConfig()
    started = time.perf_counter()
    ctx = _Context(player, cancel)
    ctx.check()
    root = _Node(board, player)
    if not board.legal_moves(player):
        # None 表示必须跳过或已经终局，不会伪造一个落子。
        value = 0.0
        if board.is_terminal():
            difference = board.count(player) - board.count(-player)
            value = 1.0 if difference > 0 else -1.0 if difference < 0 else 0.0
        return SearchResult(None, value, 1, time.perf_counter() - started)
    rng = random.Random(config.seed)
    nodes = 1
    for _ in range(config.iterations):
        ctx.check()
        node = root
        # 1. 选择：走到尚可扩展的节点或终局节点。
        while not node.untried and node.children:
            node = node.select(player, config.exploration)
        # 2. 扩展：随机取一个尚未尝试的合法动作。
        if node.untried:
            move = node.untried.pop(rng.randrange(len(node.untried)))
            child_board = node.board if move is None else node.board.play(move, node.turn)
            child = _Node(child_board, -node.turn, node, move)
            node.children.append(child)
            node = child
            nodes += 1
        # 3. 模拟：采用均匀随机策略，直到双方都不能落子。
        reward = _rollout(node.board, node.turn, player, rng, ctx)
        # 4. 回传：各节点记录同一个根棋手视角的结果。
        while node is not None:
            node.visits += 1
            node.reward += reward
            node = node.parent
    best = max(root.children, key=lambda child: (child.visits, child.reward / child.visits))
    stats = tuple(RootStat(child.move, child.visits, child.reward / child.visits)
                  for child in root.children if child.move is not None)
    return SearchResult(best.move, best.reward / best.visits, nodes, time.perf_counter() - started,
                        iterations=config.iterations, root_stats=stats)


SEARCHERS = {"minimax": minimax, "alphabeta": alphabeta, "mcts": mcts}


def choose_move(algorithm: str, board: Board, player: int, config: SearchConfig | None = None,
                cancel: Event | None = None) -> SearchResult:
    if player not in (BLACK, WHITE):
        raise ValueError("棋手必须是 BLACK 或 WHITE。")
    if algorithm not in SEARCHERS:
        raise ValueError(f"未知算法：{algorithm}")
    return SEARCHERS[algorithm](board, player, config, cancel)
