"""黑白棋规则。坐标：列 A—H，行 1—8；数组下标从 0 开始。"""
from __future__ import annotations

from dataclasses import dataclass

EMPTY, BLACK, WHITE = 0, 1, -1
DIRECTIONS = tuple((dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc)


def _build_rays():
    result = []
    for index in range(64):
        row, col = divmod(index, 8)
        rays = []
        for dr, dc in DIRECTIONS:
            ray = []
            r, c = row + dr, col + dc
            while 0 <= r < 8 and 0 <= c < 8:
                ray.append(r * 8 + c)
                r, c = r + dr, c + dc
            rays.append(tuple(ray))
        result.append(tuple(rays))
    return tuple(result)


RAYS = _build_rays()


def player_name(player: int) -> str:
    return "黑方" if player == BLACK else "白方"


def _check_player(player: int) -> None:
    if player not in (BLACK, WHITE):
        raise ValueError("棋手必须是 BLACK 或 WHITE。")


@dataclass(frozen=True, order=True, slots=True)
class Move:
    row: int
    col: int

    def __post_init__(self):
        if not (0 <= self.row < 8 and 0 <= self.col < 8):
            raise ValueError("坐标超出 8×8 棋盘。")

    @property
    def index(self) -> int:
        return self.row * 8 + self.col

    def __str__(self) -> str:
        return f"{chr(65 + self.col)}{self.row + 1}"

    @classmethod
    def parse(cls, text: str) -> Move:
        text = text.strip().upper()
        if len(text) != 2 or text[0] not in "ABCDEFGH" or text[1] not in "12345678":
            raise ValueError("请输入 A1—H8 范围的坐标，例如 D3。")
        return cls(int(text[1]) - 1, ord(text[0]) - 65)


@dataclass(frozen=True, slots=True)
class Board:
    # 不可变棋盘：搜索分支不会污染实际对局或其他分支。
    cells: tuple[int, ...]

    def __post_init__(self):
        if len(self.cells) != 64 or any(x not in (EMPTY, BLACK, WHITE) for x in self.cells):
            raise ValueError("棋盘必须包含 64 个合法格子。")
        if not isinstance(self.cells, tuple):
            object.__setattr__(self, "cells", tuple(self.cells))

    @classmethod
    def initial(cls) -> Board:
        cells = [EMPTY] * 64
        cells[3 * 8 + 4] = cells[4 * 8 + 3] = BLACK  # E4、D5
        cells[3 * 8 + 3] = cells[4 * 8 + 4] = WHITE  # D4、E5
        return cls(tuple(cells))

    @classmethod
    def from_rows(cls, rows: list[str] | tuple[str, ...]) -> Board:
        """用于实验和测试：B 黑棋，W 白棋，. 空格。"""
        if len(rows) != 8 or any(len(row) != 8 for row in rows):
            raise ValueError("需要 8 行、每行 8 个字符。")
        symbols = {".": EMPTY, "B": BLACK, "W": WHITE}
        try:
            return cls(tuple(symbols[ch] for row in rows for ch in row))
        except KeyError as exc:
            raise ValueError("棋盘字符只能是 B、W、.。") from exc

    def count(self, player: int) -> int:
        return self.cells.count(player)

    def flips_for(self, move: Move, player: int) -> tuple[int, ...]:
        """返回八个方向上全部被夹住的棋子；空元组表示非法落子。"""
        _check_player(player)
        if self.cells[move.index] != EMPTY:
            return ()
        flips = []
        for ray in RAYS[move.index]:
            captured = []
            for index in ray:
                piece = self.cells[index]
                if piece == -player:
                    captured.append(index)
                else:
                    if piece == player and captured:
                        flips.extend(captured)
                    break
        return tuple(flips)

    def legal_moves(self, player: int) -> tuple[Move, ...]:
        _check_player(player)
        moves = []
        cells = self.cells
        for index, piece in enumerate(cells):
            if piece != EMPTY:
                continue
            legal = False
            for ray in RAYS[index]:
                if not ray or cells[ray[0]] != -player:
                    continue
                for other in ray[1:]:
                    if cells[other] == -player:
                        continue
                    if cells[other] == player:
                        legal = True
                    break
                if legal:
                    moves.append(Move(*divmod(index, 8)))
                    break
        return tuple(moves)

    def play(self, move: Move, player: int) -> Board:
        flips = self.flips_for(move, player)
        if not flips:
            raise ValueError(f"{move} 不是{player_name(player)}的合法落子位置。")
        cells = list(self.cells)
        cells[move.index] = player
        for index in flips:
            cells[index] = player
        return Board(tuple(cells))

    def is_terminal(self) -> bool:
        return EMPTY not in self.cells or not (self.legal_moves(BLACK) or self.legal_moves(WHITE))

    def winner(self) -> int:
        """终局赢家；平局返回 EMPTY。非终局调用会报错。"""
        if not self.is_terminal():
            raise ValueError("对局尚未结束。")
        difference = self.count(BLACK) - self.count(WHITE)
        return BLACK if difference > 0 else WHITE if difference < 0 else EMPTY

    def __str__(self) -> str:
        symbols = {EMPTY: ".", BLACK: "B", WHITE: "W"}
        rows = ["  A B C D E F G H"]
        for row in range(8):
            rows.append(f"{row + 1} " + " ".join(symbols[x] for x in self.cells[row * 8:row * 8 + 8]))
        return "\n".join(rows)


@dataclass(frozen=True, slots=True)
class Ply:
    before: Board
    player: int
    move: Move
    flipped: int
    passed: int | None


class Game:
    """管理轮次、强制跳过、终局和悔棋。一次 play 必须真正落下一子。"""

    def __init__(self, board: Board | None = None, turn: int = BLACK):
        _check_player(turn)
        self.board = board if board is not None else Board.initial()
        self.turn = turn
        self.history: list[Ply] = []
        self.over = self.board.is_terminal()
        self.last_pass: int | None = None
        if not self.over and not self.board.legal_moves(self.turn):
            self.last_pass = self.turn
            self.turn = -self.turn

    def play(self, move: Move) -> Ply:
        if self.over:
            raise ValueError("对局已经结束。")
        before, player = self.board, self.turn
        flipped = len(before.flips_for(move, player))
        self.board = before.play(move, player)
        self.last_pass = None
        if self.board.legal_moves(-player):
            self.turn = -player
        elif self.board.legal_moves(player):
            self.turn = player
            self.last_pass = -player
        else:
            self.turn = -player
            self.over = True
        ply = Ply(before, player, move, flipped, self.last_pass)
        self.history.append(ply)
        return ply

    def undo(self) -> bool:
        if not self.history:
            return False
        ply = self.history.pop()
        self.board, self.turn, self.over = ply.before, ply.player, False
        self.last_pass = self.history[-1].passed if self.history else None
        return True
