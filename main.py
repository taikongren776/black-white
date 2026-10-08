"""运行 python main.py 打开图形界面；--cli 开启命令行对局。"""
from __future__ import annotations

import argparse
from dataclasses import replace

from reversi.core import BLACK, WHITE, EMPTY, Game, Move, player_name
from reversi.search import ALGORITHM_NAMES, SearchConfig, choose_move


def cli_game(black: str, white: str, config: SearchConfig) -> int:
    game = Game()
    players = {BLACK: black, WHITE: white}
    while not game.over:
        print("\n" + str(game.board))
        print(f"黑 {game.board.count(BLACK)} : 白 {game.board.count(WHITE)}；轮到{player_name(game.turn)}")
        actor = players[game.turn]
        if actor == "human":
            print("合法位置：" + "、".join(map(str, game.board.legal_moves(game.turn))))
            try:
                text = input("输入坐标（例如 D3），u 悔棋，q 退出：").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n已退出。")
                return 0
            if text.lower() == "q":
                return 0
            if text.lower() == "u":
                if game.undo():
                    while game.history and players[game.turn] != "human":
                        game.undo()
                else:
                    print("当前没有可以撤销的落子。")
                continue
            try:
                move = Move.parse(text)
                ply = game.play(move)
            except ValueError as exc:
                print(exc)
                continue
        else:
            turn_config = config if config.seed is None else replace(config, seed=config.seed + len(game.history))
            result = choose_move(actor, game.board, game.turn, turn_config)
            if result.move is None:
                raise RuntimeError("当前棋手有合法着法，但搜索未返回落子。")
            print(f"{ALGORITHM_NAMES[actor]}：{result.move}，节点 {result.nodes}，用时 {result.elapsed:.3f} 秒")
            if actor == "mcts":
                print(f"模拟 {result.iterations} 次，平均收益 {result.value:+.3f}")
            else:
                print(f"深度 {result.depth}，估值 {result.value:+.0f}，剪枝 {result.cutoffs} 次")
            ply = game.play(result.move)
        print(f"{player_name(ply.player)}落子 {ply.move}，翻转 {ply.flipped} 子。")
        if ply.passed is not None:
            print(f"{player_name(ply.passed)}无合法落子，自动跳过。")
    print("\n" + str(game.board))
    winner = game.board.winner()
    print(f"最终比分：黑 {game.board.count(BLACK)} : 白 {game.board.count(WHITE)}")
    print("平局。" if winner == EMPTY else f"{player_name(winner)}获胜。")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="黑白棋：最小最大搜索、Alpha-Beta 剪枝搜索、蒙特卡洛树搜索。")
    parser.add_argument("--cli", action="store_true", help="使用命令行界面")
    choices = ("human", *ALGORITHM_NAMES)
    parser.add_argument("--black", choices=choices, default="human", help="黑方棋手，默认 human")
    parser.add_argument("--white", choices=choices, default="alphabeta", help="白方棋手，默认 alphabeta")
    parser.add_argument("--depth", type=int, default=3, help="前两种算法的搜索深度，默认 3")
    parser.add_argument("--iterations", type=int, default=600, help="每步 MCTS 模拟次数，默认 600")
    parser.add_argument("--seed", type=int, default=None, help="固定随机种子，便于重复实验")
    args = parser.parse_args()
    if not 1 <= args.depth <= 5:
        parser.error("搜索深度必须为 1—5。")
    if not 10 <= args.iterations <= 10000:
        parser.error("MCTS 模拟次数必须为 10—10000。")
    config = SearchConfig(depth=args.depth, iterations=args.iterations, seed=args.seed)
    if args.cli:
        return cli_game(args.black, args.white, config)
    try:
        from reversi.gui import launch
    except ImportError as exc:
        parser.exit(1, f"图形界面需要 Tkinter：{exc}\n请安装带 Tcl/Tk 的 Python，或使用 --cli。\n")
    launch(args.black, args.white, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
