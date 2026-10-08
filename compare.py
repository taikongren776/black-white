"""在同一初始局面和同一中盘局面比较三种算法，支持导出 CSV。"""
import argparse
import csv
import random

from reversi.core import Game, player_name
from reversi.search import ALGORITHM_NAMES, SearchConfig, choose_move


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=600)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--csv", default=None, help="可选的 CSV 输出路径")
    args = parser.parse_args()
    if not 1 <= args.depth <= 5 or not 10 <= args.iterations <= 10000:
        parser.error("深度必须为 1—5，模拟次数必须为 10—10000。")
    config = SearchConfig(depth=args.depth, iterations=args.iterations, seed=args.seed)
    game = Game()
    positions = [("初始局面", game.board, game.turn)]
    rng = random.Random(args.seed)
    for _ in range(20):
        if game.over:
            break
        game.play(rng.choice(game.board.legal_moves(game.turn)))
    positions.append(("20步后局面", game.board, game.turn))
    records = []
    for label, board, turn in positions:
        print(f"\n{label}（{player_name(turn)}落子）\n{board}")
        for algorithm, name in ALGORITHM_NAMES.items():
            result = choose_move(algorithm, board, turn, config)
            record = {"position": label, "algorithm": algorithm, "move": str(result.move),
                      "value": result.value, "nodes": result.nodes, "cutoffs": result.cutoffs,
                      "depth": result.depth, "iterations": result.iterations,
                      "seconds": round(result.elapsed, 6)}
            records.append(record)
            metric = "平均收益" if algorithm == "mcts" else "估值"
            print(f"{name:18s}  落子 {result.move}  {metric} {result.value:+.3f}  "
                  f"节点 {result.nodes:6d}  剪枝 {result.cutoffs:5d}  {result.elapsed:.4f} 秒")
    print("\n说明：MCTS 平均收益与前两种算法的启发式估值不是同一量纲；节点计数含义也不同。")
    if args.csv:
        with open(args.csv, "w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        print(f"结果已保存：{args.csv}")


if __name__ == "__main__":
    main()
