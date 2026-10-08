"""运行：python -m unittest discover -s tests -v"""
import random
import threading
import unittest

from reversi.core import BLACK, WHITE, EMPTY, DIRECTIONS, Board, Game, Move
from reversi.search import SearchCancelled, SearchConfig, choose_move, evaluate, _Node


def pass_position():
    return Board.from_rows([".WBBBBBB", *(["BBBBBBBB"] * 6), "BBBBBBW."])


class RuleTests(unittest.TestCase):
    def test_initial_positions_and_legal_moves(self):
        board = Board.initial()
        self.assertEqual((board.count(BLACK), board.count(WHITE), board.count(EMPTY)), (2, 2, 60))
        for coordinate in ("E4", "D5"):
            self.assertEqual(board.cells[Move.parse(coordinate).index], BLACK)
        for coordinate in ("D4", "E5"):
            self.assertEqual(board.cells[Move.parse(coordinate).index], WHITE)
        self.assertEqual({str(m) for m in board.legal_moves(BLACK)}, {"D3", "C4", "F5", "E6"})
        self.assertEqual({str(m) for m in board.legal_moves(WHITE)}, {"E3", "F4", "C5", "D6"})

    def test_first_move_and_immutable_board(self):
        board = Board.initial()
        result = board.play(Move.parse("D3"), BLACK)
        self.assertEqual((result.count(BLACK), result.count(WHITE)), (4, 1))
        self.assertEqual(result.cells[Move.parse("D4").index], BLACK)
        self.assertEqual(board, Board.initial())

    def test_flip_all_eight_directions(self):
        cells = [EMPTY] * 64
        for dr, dc in DIRECTIONS:
            cells[(3 + dr) * 8 + 3 + dc] = WHITE
            cells[(3 + 2 * dr) * 8 + 3 + 2 * dc] = BLACK
        board = Board(tuple(cells))
        self.assertEqual(len(board.flips_for(Move(3, 3), BLACK)), 8)
        after = board.play(Move(3, 3), BLACK)
        self.assertEqual((after.count(BLACK), after.count(WHITE)), (17, 0))

    def test_empty_gap_cannot_be_captured(self):
        board = Board.from_rows([".W.B...."] + ["........"] * 7)
        self.assertEqual(board.flips_for(Move.parse("A1"), BLACK), ())

    def test_edge_does_not_wrap_to_next_row(self):
        board = Board.from_rows([".......W", "B......."] + ["........"] * 6)
        self.assertNotIn(Move.parse("G1"), board.legal_moves(BLACK))

    def test_illegal_moves_leave_game_unchanged(self):
        game = Game()
        for coordinate in ("A1", "D4"):
            with self.assertRaises(ValueError):
                game.play(Move.parse(coordinate))
            self.assertEqual(game.board, Board.initial())
            self.assertEqual(game.turn, BLACK)
            self.assertEqual(game.history, [])

    def test_auto_pass_then_terminal(self):
        game = Game(pass_position())
        ply = game.play(Move.parse("A1"))
        self.assertEqual(ply.passed, WHITE)
        self.assertEqual(game.turn, BLACK)
        self.assertFalse(game.over)
        game.play(Move.parse("H8"))
        self.assertTrue(game.over)
        self.assertEqual(game.board.count(BLACK), 64)
        self.assertEqual(game.board.winner(), BLACK)

    def test_initial_pass_is_normalized(self):
        game = Game(pass_position(), WHITE)
        self.assertEqual(game.last_pass, WHITE)
        self.assertEqual(game.turn, BLACK)

    def test_terminal_with_empty_squares(self):
        board = Board.from_rows(["B......."] + ["........"] * 7)
        self.assertTrue(Game(board).over)
        self.assertEqual(board.winner(), BLACK)

    def test_draw(self):
        board = Board.from_rows(["BBBBWWWW"] * 8)
        self.assertEqual(board.winner(), EMPTY)

    def test_undo_restores_pass_and_terminal_states(self):
        game = Game(pass_position())
        game.play(Move.parse("A1"))
        second = game.board
        game.play(Move.parse("H8"))
        self.assertTrue(game.undo())
        self.assertFalse(game.over)
        self.assertEqual(game.board, second)
        self.assertEqual(game.turn, BLACK)
        self.assertTrue(game.undo())
        self.assertEqual(game.board, pass_position())
        self.assertFalse(game.undo())

    def test_coordinate_validation(self):
        self.assertEqual(str(Move.parse(" d3 ")), "D3")
        for text in ("A0", "I1", "A10", "3D", ""):
            with self.assertRaises(ValueError):
                Move.parse(text)
        with self.assertRaises(ValueError):
            Move(8, 0)


class SearchTests(unittest.TestCase):
    def test_minimax_alpha_beta_same_value_at_multiple_positions(self):
        rng = random.Random(2026)
        game = Game()
        positions = [(game.board, game.turn)]
        for index in range(24):
            game.play(rng.choice(game.board.legal_moves(game.turn)))
            if index in (7, 15, 23):
                positions.append((game.board, game.turn))
        for board, player in positions:
            with self.subTest(empty=board.count(EMPTY), player=player):
                config = SearchConfig(depth=3)
                plain = choose_move("minimax", board, player, config)
                pruned = choose_move("alphabeta", board, player, config)
                self.assertEqual(plain.value, pruned.value)
                self.assertLessEqual(pruned.nodes, plain.nodes)
                self.assertIn(plain.move, board.legal_moves(player))
                self.assertIn(pruned.move, board.legal_moves(player))

    def test_search_counts_pass_without_spending_move_depth(self):
        for algorithm in ("minimax", "alphabeta"):
            with self.subTest(algorithm=algorithm):
                result = choose_move(algorithm, pass_position(), BLACK, SearchConfig(depth=2))
                self.assertEqual(result.value, 1_006_400)
                passed = choose_move(algorithm, pass_position(), WHITE, SearchConfig(depth=2))
                self.assertIsNone(passed.move)
                self.assertEqual(passed.value, -1_006_400)

    def test_evaluation_is_symmetric(self):
        board = Board.initial().play(Move.parse("D3"), BLACK)
        self.assertEqual(evaluate(board, BLACK), -evaluate(board, WHITE))

    def test_mcts_budget_and_repeatability(self):
        config = SearchConfig(iterations=120, seed=47)
        result = choose_move("mcts", Board.initial(), BLACK, config)
        repeated = choose_move("mcts", Board.initial(), BLACK, config)
        self.assertEqual(result.iterations, 120)
        self.assertEqual(sum(stat.visits for stat in result.root_stats), 120)
        self.assertEqual(result.root_stats, repeated.root_stats)
        self.assertEqual(result.move, repeated.move)
        self.assertIn(result.move, Board.initial().legal_moves(BLACK))
        self.assertTrue(all(-1 <= stat.mean_reward <= 1 for stat in result.root_stats))

    def test_mcts_opponent_tries_to_reduce_root_reward(self):
        # 对手节点应选择对根棋手更不利的分支，不能把对手当作队友。
        parent = _Node(Board.initial(), WHITE)
        root_favorable = _Node(Board.initial(), BLACK, parent, Move.parse("D3"))
        root_unfavorable = _Node(Board.initial(), BLACK, parent, Move.parse("C4"))
        root_favorable.visits = root_unfavorable.visits = 10
        root_favorable.reward, root_unfavorable.reward = 8, -8
        parent.visits, parent.children = 20, [root_favorable, root_unfavorable]
        self.assertIs(parent.select(BLACK, 0), root_unfavorable)
        parent.turn = BLACK
        self.assertIs(parent.select(BLACK, 0), root_favorable)

    def test_mcts_forced_move_and_forced_pass(self):
        board = Board.from_rows([".WBBBBBB"] + ["BBBBBBBB"] * 7)
        result = choose_move("mcts", board, BLACK, SearchConfig(iterations=20, seed=1))
        self.assertEqual(result.move, Move.parse("A1"))
        self.assertEqual(result.value, 1)
        self.assertIsNone(choose_move("mcts", board, WHITE).move)

    def test_all_searchers_terminal_draw(self):
        board = Board.from_rows(["BBBBWWWW"] * 8)
        for algorithm in ("minimax", "alphabeta", "mcts"):
            result = choose_move(algorithm, board, BLACK)
            self.assertIsNone(result.move)
            self.assertEqual(result.value, 0)

    def test_all_searchers_can_be_cancelled(self):
        cancel = threading.Event()
        cancel.set()
        for algorithm in ("minimax", "alphabeta", "mcts"):
            with self.subTest(algorithm=algorithm), self.assertRaises(SearchCancelled):
                choose_move(algorithm, Board.initial(), BLACK, cancel=cancel)

    def test_complete_games_between_algorithms(self):
        pairs = (("minimax", "alphabeta"), ("alphabeta", "mcts"), ("mcts", "minimax"))
        for black, white in pairs:
            with self.subTest(black=black, white=white):
                game = Game()
                while not game.over:
                    self.assertLess(len(game.history), 60)
                    old_empty = game.board.count(EMPTY)
                    algorithm = black if game.turn == BLACK else white
                    config = SearchConfig(depth=1, iterations=10, seed=100 + len(game.history))
                    result = choose_move(algorithm, game.board, game.turn, config)
                    self.assertIn(result.move, game.board.legal_moves(game.turn))
                    game.play(result.move)
                    self.assertEqual(game.board.count(EMPTY), old_empty - 1)
                self.assertFalse(game.board.legal_moves(BLACK))
                self.assertFalse(game.board.legal_moves(WHITE))


if __name__ == "__main__":
    unittest.main()
