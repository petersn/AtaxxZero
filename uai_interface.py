#!/usr/bin/python

import sys, string
import ataxx_rules

def uai_encode_square(xy):
	x, y = xy
	y = 6 - y
	return "%s%i" % (string.ascii_lowercase[x], y + 1)

def uai_encode_move(move):
	if move == "pass":
		return "0000"
	start, end = move
	if start == "c":
		return uai_encode_square(end)
	return "%s%s" % (uai_encode_square(start), uai_encode_square(end))

def uai_decode_square(s):
	x, y = string.ascii_lowercase.index(s[0].lower()), int(s[1]) - 1
	y = 6 - y
	return x, y

def uai_decode_move(s):
	if s in ("pass", "none", "0000"):
		return "pass"
	elif len(s) == 2:
		return "c", uai_decode_square(s)
	elif len(s) == 4:
		return uai_decode_square(s[:2]), uai_decode_square(s[2:])
	else:
		raise Exception("Bad UAI move string: %r" % s)

def test():
	for s in ["f2", "c3d5"]:
		assert uai_encode_move(uai_decode_move(s)) == s
	for m in [("c", (4, 3)), ((4, 3), (2, 5))]:
		assert uai_decode_move(uai_encode_move(m)) == m
test()

def main(args):
	board = ataxx_rules.AtaxxState.initial()
	eng = engine.MCTSEngine()
	if args.visits != None:
#		eng.VISITS = args.visits
		eng.MAX_STEPS = args.visits

	while True:
		line = input()
		if line == "quit":
			exit()
		elif line == "uai":
			print("id name AtaxxZero")
			print("id author Peter Schmidt-Nielsen")
			print("uaiok")
		elif line == "uainewgame":
			board = ataxx_rules.AtaxxState.initial()
			eng = engine.MCTSEngine()
		elif line == "isready":
			print("readyok")
		elif line.startswith("moves "):
			for move in line[6:].split():
				move = uai_decode_move(move)
				board.move(move)
			eng.set_state(board.copy())
		elif line.startswith("position fen "):
			board = ataxx_rules.AtaxxState.from_fen(line[13:])
			eng.set_state(board)
			if args.show_game:
				print("===", file=sys.stderr)
				print(board, file=sys.stderr)
		elif line.startswith("go movetime "):
			ms = int(line[12:])
			ms -= args.safety_ms
			if args.visits == None:
				move = eng.genmove(ms * 1e-3, use_weighted_exponent=5.0)
			else:
				# This is safe, because of the visit limit we set above.
				move = eng.genmove(1000000.0, use_weighted_exponent=5.0)
			print("bestmove %s" % (uai_encode_move(move),))
			if args.show_game:
				post_move_board = board.copy()
				post_move_board.move(move)
				print(post_move_board, file=sys.stderr)
		elif line == "showboard":
			print(board)
			print("boardok")
		sys.stdout.flush()

if __name__ == "__main__":
	import engine
	engine.model.Network.FILTERS = 128
	engine.model.Network.BLOCK_COUNT = 12

	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument("--network-path", metavar="NETWORK", type=str, help="Name of the model to load.")
	parser.add_argument("--visits", metavar="VISITS", default=None, type=int, help="Number of visits during MCTS.")
	parser.add_argument("--safety-ms", metavar="MS", default=0, type=int, help="Number of milliseconds to shave off of each movetime for safety.")
	parser.add_argument("--show-game", action="store_true", help="Show the game on stderr.")
	args = parser.parse_args()
	print(args, file=sys.stderr)

	engine.setup_evaluator(use_rpc=False)
	engine.initialize_model(args.network_path)
	main(args)

