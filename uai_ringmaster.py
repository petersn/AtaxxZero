#!/usr/bin/python

import sys, random, string, subprocess, atexit, time, datetime, itertools
import uai_interface
import ataxx_rules

OPENING_DEPTH = 4

class UAIPlayer:
	def __init__(self, cmd):
		self.cmd = cmd
		# WARNING: What I'm doing here is technically invalid!
		# In theory the output pipe of this process could fill up before I read, making it hang.
		# TODO: Decide if I care.
		self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
#		atexit.register(self.proc.kill)
		self.send("uai\n")
		self.send("setoption name Hash value 1024\n")
#		self.send("setoption name Search value most-captures\n")

	def send(self, s):
		self.proc.stdin.write(s)
		self.proc.stdin.flush()

	def quit(self):
		self.send("quit\n")
		try:
			self.proc.kill()
		except OSError:
			pass
		self.proc.wait()

	def reset(self):
		self.send("uainewgame\n")

	def set_state(self, board):
		self.send("position fen %s\n" % board.fen())

	def move(self, move):
		self.send("moves %s\n" % uai_interface.uai_encode_move(move))

	def genmove(self, ms=1000):
#		if ("uai_interface" in " ".join(self.cmd)) or True:
#			print "More time alotted."
#			time *= 5
#		if "ataxx-engine" in " ".join(self.cmd):
#			print " ".join(self.cmd)[:10], "--", self.showboard()
		self.send("go movetime %i\n" % ms)
		while True:
			line = self.proc.stdout.readline().strip()
			if not line:
				raise Exception("Bad UAI!")
			if line.startswith("bestmove "):
#				print " ".join(self.cmd)[:10], "##", line
				return uai_interface.uai_decode_move(line[9:])
#			print " ".join(self.cmd)[:10], "::", line

	def showboard(self):
		self.send("showboard\n")
		lines = []
		while True:
			line = self.proc.stdout.readline()
			if line.strip() == "boardok":
				break
			lines.append(line)
		s = "".join(lines)
		s = s.replace("X", ataxx_rules.RED + "X" + ataxx_rules.ENDC)
		s = s.replace("O", ataxx_rules.BLUE + "O" + ataxx_rules.ENDC)
		return s

def play_one_game(args, engine1, engine2, opening_moves):
	print('Game: "%s" vs "%s" with opening: [%s]' % (
		" ".join(engine1),
		" ".join(engine2),
		", ".join(uai_interface.uai_encode_move(move) for move in opening_moves),
	))
	game = {
		"moves": [],
		"opening": opening_moves,
		"start_time": time.time(),
		"white": engine1,
		"black": engine2,
	}

	players = [UAIPlayer(engine1), UAIPlayer(engine2)]
	board = ataxx_rules.AtaxxState.initial()
	ply_number = 0

	def print_state():
		if args.show_games:
			print()
			print("===================== Player %i move." % (ply_number % 2 + 1,))
			print("[%3i plies] Score: %2i - %2i" % (ply_number, board.board.count(1), board.board.count(2)))
			print(board.fen())
			print(board)
		else:
			print("\r[%3i plies] Score: %2i - %2i " % (ply_number, board.board.count(1), board.board.count(2)), end=' ')
			sys.stdout.flush()

	while board.result() == None:
		print_state()
		# If there is only one legal move then force it.
		if ply_number < len(opening_moves):
			move = opening_moves[ply_number]
		elif len(board.legal_moves()) == 1:
			move, = board.legal_moves()
		else:
			ms = int(args.tc * 1000)
			move = players[ply_number % 2].genmove(ms)
		if args.show_games:
			print("Move:", uai_interface.uai_encode_move(move))
		try:
			board.move(move)
		except Exception as e:
			print("Exception:", e)
			print(move)
			print(board)
			print(game)
			print(board.fen())
			print(uai_interface.uai_encode_move(move))
			raise e
		game["moves"].append(move)
		for player in players:
			player.move(move)
#		for player in players:
#			player.set_state(board)
		ply_number += 1
		if args.max_plies != None and ply_number > args.max_plies:
			break

	for player in players:
		player.quit()

	result = board.result()
	if result == None:
		result = "invalid"
	game["result"] = result
	game["end_time"] = time.time()
	game["final_score"] = board.board.count(1), board.board.count(2)

	print_state()
	# Print a final newline to finish the line we're "\r"ing over and over.
	if not args.show_games:
		print()

	return game

def write_game_to_pgn(args, path, game, round_index=1):
	with open(path, "a+") as f:
		print('[Event "?"]', file=f)
		print('[Site "?"]', file=f)
		print('[Date "%s"]' % (datetime.datetime.now().strftime("%Y.%m.%d"),), file=f)
		print('[Round "%i"]' % (round_index,), file=f)
		print('[White "%s"]' % (" ".join(game["white"]),), file=f)
		print('[Black "%s"]' % (" ".join(game["black"]),), file=f)
		print('[Opening "%s"]' % (", ".join(map(uai_interface.uai_encode_move, game["opening"])),), file=f)
		print('[GameStartTime "%s"]' % (datetime.datetime.fromtimestamp(game["start_time"]).isoformat(),), file=f)
		print('[GameEndTime "%s"]' % (datetime.datetime.fromtimestamp(game["end_time"]).isoformat(),), file=f)
		print('[Plycount "%i"]' % (len(game["moves"]),), file=f)
		result_string = {1: "1-0", 2: "0-1", "invalid": "1/2-1/2"}[game["result"]]
		print('[Result "%s"]' % (result_string,), file=f)
		print('[FinalScore "%i-%i"]' % game["final_score"], file=f)
		print('[TimeControl "+%r"]' % (args.tc,), file=f)
		print(file=f)
		print(" ".join(map(uai_interface.uai_encode_move, game["moves"])), file=f)
		print(file=f)
#		for i, m in game["moves"]:
#			if i % 2 == 0:
#				print >>f, "%i. "

def get_opening(args):
	if args.opening != None:
		if not args.opening.strip():
			return []
		return [uai_interface.uai_decode_move(m.strip()) for m in args.opening.split(",")]
	board = ataxx_rules.AtaxxState.initial()
	opening_moves = []
	for _ in range(OPENING_DEPTH):
		move = random.choice(board.legal_moves())
		opening_moves.append(move)
		board.move(move)
	return opening_moves

if __name__ == "__main__":
	import argparse, shlex
	parser = argparse.ArgumentParser()
	parser.add_argument("--engine", metavar="CMD", action="append", help="Engine command.")
	parser.add_argument("--show-games", action="store_true", help="Show the games while they're being generated.")
	parser.add_argument("--opening", metavar="MOVES", type=str, default=None, help="Comma separated sequence of moves for the opening.")
	parser.add_argument("--max-plies", metavar="N", type=int, default=None, help="Maximum number of plies in a game before it's aborted and rejected.")
	parser.add_argument("--pgn-out", metavar="PATH", type=str, default=None, help="PGN file path to accumulate games into. Writes in append mode.")
	parser.add_argument("--gauntlet", action="store_true", help="Just the first engine plays against all the other engines.")
	parser.add_argument("--tc", metavar="SEC", type=float, default=1.0, help="Seconds per move for all engines.")
	args = parser.parse_args()
	print("Options:", args)

	print("Engines:")
	engines = list(map(tuple, list(map(shlex.split, args.engine))))
	for i, engine in enumerate(engines):
		print("%4i: %s" % (i + 1, engine))

	win_counter = {engine: 0 for engine in engines}
	games_written = 0
	games_queue = []
	annulled_games = 0

	while True:
		if not games_queue:
			if args.gauntlet:
				pairings = [(engines[0], eng) for eng in engines[1:]]
			else:
				pairings = list(itertools.combinations(engines, 2))
				def get_model_number(s):
					import re
					assert isinstance(s, list) or isinstance(s, tuple)
					s = " ".join(s)
					return int(re.search("model-([0-9]+)", s).groups()[0])
#				def too_low(s):
#					s = " ".join(s)
#					return any(("model-%03i" % i) in s for i in xrange(38))
				pairings = [
					(a, b)
					for a, b in pairings
					if ("005-pre2" in " ".join(a) or "005-pre2" in " ".join(b) or "ataxx-engine" in " ".join(a) or "ataxx-engine" in " ".join(b)) or
						(abs(get_model_number(a) - get_model_number(b)) <= 10 and (get_model_number(a) > 0 or get_model_number(b) > 0))
				]
			random.shuffle(pairings)
			for pairing in pairings:
				opening = get_opening(args)
				# Append games going both ways.
				games_queue.append((opening, pairing))
				games_queue.append((opening, pairing[::-1]))

		opening, engine_pair = games_queue.pop()

		game = play_one_game(args, engine_pair[0], engine_pair[1], opening_moves=opening)
		if game["result"] in (1, 2):
			winning_engine = engine_pair[game["result"] - 1]
			win_counter[winning_engine] += 1
		else:
			win_counter[engine_pair[0]] += 0.5
			win_counter[engine_pair[1]] += 0.5
			annulled_games += 1
		print("Wins: %s (annulled: %i)" % (
			" - ".join(str(win_counter[eng]) for eng in engines),
			annulled_games,
		))

		if args.pgn_out:
			games_written += 1
			write_game_to_pgn(args, args.pgn_out, game, round_index=games_written)

