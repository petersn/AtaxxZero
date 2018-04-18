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
		atexit.register(self.proc.kill)
		self.send("uai\n")

	def send(self, s):
		self.proc.stdin.write(s)
		self.proc.stdin.flush()

	def reset(self):
		self.send("uainewgame\n")

	def set_state(self, board):
		self.send("position fen %s\n" % board.fen())

	def move(self, move):
		self.send("moves %s\n" % uai_interface.uai_encode_move(move))

	def genmove(self, time=1000):
#		if ("uai_interface" in " ".join(self.cmd)) or True:
#			print "More time alotted."
#			time *= 5
		self.send("go movetime %i\n" % time)
		while True:
			line = self.proc.stdout.readline().strip()
			if not line:
				raise Exception("Bad UAI!")
			if line.startswith("bestmove "):
				return uai_interface.uai_decode_move(line[9:])

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
	print 'Game: "%s" vs "%s" with opening: [%s]' % (
		" ".join(engine1),
		" ".join(engine2),
		", ".join(uai_interface.uai_encode_move(move) for move in opening_moves),
	)
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
			print
			print "===================== Player %i move." % (ply_number % 2 + 1,)
			print "[%3i plies] Score: %2i - %2i" % (ply_number, board.board.count(1), board.board.count(2))
			print board.fen()
			print board
		else:
			print "\r[%3i plies] Score: %2i - %2i " % (ply_number, board.board.count(1), board.board.count(2)),
			sys.stdout.flush()

	while board.result() == None:
		print_state()
		# If there is only one legal move then force it.
		if ply_number < len(opening_moves):
			move = opening_moves[ply_number]
		elif len(board.legal_moves()) == 1:
			move, = board.legal_moves()
		else:
			move = players[ply_number % 2].genmove()
		if args.show_games:
			print "Move:", uai_interface.uai_encode_move(move)
		board.move(move)
		game["moves"].append(move)
		for player in players:
			player.move(move)
#		for player in players:
#			player.set_state(board)
		ply_number += 1
		if args.max_plies != None and ply_number > args.max_plies:
			break

	for player in players:
		player.send("quit\n")

	result = board.result()
	if result == None:
		result = "invalid"
	game["result"] = result
	game["end_time"] = time.time()
	game["final_score"] = board.board.count(1), board.board.count(2)

	print_state()
	# Print a final newline to finish the line we're "\r"ing over and over.
	if not args.show_games:
		print

	return game

def write_game_to_pgn(path, game, round_index=1):
	with open(path, "a+") as f:
		print >>f, '[Event "?"]'
		print >>f, '[Site "?"]'
		print >>f, '[Date "%s"]' % (datetime.datetime.now().strftime("%Y.%m.%d"),)
		print >>f, '[Round "%i"]' % (round_index,)
		print >>f, '[White "%s"]' % (" ".join(game["white"]),)
		print >>f, '[Black "%s"]' % (" ".join(game["black"]),)
		print >>f, '[Opening "%s"]' % (", ".join(map(uai_interface.uai_encode_move, game["opening"])),)
		print >>f, '[GameStartTime "%s"]' % (datetime.datetime.fromtimestamp(game["start_time"]).isoformat(),)
		print >>f, '[GameEndTime "%s"]' % (datetime.datetime.fromtimestamp(game["end_time"]).isoformat(),)
		print >>f, '[Plycount "%i"]' % (len(game["moves"]),)
		result_string = {1: "1-0", 2: "0-1", "invalid": "1/2-1/2"}[game["result"]]
		print >>f, '[Result "%s"]' % (result_string,)
		print >>f, '[FinalScore "%i-%i"]' % game["final_score"]
		print >>f
		print >>f, " ".join(map(uai_interface.uai_encode_move, game["moves"]))
		print >>f
#		for i, m in game["moves"]:
#			if i % 2 == 0:
#				print >>f, "%i. "

def get_opening(args):
	if args.opening != None:
		return [uai_interface.uai_decode_move(m.strip()) for m in args.opening.split(",")]
	board = ataxx_rules.AtaxxState.initial()
	opening_moves = []
	for _ in xrange(OPENING_DEPTH):
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
	args = parser.parse_args()
	print "Options:", args

	print "Engines:"
	engines = map(tuple, map(shlex.split, args.engine))
	for i, engine in enumerate(engines):
		print "%4i: %s" % (i + 1, engine)

	win_counter = {engine: 0 for engine in engines}
	games_written = 0
	games_queue = []
	annulled_games = 0

	while True:
		if not games_queue:
			pairings = list(itertools.combinations(engines, 2))
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
#			annulled_games += 1
		print "Wins: %s (annulled: %i)" % (
			" - ".join(str(win_counter[eng]) for eng in engines),
			annulled_games,
		)

		if args.pgn_out:
			games_written += 1
			write_game_to_pgn(args.pgn_out, game, round_index=games_written)

