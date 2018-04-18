#!/usr/bin/python

import sys, random, string, subprocess, atexit
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

def play_one_game(args, opening_moves, swap=False):
	players = [
		UAIPlayer(args.command1),
		UAIPlayer(args.command2),
	]
	if swap:
		players.reverse()

	print "Launching with opening: [%s]" % (", ".join(uai_interface.uai_encode_move(move) for move in opening_moves),)
	board = ataxx_rules.AtaxxState.initial()
	ply_number = 0
	while board.result() == None:
		if args.show_games:
			print
			print "===================== Player %i move." % (ply_number % 2 + 1,)
			print "[%i plies] Score: %i - %i" % (ply_number, board.board.count(1), board.board.count(2))
			print board.fen()
			print board
		else:
			print "\r[%i plies] Score: %i - %i" % (ply_number, board.board.count(1), board.board.count(2)),
			sys.stdout.flush()
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
		for player in players:
			player.move(move)
#		for player in players:
#			player.set_state(board)
		ply_number += 1
		if args.max_plies != None and ply_number > args.max_plies:
			return "invalid"

	for player in players:
		player.send("quit\n")

	if not args.show_games:
		print

	result = board.result()
	if swap:
		result = {1: 2, 2: 1}[result]
	return result

if __name__ == "__main__":
	import argparse, shlex
	parser = argparse.ArgumentParser()
	parser.add_argument("--command1", metavar="CMD")
	parser.add_argument("--command2", metavar="CMD")
	parser.add_argument("--show-games", action="store_true", help="Show the games while they're being generated.")
	parser.add_argument("--opening", metavar="MOVES", type=str, default=None, help="Comma separated sequence of moves for the opening.")
	parser.add_argument("--max-plies", metavar="N", type=int, default=None, help="Maximum number of plies in a game before it's aborted and rejected.")
	args = parser.parse_args()
	args.command1 = shlex.split(args.command1)
	args.command2 = shlex.split(args.command2)
	print "Options:", args

	win_counter = {1: 0, 2: 0, "invalid": 0}

	swap = False
	while True:
		if not swap:
			opening = ataxx_rules.AtaxxState.initial()
			opening_moves = []
			for _ in xrange(OPENING_DEPTH):
				move = random.choice(opening.legal_moves())
				opening_moves.append(move)
				opening.move(move)
			if args.opening != None:
				opening_moves = [uai_interface.uai_decode_move(m.strip()) for m in args.opening.split(",")]

		outcome = play_one_game(args, opening_moves=opening_moves, swap=swap)
		swap = not swap
		win_counter[outcome] += 1
		print "Wins: %i - %i (invalid: %i) (%s - %s)" % (win_counter[1], win_counter[2], win_counter["invalid"], args.command1, args.command2)

