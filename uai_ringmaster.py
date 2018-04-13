#!/usr/bin/python

import string, subprocess, atexit
import uai_interface
import ataxx_rules

class UAIPlayer:
	def __init__(self, cmd):
		self.cmd = cmd
		# WARNING: What I'm doing here is technically invalid!
		# In theory the output pipe of this process could fill up before I read, making it hang.
		# TODO: Decide if I care.
		self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stdin=subprocess.PIPE)
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
		if ("uai_interface" in " ".join(self.cmd)) or True:
			print "More time alotted."
			time *= 5
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

def play_one_game(args, swap=False):
	players = [
		UAIPlayer(args.command1),
		UAIPlayer(args.command2),
	]
	if swap:
		players.reverse()

	print "Launching."
	board = ataxx_rules.AtaxxState.initial()
	ply_number = 0
	while board.result() == None:
		print
		print "===================== Player %i move." % (ply_number % 2 + 1,)
		print "Score: %i - %i" % (board.board.count(1), board.board.count(2))
		print board.fen()
		print board
		# If there is only one legal move then force it.
		if len(board.legal_moves()) == 1:
			move, = board.legal_moves()
		else:
			move = players[ply_number % 2].genmove()
		print "Move:", uai_interface.uai_encode_move(move)
		board.move(move)
		for player in players:
			player.move(move)
		ply_number += 1

	for player in players:
		player.send("quit\n")

	result = board.result()
	if swap:
		result = {1: 2, 2: 1}[result]
	return result

if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument("--command1", metavar="CMD", nargs="+")
	parser.add_argument("--command2", metavar="CMD", nargs="+")
	args = parser.parse_args()
	print "Options:", args

	win_counter = {1: 0, 2: 0}

	swap = False
	while True:
		outcome = play_one_game(args, swap=swap)
		swap = not swap
		win_counter[outcome] += 1
		print "Wins: %i - %i (%s - %s)" % (win_counter[1], win_counter[2], args.command1, args.command2)

