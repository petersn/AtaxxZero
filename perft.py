#!/usr/bin/python

import ataxx_rules

def perft(position, depth):
	ensemble = [position.copy()]
	for _ in xrange(depth):
		new_ensemble = []
		for b in ensemble:
			for move in b.legal_moves():
				new_board = b.copy()
				new_board.move(move)
				new_ensemble.append(new_board)
		ensemble = new_ensemble
	print "Size:", len(ensemble)

board = ataxx_rules.AtaxxState.initial()
for move in board.legal_moves():
	copy = board.copy()
	copy.move(move)
	print "Move:", move, 
	perft(copy, 4)

