#!/usr/bin/python

import ataxx_rules

def perft(position, depth):
	ensemble = [position.copy()]
	for _ in range(depth):
		new_ensemble = []
		for b in ensemble:
			for move in b.legal_moves():
				new_board = b.copy()
				new_board.move(move)
				new_ensemble.append(new_board)
		ensemble = new_ensemble
	print("Size:", len(ensemble))
	return len(ensemble)

board = ataxx_rules.AtaxxState.initial()
total = 0
for move in board.legal_moves():
	copy = board.copy()
	copy.move(move)
	print("Move:", move, end=' ') 
	total += perft(copy, 3)

print("Total:", total)

