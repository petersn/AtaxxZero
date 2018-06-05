#!/usr/bin/python

import os, time, random, json, argparse, pprint, logging
import ataxx_rules
import engine

logging.basicConfig(
	format="[%(process)5d] %(message)s",
	level=logging.DEBUG,
)

def generate_game(args):
	board = ataxx_rules.AtaxxState.initial()
	e = engine.MCTSEngine()
	while True:
		print board
		selected_move = e.genmove(1.0, early_out=False, use_weighted_exponent=5.0)
		board.move(selected_move)
		e.set_state(board.copy())
		if board.result() != None:
			break
		raw_input(">")

if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument("--network", metavar="PATH", required=True, help="Path of the model to load.")
	args = parser.parse_args()

	engine.setup_evaluator(use_rpc=False)
	engine.initialize_model(args.network)

	generate_game(args)

