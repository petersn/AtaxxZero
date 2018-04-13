#!/usr/bin/python

import os, time, random, json, argparse, pprint, logging
import ataxx_rules
import engine
import train

logging.basicConfig(
	format="[%(process)5d] %(message)s",
	level=logging.DEBUG,
)

def generate_game(args):
	board = ataxx_rules.AtaxxState.initial()
	e = engine.MCTSEngine()
	while True:
		print board
		selected_move = e.genmove(0.7, early_out=False)
		board.move(selected_move)
		e.set_state(board.copy())
		if board.result() != None:
			break
		raw_input(">")

if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument("--network", metavar="NAME", required=True, help="Name of the model to load.")
	args = parser.parse_args()

	network_path = train.model_path(args.network)
	network_name = args.network

	engine.setup_evaluator(use_rpc=False)
	engine.initialize_model(network_path)

	generate_game(args)

