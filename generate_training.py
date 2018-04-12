#!/usr/bin/python

import os, time, random, json, argparse
import ataxx_rules
import engine
import train

STEP_COUNT = 100
TEMPERATURE = 0.1
TOTALLY_RANDOM_PROB = 0.005

def generate_game(args):
	board = ataxx_rules.AtaxxState.initial()
	if not args.random_play:
		m = engine.MCTS(board)
	entry = {"boards": [], "moves": []}
	plies = 0
	while True:
		if args.random_play or random.random() < TOTALLY_RANDOM_PROB:
			selected_move = random.choice(board.legal_moves())
		else:
			# Do steps until the root is sufficiently visited.
			while m.root_node.all_edge_visits < STEP_COUNT:
				m.step()
			# Pick a move with noise.
			scores = {}
			for move in m.root_node.board.legal_moves():
				scores[move] = 0.0
				if move in m.root_node.outgoing_edges:
					edge = m.root_node.outgoing_edges[move]
					scores[move] = edge.edge_visits / float(m.root_node.all_edge_visits)
				scores[move] += random.normalvariate(0, TEMPERATURE)
			selected_move = max(scores.iterkeys(), key=lambda move: scores[move])
		entry["boards"].append(list(board.board[:]))
		entry["moves"].append(selected_move)
		# Execute the move.
		plies += 1
		if not args.random_play:
			m.play(board.to_move, selected_move)
		board.move(selected_move)
		if board.result() != None:
			break
		if args.show_game:
			print board
			raw_input(">")
		if args.die_if_present and os.path.exists(args.die_if_present):
			print "Exiting due to signal file!"
			exit()
	entry["result"] = board.result()
	return entry

if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument("--network", metavar="NAME", required=True, help="Name of the model to load.")
	parser.add_argument("--group-index", metavar="N", default=0, type=int, help="Our index in the work group.")
	parser.add_argument("--use-rpc", action="store_true", help="Use RPC for NN evaluation.")
	parser.add_argument("--random-play", action="store_true", help="Generate games by totally random play.")
	parser.add_argument("--die-if-present", metavar="PATH", default=None, type=str, help="Die once a file is present at the target path.")
	parser.add_argument("--show-game", action="store_true", help="Show the game while it's generating.")
	parser.add_argument("--game-count", metavar="N", default=None, type=int, help="Maximum number of games to generate.")
	args = parser.parse_args()

	network_path = train.model_path(args.network)
	network_name = args.network

	if args.random_play:
		print "Doing random play! Loading no model, and not using RPC."
	elif args.use_rpc:
		engine.setup_evaluator(use_rpc=True)
	else:
		engine.setup_evaluator(use_rpc=False)
		engine.initialize_model(network_path)

	output_directory = os.path.join("games", network_name)
	if not os.path.exists(output_directory):
		os.mkdir(output_directory)
	output_path = os.path.join(output_directory, os.urandom(8).encode("hex") + ".json")
	print "[%3i] Writing to: %s" % (args.group_index, output_path)

	with open(output_path, "w") as f:
		games_generated = 0
		while True:
			entry = generate_game(args)
			print "[%3i] Generated a %i ply game with result %i." % (args.group_index, len(entry["boards"]), entry["result"])
			json.dump(entry, f)
			f.write("\n")
			f.flush()
			games_generated += 1
			if args.game_count != None and games_generated >= args.game_count:
				print "Done generating games."
				break

