#!/usr/bin/python

import os, time, random, json, argparse, pprint
import ataxx_rules
import engine
import train
import uai_ringmaster

#STEP_COUNT = 200
VISIT_COUNT = 100
MAX_STEP_COUNT = 1000
TEMPERATURE = 0.05
TOTALLY_RANDOM_PROB = 0.05

def generate_game(args):
	board = ataxx_rules.AtaxxState.initial()
	if not args.random_play:
		m = engine.MCTS(board.copy())
	entry = {"boards": [], "moves": []}
	plies = 0
	all_steps = 0
	while True:
		if args.random_play:
			best_move = selected_move = random.choice(board.legal_moves())
		else:
			if args.supervised:
				args.uai_player.set_state(board)
				try:
					best_move = selected_move = args.uai_player.genmove(args.supervised_ms)
				except ValueError, e:
					print "Board state:"
					print board.fen()
					print board
					raise e
			else:
				# Do steps until the root is sufficiently visited.
				most_visits = 0 
#				while m.root_node.all_edge_visits < STEP_COUNT:
				while most_visits < VISIT_COUNT and m.root_node.all_edge_visits < MAX_STEP_COUNT:
					all_steps += 1
					edge = m.step()
					most_visits = max(most_visits, edge.edge_visits)
				# Pick a move without noise for training.
				best_move = max(
					m.root_node.outgoing_edges.itervalues(),
					key=lambda edge: edge.edge_visits,
				).move
				# Pick a move with noise.
				scores = {}
				for move in m.root_node.board.legal_moves():
					scores[move] = 0.0
					if move in m.root_node.outgoing_edges:
						edge = m.root_node.outgoing_edges[move]
						scores[move] = edge.edge_visits / float(m.root_node.all_edge_visits)
					scores[move] += random.normalvariate(0, TEMPERATURE)
				selected_move = max(scores.iterkeys(), key=lambda move: scores[move])
			prob = TOTALLY_RANDOM_PROB
			if plies < 8:
				prob = 0.25
			if random.random() < prob:
				selected_move = random.choice(board.legal_moves())
		entry["boards"].append(list(board.board[:]))
		entry["moves"].append(best_move)
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
	print "[%3i] Generated a %i ply game (%.2f avg steps) with result %i." % (args.group_index, len(entry["boards"]), all_steps / float(plies), entry["result"])
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
	parser.add_argument("--no-write", action="store_true", help="Don't write out generated games at all.")
	parser.add_argument("--supervised", metavar="CMD", default=None, type=str, help="Command for a UAI engine.")
	parser.add_argument("--supervised-ms", metavar="N", default=100, type=int, help="Number of milliseconds per move for supervised generation.")
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

	if args.supervised != None:
		args.uai_player = uai_ringmaster.UAIPlayer(args.supervised)

	output_directory = os.path.join("games", network_name)
	if not os.path.exists(output_directory):
		os.mkdir(output_directory)
	output_path = os.path.join(output_directory, os.urandom(8).encode("hex") + ".json")
	if args.no_write:
		output_path = "/dev/null"
	print "[%3i] Writing to: %s" % (args.group_index, output_path)

	with open(output_path, "w") as f:
		games_generated = 0
		while True:
			entry = generate_game(args)
			json.dump(entry, f)
			f.write("\n")
			f.flush()
			games_generated += 1
			if args.game_count != None and games_generated >= args.game_count:
				print "Done generating games."
				break

