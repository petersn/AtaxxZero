#!/usr/bin/python

import os, time, random, json, argparse
import ataxx_rules
import engine

STEP_COUNT = 100
TEMPERATURE = 0.1

def generate_game():
	board = ataxx_rules.AtaxxState.initial()
	m = engine.MCTS(board)
	entry = {"boards": [], "moves": []}
	plies = 0
	while True:
		# Do steps until the root is sufficiently visited.
		while m.root_node.all_edge_visits < STEP_COUNT:
			m.step()
		# Pick a move with noise.
		scores = {
			edge: (edge.edge_visits / m.root_node.all_edge_visits) + random.normalvariate(0, TEMPERATURE)
			for edge in m.root_node.outgoing_edges.itervalues()
		}
		edge = max(scores.iterkeys(), key=lambda edge: scores[edge])
		entry["boards"].append(list(board.board[:]))
		entry["moves"].append(edge.move)
		# Execute the move.
		plies += 1
		m.play(board.to_move, edge.move)
		board.move(edge.move)
		if board.result() != None:
			break
	entry["result"] = board.result()
	return entry

if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument("--network", metavar="PATH", required=True, help="Network to generate training games with.")
	args = parser.parse_args()

	network_path = args.network
	network_name = os.path.split(network_path)[1]

	engine.initialize_model(network_path)
	output_directory = os.path.join("games", network_name)
	if not os.path.exists(output_directory):
		os.mkdir(output_directory)
	output_path = os.path.join(output_directory, os.urandom(8).encode("hex") + ".json")
	print "Writing to:", output_path

	with open(output_path, "w") as f:
		while True:
			entry = generate_game()
			print "Generated a %i ply game with result %i." % (len(entry["boards"]), entry["result"])
			json.dump(entry, f)
			f.write("\n")
			f.flush()

