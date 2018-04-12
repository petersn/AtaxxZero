#!/usr/bin/python

import os, glob, json, random
import tensorflow as tf
import numpy as np
import ataxx_rules
import engine
import model

def apply_symmetry(index, arr):
	assert len(arr.shape) == 3 and arr.shape[:2] == (model.BOARD_SIZE, model.BOARD_SIZE)
	assert index in xrange(8)
	coin1, coin2, coin3 = index & 1, (index >> 1) & 1, (index >> 2) & 1
	# Break views to avoid mutating our input.
	arr = np.array(arr)
	if coin1:
		arr = arr[::-1,:,:]
	if coin2:
		arr = arr[:,::-1,:]
	if coin3:
		arr = np.swapaxes(arr, 0, 1)
	return arr

# WARNING: Loops infinitely if there are no games with no non-passing moves.
def get_sample_from_entries(entries):
	while True:
		entry = random.choice(entries)
		ply = random.randrange(len(entry["boards"]))
		to_move = 1 if ply % 2 == 0 else 2
		board = ataxx_rules.AtaxxState(entry["boards"][ply], to_move=to_move)
		move  = entry["moves"][ply]
		if move == "pass":
			continue
		# Convert the board into encoded features.
		features = engine.board_to_features(board)
		desired_policy = engine.encode_move_as_heatmap(move)
		desired_value = [1 if entry["result"] == to_move else -1]
		# TODO: Add in random dihedral symmetry here.
		symmetry_index = random.randrange(8)
		features       = apply_symmetry(symmetry_index, features)
		desired_policy = apply_symmetry(symmetry_index, desired_policy)
		return features, desired_policy, desired_value

def load_entries(paths):
	entries = []
	for path in paths:
		with open(path) as f:
			for line in f:
				line = line.strip()
				if not line:
					continue
				entries.append(json.loads(line))
	return entries

def model_path(name):
	return os.path.join("models", name + ".npy")

if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument("--games", metavar="PATH", required=True, help="Directory with .json self-play games.")
	parser.add_argument("--old-name", metavar="NAME", help="Name for input network.")
	parser.add_argument("--new-name", metavar="NAME", required=True, help="Name for output network.")
	parser.add_argument("--steps", metavar="COUNT", type=int, default=1000, help="Training steps.")
	parser.add_argument("--minibatch-size", metavar="COUNT", type=int, default=512, help="Minibatch size.")
	parser.add_argument("--learning-rate", metavar="LR", type=float, default=0.005, help="Learning rate.")
	args = parser.parse_args()
	print "Arguments:", args

	paths = glob.glob(os.path.join(args.games, "*.json"))
	entries = load_entries(paths)
	ply_count = sum(len(entry["moves"]) for entry in entries)
	print "Found %i games with %i plies." % (len(entries), ply_count)

	network = model.Network("training_net/", build_training=True)
	sess = tf.InteractiveSession()
	sess.run(tf.initialize_all_variables())
	model.sess = sess

	if args.old_name != None:
		print "Loading old model."
		model.load_model(network, model_path(args.old_name))
	else:
		print "WARNING: Not loading a previous model!"

	def make_minibatch(size):
		batch = {"features": [], "policies": [], "values": []}
		for _ in xrange(size):
			feature, policy, value = get_sample_from_entries(entries)
			batch["features"].append(feature)
			batch["policies"].append(policy)
			batch["values"].append(value)
		return batch

	in_sample_val_set = make_minibatch(1024)

	print
	print "Have %i augmented samples, and sampling %i in total." % (ply_count * 8, args.steps * args.minibatch_size)
	print "=== BEGINNING TRAINING ==="

	# Begin training.
	for step_number in xrange(args.steps):
		if step_number % 50 == 0:
			loss = network.get_loss(in_sample_val_set)
			print "Step: %4i -- loss: %.6f" % (step_number, loss)
		minibatch = make_minibatch(args.minibatch_size)
		network.train(minibatch, learning_rate=args.learning_rate)

	# Write out the trained model.
	model.save_model(network, model_path(args.new_name))

