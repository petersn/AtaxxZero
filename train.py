#!/usr/bin/python

import os, glob, json, random
import tensorflow as tf
import numpy as np
import ataxx_rules
import uai_interface
import engine
import model

def apply_symmetry(index, arr):
	assert len(arr.shape) == 3 and arr.shape[:2] == (model.BOARD_SIZE, model.BOARD_SIZE)
	assert index in xrange(8)
	coin1, coin2, coin3 = index & 1, (index >> 1) & 1, (index >> 2) & 1
	# Break views to avoid mutating our input.
	arr = np.array(arr).copy()
	if coin1:
		arr = arr[::-1,:,:].copy()
	if coin2:
		arr = arr[:,::-1,:].copy()
	if coin3:
		arr = np.swapaxes(arr.copy(), 0, 1).copy()
	return arr

def apply_symmetry_to_move(index, move):
	assert index in xrange(8)
	coin1, coin2, coin3 = index & 1, (index >> 1) & 1, (index >> 2) & 1
	def apply_to_coord(xy):
		x, y = xy
		if coin1:
			x = (model.BOARD_SIZE - 1) - x
		if coin2:
			y = (model.BOARD_SIZE - 1) - y
		if coin3:
			x, y = y, x
		return x, y
	start, end = move
	if start == "c":
		return "c", apply_to_coord(end)
	return apply_to_coord(start), apply_to_coord(end)

# WARNING: Loops infinitely if there are no games with no non-passing moves.
def get_sample_from_entries(entries):
	while True:
		entry = random.choice(entries)
		ply = random.randrange(len(entry["boards"]))
		to_move = 1 if ply % 2 == 0 else 2
		board = ataxx_rules.AtaxxState(entry["boards"][ply], to_move=to_move).copy()
		move  = entry["moves"][ply]
		if move == "pass":
			continue
		if isinstance(move, (str, unicode)):
			move = uai_interface.uai_decode_move(move)
		# Convert the board into encoded features.
		features = engine.board_to_features(board)
		desired_value = [1 if entry["result"] == to_move else -1]
		# Apply a dihedral symmetry.
		symmetry_index = random.randrange(8)
		features = apply_symmetry(symmetry_index, features)
		move = apply_symmetry_to_move(symmetry_index, move)
		desired_policy = engine.encode_move_as_heatmap(move)
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
	random.shuffle(entries)
	return entries

if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument("--games", metavar="PATH", required=True, nargs="+", help="Path to .json self-play games files.")
	parser.add_argument("--old-path", metavar="PATH", help="Path for input network.")
	parser.add_argument("--new-path", metavar="PATH", required=True, help="Path for output network.")
	parser.add_argument("--steps", metavar="COUNT", type=int, default=1000, help="Training steps.")
	parser.add_argument("--minibatch-size", metavar="COUNT", type=int, default=512, help="Minibatch size.")
	parser.add_argument("--learning-rate", metavar="LR", type=float, default=0.002, help="Learning rate.")
	args = parser.parse_args()
	print "Arguments:", args

	# Shuffle the loaded games deterministically.
	random.seed(123456789)
	entries = load_entries(args.games)
	ply_count = sum(len(entry["moves"]) for entry in entries)
	print "Found %i games with %i plies." % (len(entries), ply_count)

	test_entries = entries[:10]
	train_entries = entries[10:]

	network = model.Network("training_net/", build_training=True)
	sess = tf.InteractiveSession()
	sess.run(tf.initialize_all_variables())
	model.sess = sess

	if args.old_path != None:
		print "Loading old model."
		model.load_model(network, args.old_path)
	else:
		print "WARNING: Not loading a previous model!"

	def make_minibatch(entries, size):
		batch = {"features": [], "policies": [], "values": []}
		for _ in xrange(size):
			feature, policy, value = get_sample_from_entries(entries)
			batch["features"].append(feature)
			batch["policies"].append(policy)
			batch["values"].append(value)
		return batch

	# Choose the test set deterministically.
	random.seed(123456789)
	in_sample_val_set = make_minibatch(test_entries, 2048)

	print
	print "Model dimensions: %i filters, %i blocks, %i parameters." % (model.Network.FILTERS, model.Network.BLOCK_COUNT, network.total_parameters)
	print "Have %i augmented samples, and sampling %i in total." % (ply_count * 8, args.steps * args.minibatch_size)
	print "=== BEGINNING TRAINING ==="

	# Begin training.
	for step_number in xrange(args.steps):
		if step_number % 100 == 0:
			policy_loss = network.run_on_samples(network.policy_loss.eval, in_sample_val_set)
			value_loss  = network.run_on_samples(network.value_loss.eval, in_sample_val_set)
#			loss = network.get_loss(in_sample_val_set)
			print "Step: %4i -- loss: %.6f  (policy: %.6f  value: %.6f)" % (
				step_number,
				policy_loss + value_loss,
				policy_loss,
				value_loss,
			)
		minibatch = make_minibatch(train_entries, args.minibatch_size)
		network.train(minibatch, learning_rate=args.learning_rate)

	# Write out the trained model.
	model.save_model(network, args.new_path)

