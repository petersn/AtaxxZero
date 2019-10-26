#!/usr/bin/python3

import itertools, array
import numpy as np
import ataxx_rules
import engine

def apply_symmetry(tensor, symmetry):
	assert 0 <= symmetry < 8
	if symmetry & 1:
		tensor = tensor[::-1, :]
	if symmetry & 2:
		tensor = tensor[:, ::-1]
	if symmetry & 4:
		tensor = np.moveaxis(tensor, 0, 1)
	return tensor

# To recompute this table if I change the above:
# 	x = np.random.randn(2, 3, 4)
# 	{
# 		i: [
# 			np.all(x == apply_symmetry(apply_symmetry(x, i), j))
# 			for j in range(8)
# 		].index(True)
# 		for i in range(8)
# 	}
inverse_symmetry = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 6, 6: 5, 7: 7}

# All symmetries of the opening position.
# Compute with:
# 	b = ataxx_rules.AtaxxState.initial()
# 	[i for i in range(8) if symmetry_canonicalize(b, [i]) == b]
opening_symmetries = [0, 3, 4, 7]

def symmetry_canonicalize(base_board, allowed_symmetries):
	tensor = np.array(base_board.board, dtype=np.int64).reshape((7, 7))
	return min(
		(
			ataxx_rules.AtaxxState(
				board=array.array("b", apply_symmetry(tensor, symmetry).flatten()),
				to_move=base_board.to_move,
			)
			for symmetry in allowed_symmetries
		),
		key=lambda b: tuple(b.board),
	)

def evaluate(board):
	features = engine.board_to_features(board)
	all_features = [apply_symmetry(features, symmetry) for symmetry in range(8)]
	policy, value = engine.sess.run(
		[engine.network.policy_output, engine.network.value_output],
		feed_dict={
			engine.network.input_ph: all_features,
			engine.network.is_training_ph: False,
		},
	)
	policy = [
		apply_symmetry(x, inverse_symmetry[i])
		for i, x in enumerate(policy)
	]
	return np.mean(policy, axis=0), np.mean(value)

def load_default_model():
	engine.model.Network.FILTERS = 128
	engine.model.Network.BLOCK_COUNT = 12
	engine.setup_evaluator(use_rpc=False)
	engine.initialize_model("/tmp/model-156.npy")

if __name__ == "__main__":
	load_default_model()
	board = ataxx_rules.AtaxxState.initial()
	policies, values = evaluate(board)

