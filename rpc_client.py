#!/usr/bin/python

# TODO: XXX: DRY this whole file out!
# I copy pasted a ton of stuff and it's sickening.

import engine
import model
import mprpc
import numpy as np

def setup_rpc(port=6000):
	global rpc_connection
	rpc_connection = mprpc.RPCClient("127.0.0.1", port)

def evaluate(board):
	features = engine.board_to_features(board)
	assert features.dtype == np.int8
	feature_string = features.tostring()
	assert len(feature_string) == model.BOARD_SIZE * model.BOARD_SIZE * model.Network.INPUT_FEATURE_COUNT
	posterior, value = rpc_connection.call("network", feature_string)
	raw_posterior = np.fromstring(posterior, dtype=np.float32).reshape((model.BOARD_SIZE, model.BOARD_SIZE, model.MOVE_TYPES))

	softmax_posterior = engine.softmax(raw_posterior)
	posterior = {move: engine.get_move_score(softmax_posterior, move) for move in board.legal_moves()}
	# Renormalize the posterior. Add a small epsilon into the denominator to prevent divison by zero.
	denominator = sum(posterior.itervalues()) + 1e-6
	posterior = {move: prob / denominator for move, prob in posterior.iteritems()}

	return posterior, value

class RPCEvaluator:
	def populate(self, board):
		# XXX: This is ugly...
		if hasattr(board, "evaluations"):
			return

		# Get an RPC value and posterior.
		posterior, value = evaluate(board)

		entry = engine.NNEvaluator.Entry(board=board, value=value, posterior=posterior, game_over=False)
		# Evaluate special value adjustments.
		result = board.result()
		if result != None:
			assert result in (1, 2) and board.to_move in (1, 2)
			entry.value = 1.0 if result == board.to_move else -1.0
			entry.game_over = True
		board.evaluations = entry

	def add_to_queue(self, board):
		pass

if __name__ == "__main__":
	import ataxx_rules
	setup_rpc()
	evaluator = RPCEvaluator()
	board = ataxx_rules.AtaxxState.initial()
	evaluator.populate(board)
	print board.evaluations.posterior, board.evaluations.value

#	array = np.zeros((7, 7, 4), dtype=np.int8)
#	evaluate(array)

