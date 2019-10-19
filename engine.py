#!/usr/bin/python

import logging, time, sys, random, collections
import numpy as np
import tensorflow as tf
import ataxx_rules
import model
import uai_interface

RED  = "\x1b[91m"
ENDC = "\x1b[0m"
DIRICHLET_ALPHA  = 0.15
DIRICHLET_WEIGHT = 0.25

initialized = False

def initialize_model(path):
	global network, sess, initialized
	assert not initialized
	network = model.Network("net/", build_training=False)
	sess = tf.InteractiveSession()
	sess.run(tf.initialize_all_variables())
	model.sess = sess
	model.load_model(network, path)
	initialized = True

def setup_evaluator(use_rpc=False, temperature=0.0):
	global global_evaluator
	if use_rpc:
		print("Using RPC evaluator.")
		import rpc_client
		rpc_client.setup_rpc()
		global_evaluator = rpc_client.RPCEvaluator(temperature=temperature)
	else:
		global_evaluator = NNEvaluator(temperature=temperature)

def sample_by_weight(weights):
	assert abs(sum(weights.values()) - 1) < 1e-6, "Distribution not normalized: %r" % (weights,)
	x = random.random()
	for outcome, weight in weights.items():
		if x <= weight:
			return outcome
		x -= weight
	# If we somehow failed to pick anyone due to rounding then return an arbitrary element.
	return next(iter(weights.keys()))

def softmax(logits):
	"""Somewhat numerically stable softmax routine."""
	e_x = np.exp(logits - np.max(logits))
	return e_x / e_x.sum()

def board_to_features(board):
	features = np.zeros(
		(model.BOARD_SIZE, model.BOARD_SIZE, model.Network.INPUT_FEATURE_COUNT),
		dtype=np.int8,
	)
	for y in range(model.BOARD_SIZE):
		for x in range(model.BOARD_SIZE):
			# Fill layer 0 will ones, to help the convolution out.
			features[x, y, 0] = 1
			# Put the piece into a layer 1 if it's of the player to move, and otherwise layer 2.
			piece = board[x, y]
			layer_index = {
				0: 0,
				board.to_move: 1,
				ataxx_rules.OTHER_PLAYER[board.to_move]: 2,
			}[piece]
			features[x, y, layer_index] = 1
			# Finally, fill layer 3 with a 1 where blocked cells are.
			if (x, y) in ataxx_rules.BLOCKED_CELLS:
				features[x, y, 3] = 1
	return features

position_delta_layers = {delta: i for i, delta in enumerate(ataxx_rules.FAR_NEIGHBOR_OFFSETS)}
assert len(position_delta_layers) == 16
FAR_NEIGHBOR_OFFSETS_SET = frozenset(ataxx_rules.FAR_NEIGHBOR_OFFSETS)

def add_move_to_heatmap(heatmap, move, coef=1):
	# TODO: DRY this with the below code.
	start, end = move
	if start == "c":
		heatmap[end[0], end[1], model.MOVE_TYPES - 1] += coef
	else:
		delta = end[0] - start[0], end[1] - start[1]
		layer = position_delta_layers[delta]
		heatmap[end[0], end[1], layer] += coef

def encode_move_as_heatmap(move):
	heatmap = np.zeros(
		(model.BOARD_SIZE, model.BOARD_SIZE, model.MOVE_TYPES),
		dtype=np.int8,
	)
	add_move_to_heatmap(heatmap, move)
	assert heatmap.sum() == 1
	return heatmap

def get_move_score(softmaxed_posterior, move):
	assert softmaxed_posterior.shape == (7, 7, 17)
	if move == "pass":
		return 1.0
	# TODO: DRY this with the above code.
	start, end = move
	# Clone moves are handled separately.
	if start == "c":
		return softmaxed_posterior[end[0], end[1], model.MOVE_TYPES - 1]
	else:
		delta = end[0] - start[0], end[1] - start[1]
		layer = position_delta_layers[delta]
		return softmaxed_posterior[end[0], end[1], layer]

def add_noise_to_logits(raw_posterior, temperature):
	assert raw_posterior.shape == (model.BOARD_SIZE, model.BOARD_SIZE, model.MOVE_TYPES)
	noise = np.random.randn(model.BOARD_SIZE, model.BOARD_SIZE, model.MOVE_TYPES) * temperature
	return raw_posterior + noise

def add_dirichlet_noise_to_posterior(posterior, alpha, weight):
	noise = np.random.dirichlet([alpha] * len(posterior))
	posterior = {
		move: (1.0 - weight) * prob + weight * n
		for (move, prob), n in zip(iter(posterior.items()), noise)
	}
	# TODO: Comment this out.
	assert abs(sum(posterior.values()) - 1) < 1e-3
	return posterior

class NNEvaluator:
	ENSEMBLE_SIZE = 16
	QUEUE_DEPTH = 4096
	PROBABILITY_THRESHOLD = 0.09
	MAXIMUM_CACHE_ENTRIES = 200000

	class Entry:
		__slots__ = ["board", "value", "posterior", "noisy_posterior", "game_over"]

		def __init__(self, board, value, posterior, game_over):
			self.board = board
			self.value = value
			self.posterior = posterior
			self.noisy_posterior = None
			self.game_over = game_over

		def populate_noisy_posterior(self):
			if self.noisy_posterior is None:
				self.noisy_posterior = add_dirichlet_noise_to_posterior(self.posterior, DIRICHLET_ALPHA, DIRICHLET_WEIGHT)

	def __init__(self, temperature):
		self.temperature = temperature
		self.cache = {}
		self.board_queue = collections.deque(maxlen=NNEvaluator.QUEUE_DEPTH)
		self.ensemble_sizes = []

	def __repr__(self):
		return "<NNEvaluator cache=%i queue=%i>" % (len(self.cache), len(self.board_queue))

	@staticmethod
	def board_key(b):
		return (
			b.to_move,
			tuple(b.board),
		)

	def __contains__(self, board):
		return NNEvaluator.board_key(board) in self.cache

	def evaluate(self, input_board):
		# Build up an ensemble to evaluate together.
		ensemble = [input_board]
		while self.board_queue and len(ensemble) < NNEvaluator.ENSEMBLE_SIZE:
			queued_board = self.board_queue.popleft()
			# The board might have been evaluated since we queued it, in which case skip it.
			# TODO: Evaluate if this is worth it. How many transpositions do we get?
			if queued_board not in self:
				ensemble.append(queued_board)

		# Evaluate the boards together.
		self.ensemble_sizes.append(len(ensemble))
		features = list(map(board_to_features, ensemble))
		posteriors, values = sess.run(
			[network.policy_output, network.value_output],
			feed_dict={
				network.input_ph: features,
				network.is_training_ph: False,
			},
		)

		# Write an entry into our cache.
		for board, raw_posterior, (value,) in zip(ensemble, posteriors, values):
			raw_posterior = add_noise_to_logits(raw_posterior, self.temperature)
			softmax_posterior = softmax(raw_posterior)
			posterior = {move: get_move_score(softmax_posterior, move) for move in board.legal_moves()}
			# Renormalize the posterior. Add a small epsilon into the denominator to prevent divison by zero.
			denominator = sum(posterior.values()) + 1e-6
			posterior = {move: prob / denominator for move, prob in posterior.items()}
			entry = NNEvaluator.Entry(board=board, value=value, posterior=posterior, game_over=False)
			self.cache[NNEvaluator.board_key(board)] = entry

	def add_to_queue(self, board):
		if board not in self:
			self.board_queue.append(board)

	def populate(self, board):
		# XXX: This is ugly...
		if hasattr(board, "evaluations"):
			return

		# Get base value and posterior, independent of special value adjustments.
		if board not in self:
			self.evaluate(board)
		entry = self.cache[NNEvaluator.board_key(board)]
		# XXX: I think I do this separated out result adjustment thing in case there are properties like 3 fold repetition,
		# but in that case I shouldn't be mutating the cached entry! Uh oh. I really have to look into this later.

		# Evaluate special value adjustments.
		result = board.result()
		if result != None:
			assert result in (1, 2) and board.to_move in (1, 2)
			entry.value = 1.0 if result == board.to_move else -1.0
			entry.game_over = True
		board.evaluations = entry

		# If we exceed our cache size then empty our cache.
		if len(self.cache) > NNEvaluator.MAXIMUM_CACHE_ENTRIES:
			logging.debug("Emptying cache!")
			self.cache = {}

class MCTSEdge:
	def __init__(self, move, child_node, parent_node=None):
		self.move = move
		self.child_node = child_node
		self.parent_node = parent_node
		self.edge_visits = 0
		self.edge_total_score = 0

	def get_edge_score(self):
		return self.edge_total_score / self.edge_visits

	def adjust_score(self, new_score):
		self.edge_visits += 1
		self.edge_total_score += new_score

	def __str__(self):
		return "<%s %4.1f%% v=%i s=%.5f c=%i>" % (
			uai_interface.uai_encode_move(self.move),
			100.0 * self.parent_node.board.evaluations.posterior[self.move],
			self.edge_visits,
			self.get_edge_score(),
			len(self.child_node.outgoing_edges),
		)

class MCTSNode:
	def __init__(self, board, parent=None):
		self.board = board
		self.parent = parent
		self.all_edge_visits = 0
		self.outgoing_edges = {}
		self.graph_name_suffix = ""

	def total_action_score(self, move):
		if move in self.outgoing_edges:
			edge = self.outgoing_edges[move]
			u_score = MCTS.exploration_parameter * self.board.evaluations.posterior[move] * (1.0 + self.all_edge_visits)**0.5 / (1.0 + edge.edge_visits)
			Q_score = edge.get_edge_score() if edge.edge_visits > 0 else 0.0
		else:
			u_score = MCTS.exploration_parameter * self.board.evaluations.posterior[move] * (1.0 + self.all_edge_visits)**0.5
			Q_score = 0.0
		return Q_score + u_score

	def select_action(self, use_dirichlet_noise):
		global_evaluator.populate(self.board)
		# If we have no legal moves then return None.
		if not self.board.evaluations.posterior:
			return
		# If the game is over then return None.
		if self.board.evaluations.game_over:
			return
		# WARNING: Does this actually use Dirichlet noise? I don't think it does.
		posterior = self.board.evaluations.posterior
		if use_dirichlet_noise:
			self.board.evaluations.populate_noisy_posterior()
			posterior = self.board.evaluations.noisy_posterior
		return max(posterior, key=self.total_action_score)

	def graph_name(self, name_cache):
		if self not in name_cache:
			name_cache[self] = "n%i%s" % (len(name_cache), "") #self.graph_name_suffix)
		return name_cache[self]

	def make_graph(self, name_cache):
		l = []
		for edge in self.outgoing_edges.values():
			l.append("%s -> %s;" % (self.graph_name(name_cache), edge.child_node.graph_name(name_cache)))
		for edge in self.outgoing_edges.values():
			# Quadratic time here from worst case for deep graphs.
			l.extend(edge.child_node.make_graph(name_cache))
		return l

class TopN:
	def __init__(self, N, key):
		self.N = N
		self.key = key
		self.entries = []

	def add(self, item):
		if item not in self.entries:
			self.entries += [item]
		self.entries = sorted(self.entries, key=self.key)[-self.N:]

	def update(self, items):
		for i in items:
			self.add(i)

class MCTS:
	exploration_parameter = 1.0

	def __init__(self, root_board, use_dirichlet_noise=False):
		self.root_node = MCTSNode(root_board.copy())
		self.use_dirichlet_noise = use_dirichlet_noise

	def select_principal_variation(self, best=False):
		node = self.root_node
		edges_on_path = []
		while True:
			if best:
				if not node.outgoing_edges:
					break
				move = max(iter(node.outgoing_edges.values()), key=lambda edge: edge.edge_visits).move
			else:
				move = node.select_action(
					# Use dirichlet noise only if we are configured to, AND only at the root of the search tree.
					use_dirichlet_noise = self.use_dirichlet_noise and node == self.root_node,
				)
			if move not in node.outgoing_edges:
				break
			edge = node.outgoing_edges[move]
			edges_on_path.append(edge)
			node = edge.child_node
		return node, move, edges_on_path

	def step(self):
		def to_move_name(move):
			return "_%s" % (move,)
		# 1) Pick a child by repeatedly taking the best child.
		node, move, edges_on_path = self.select_principal_variation()
		# 2) If the move is non-null, expand once.
		if move != None:
			new_board = node.board.copy()
			try:
				new_board.move(move)
			except AssertionError as e:
				import sys
				print(node.board, file=sys.stderr)
				print(node.board.legal_moves(), file=sys.stderr)
				print(move in node.board.legal_moves(), file=sys.stderr)
				print(new_board, file=sys.stderr)
				print(new_board.legal_moves(), file=sys.stderr)
				print(move in new_board.legal_moves(), file=sys.stderr)
				print(self.root_node.outgoing_edges, file=sys.stderr)
				print(self.root_node.all_edge_visits, file=sys.stderr)
				print(self.root_node.select_action(), file=sys.stderr)
				print(self.root_node.board.evaluations.posterior, file=sys.stderr)
				print(self.root_node.board.evaluations.value, file=sys.stderr)
				del self.root_node.board.evaluations
				global_evaluator.populate(self.root_node.board)
				print(self.root_node.board.evaluations.posterior, file=sys.stderr)
				print(self.root_node.board.evaluations.value, file=sys.stderr)
				print("Move:", move, file=sys.stderr)
				raise e
			new_node = MCTSNode(new_board, parent=node)
			new_node.graph_name_suffix = to_move_name(move)
			new_edge = node.outgoing_edges[move] = MCTSEdge(move, new_node, parent_node=node)
			edges_on_path.append(new_edge)
		else:
			# 2b) If the move is null, then we had no legal moves, and just propagate the score again.
			new_node = node
		# 3a) Evaluate the new node.
		global_evaluator.populate(new_node.board)
		# 3b) Queue up some children just for efficiency.
		for m, probability in new_node.board.evaluations.posterior.items():
			if probability > NNEvaluator.PROBABILITY_THRESHOLD:
				new_board = new_node.board.copy()
				new_board.move(m)
				global_evaluator.add_to_queue(new_board)
		# Convert the expected value result into a score.
		value_score = (new_node.board.evaluations.value + 1) / 2.0
		# 4) Backup.
		inverted = False
		for edge in reversed(edges_on_path):
			# Remember that each edge corresponds to an alternating player, so we have to reverse scores.
			inverted = not inverted
			value_score = 1 - value_score
			assert inverted == (edge.parent_node.board.to_move != new_node.board.to_move)
			edge.adjust_score(value_score)
			edge.parent_node.all_edge_visits += 1
		if not edges_on_path:
			self.write_graph()
			logging.debug("WARNING no edges on path!")
		# The final value of edge encodes the very first edge out of the root.
		return edge

	def play(self, player, move, print_variation_count=True):
		assert self.root_node.board.to_move == player, "Bad play direction for MCTS!"
		if move not in self.root_node.outgoing_edges:
			if print_variation_count:
				logging.debug("Completely unexpected variation!")
			new_board = self.root_node.board.copy()
			new_board.move(move)
			self.root_node = MCTSNode(new_board)
			return
		edge = self.root_node.outgoing_edges[move]
		if print_variation_count:
			logging.debug("Traversing to variation with %i visits." % edge.edge_visits)
		self.root_node = edge.child_node
		self.root_node.parent = None

	def write_graph(self):
		name_cache = {}
		with open("/tmp/mcts.dot", "w") as f:
			f.write("digraph G {\n")
			f.write("\n".join(self.root_node.make_graph(name_cache)))
			f.write("\n}\n")
		return name_cache

class MCTSEngine:
	VISITS    = 10000000
	MAX_STEPS = 10000000
	TIME_SAFETY_MARGIN = 0.1
	IMPORTANCE_FACTOR = {
		1: 0.1, 2: 0.2,
		3: 0.3, 4: 0.4,
		5: 0.5, 6: 0.6,
		7: 0.7, 8: 0.8,
		9: 0.9,
	}
	# XXX: This is awful. Switch this over to run-time benchmarking.
	MAX_STEPS_PER_SECOND = 1500.0

	def __init__(self):
		self.state = ataxx_rules.AtaxxState.initial()
		self.mcts = MCTS(self.state)

	def set_state(self, new_board):
		# XXX: You can pick out the wrong board because board equality doesn't include history.
		# For example, you can't distinguish different kinds of promotions that are followed by a capture.
		# TODO: Evaluate if this can cause a really rare bug.

		# Check to see if this board is one of our children's children.
		for edge1 in self.mcts.root_node.outgoing_edges.values():
			for edge2 in edge1.child_node.outgoing_edges.values():
				if edge2.child_node.board == new_board:
					# We found a match! Reuse part of the tree.
					self.mcts.play(self.state.to_move, edge1.move)
					self.mcts.play(not self.state.to_move, edge2.move)
					self.state = new_board
					return
		logging.debug(RED + "Failed to match a subtree." + ENDC)
		self.state = new_board
		# XXX: Some UCI masters will ask us to make a move even when we can claim a threefold repetition.
		# If we can claim a threefold repetiton, clear out the ML_solid_result, so we make progress.

		self.mcts = MCTS(self.state)

	def genmove(self, time_to_think, early_out=True, use_weighted_exponent=None):
		start_time = time.time()
		most_visited_edges = TopN(2, key=lambda edge: edge.edge_visits)
		most_visited_edges.update(iter(self.mcts.root_node.outgoing_edges.values()))
		total_steps = 0
		for step_number in range(self.MAX_STEPS):
			now = time.time()
			# Compute remaining time we have left to think.
			remaining_time = time_to_think - (now - start_time)
			if remaining_time <= 0.0 and total_steps > 0:
				break
			# If we don't have enough time for the number two option to catch up, early out.
			if early_out and len(most_visited_edges.entries) == 2 and total_steps > 0:
				runner_up, top_choice = most_visited_edges.entries
				#logging.debug("Step number: %i runner_up: %r top_choice: %r %r" % (step_number, runner_up, top_choice, remaining_time))
				if runner_up.edge_visits + remaining_time * MCTSEngine.MAX_STEPS_PER_SECOND < top_choice.edge_visits:
					logging.debug("Early out; cannot catch up in %f seconds." % (remaining_time,))
					break
			total_steps += 1
			visited_edge = self.mcts.step()
			assert visited_edge.parent_node == self.mcts.root_node
			most_visited_edges.add(visited_edge)

			# We early out if we reach our POST value, and just visited the most visited edge.
			if visited_edge.edge_visits >= self.VISITS:
				break
			# We early out if we don't have enough time
			# Print debugging values.
			if step_number == 0 or (step_number + 1) % 250 == 0:
				logging.debug("Steps: %5i (C=%i/%i Top: %s This: %s)" % (
					step_number + 1,
					len(self.mcts.root_node.outgoing_edges),
					len(self.mcts.root_node.board.evaluations.posterior),
					(most_visited_edges.entries[-1] if most_visited_edges.entries else None),
					visited_edge,
				))
				if (step_number + 1) % 1000 == 0:
					self.print_principal_variation()
		logging.debug("Completed %i steps." % total_steps)
		logging.debug("Exploration histogram:\n%s" % (
			"\n".join(
				"    " + str(edge)
				for edge in sorted(
					iter(self.mcts.root_node.outgoing_edges.values()),
					key=lambda edge: -edge.edge_visits,
				)
			),
		))
		self.print_principal_variation()
		logging.debug("Cache entries: %i" % (len(global_evaluator.cache),))
		if not use_weighted_exponent:
			return most_visited_edges.entries[-1].move
		else:
			return self.sample_with_exponential_weight(use_weighted_exponent)

	def sample_with_exponential_weight(self, exponent):
		move_weights = {
			move: (edge.edge_visits / float(self.mcts.root_node.all_edge_visits)) ** exponent
			for move, edge in self.mcts.root_node.outgoing_edges.items()
		}
		# Normalize the weights.
		normalization = 1.0 / sum(move_weights.values())
		move_weights = {
			move: weight * normalization
			for move, weight in move_weights.items()
		}
		return sample_by_weight(move_weights)

	def genmove_with_time_control(self, our_time, our_increment):
		# First, figure out how many plies we probably have remaining in the game.
		moves_remaining = 20.0
		# Assume we will have to make this many additional moves.
		total_time = our_time + our_increment * moves_remaining
		time_budget = total_time / moves_remaining
		# Compute an importance factor.
		importance_factor = self.IMPORTANCE_FACTOR.get(self.state.fullmove_number, 1.3)
		importance_factor *= 1.5
		time_budget *= importance_factor
		# Never budget more than 50% of our remaining time, minus a little margin.
		time_budget = max(0.0, min(time_budget, 0.5 * our_time - self.TIME_SAFETY_MARGIN))
		logging.debug("Budgeting %.2fms for this move." % (time_budget * 1e3,))
		return self.genmove(time_budget)

	def print_principal_variation(self):
		_, _, pv = self.mcts.select_principal_variation(best=True)
		logging.debug("PV [%2i]: %s" % (
			len(pv),
			" ".join(
				["%s", RED + "%s" + ENDC][i % 2] % (uai_interface.uai_encode_move(edge.move),)
				for i, edge in enumerate(pv)
			),
		))

if __name__ == "__main__":
	#__import__("pprint").pprint(sorted(score_moves(chess.Board())))
	logging.basicConfig(
		format="[%(process)5d] %(message)s",
		level=logging.DEBUG,
	)
	initialize_model("models/96x12-sample.npy")
	setup_evaluator()
	engine = MCTSEngine()
	for _ in range(2):
		print("Doing warmup evaluation.")
		start = time.time()
		engine.genmove(0.1)
		stop = time.time()
		print("Warmup took:", stop - start)

	print("Starting performance section.")
	measure_time = 10.0
	engine.genmove(measure_time, early_out=False)

	total_visits = engine.mcts.root_node.all_edge_visits
	print("Total visits:", total_visits)
	print("Ensembles:", len(global_evaluator.ensemble_sizes))
	print("Average ensemble:", np.average(global_evaluator.ensemble_sizes))

	with open("speeds", "a+") as f:
		print("ES=%i  QD=%4i  PT=%.3f  (MT=%f)  append (renorm)  Speed: %f" % (
			NNEvaluator.ENSEMBLE_SIZE,
			NNEvaluator.QUEUE_DEPTH,
			NNEvaluator.PROBABILITY_THRESHOLD,
			measure_time,
			total_visits / measure_time,
		), file=f)

	print(global_evaluator.ensemble_sizes)

