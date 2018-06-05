#!/usr/bin/python

import os, glob, signal, subprocess, socket, atexit, time

def count_games(paths):
	total_games = 0
	for path in paths:
		with open(path) as f:
			for line in f:
				if line.strip():
					total_games += 1
	return total_games

def kill(proc):
	print "Killing:", proc
	try:
		os.kill(proc.pid, signal.SIGTERM)
	except Exception, e:
		print "ERROR in kill:", e
	proc.kill()

def generate_games(model_number):
	# Touch the games file to initialize it empty if it doesn't exist.
	open(index_to_games_path(model_number), "a").close()

	# Before we even launch check if we have enough games.
	if count_games([index_to_games_path(model_number)]) >= args.game_count:
		print "Enough games to start with!"
		return

	# Launch the games generation.
	games_proc = subprocess.Popen([
		"python", "accelerated_generate_games.py",
			"--network", index_to_model_path(model_number),
			"--output-games", index_to_games_path(model_number),
			"--visits", str(args.visits),
	], close_fds=True)
	# If our process dies take the games generation down with us.
	def _(games_proc):
		atexit.register(lambda: kill(games_proc))
	_(games_proc)

	# We now periodically check up on how many games we have.
	while True:
		game_count = count_games([index_to_games_path(model_number)])
		print "Game count:", game_count
		time.sleep(10)
		if game_count >= args.game_count:
			break

	# Signal the process to die gracefully.
	os.kill(games_proc.pid, signal.SIGTERM)
	# Wait up to two seconds, then forcefully kill it.
	time.sleep(2)
	kill(games_proc)
	print "Exiting."

def index_to_model_path(i):
	return os.path.join(args.prefix, "models", "model-%03i.npy" % i)

def index_to_games_path(i):
	return os.path.join(args.prefix, "games", "model-%03i.json" % i)

if __name__ == "__main__":
	import argparse
	class Formatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter): pass
	parser = argparse.ArgumentParser(
		description="""
Performs the main loop of alternating game generation (via accelerated_generate_games.py)
with training (via train.py). You specify a prefix path which will have the structure:
  PREFIX/
    games/
      model-001.json
      model-002.json
      ...
    models/
      model-001.npy
      model-002.npy
      ...
You are expected to create the PREFIX/games/ and PREFIX/models/ directories, and populate
the initial PREFIX/models/model-001.npy file. Then looper.py will run in a main loop of:

  1) Generate self-play games with the highest numbered present
     model until reaching the minimum game count.
  2) Train model n+1 from the current highest numbered model,
     and a pool of games from recent iterations.

It is relatively safe to interrupt and restart, as looper.py will automatically resume on
the most recent model. (However, interrupting and restarting looper.py of course
technically statistically biases the games slightly towards being shorter.)
""",
		formatter_class=Formatter,
	)
	parser.add_argument("--prefix", metavar="PATH", default=".", help="Prefix directory. Make sure this directory contains games/ and models/ subdirectories.")
	parser.add_argument("--visits", metavar="N", type=int, default=200, help="At each move in the self-play games perform MCTS until the root node has N visits.")
	parser.add_argument("--game-count", metavar="N", type=int, default=500, help="Minimum number of games to generate per iteration.")
	parser.add_argument("--training-steps-const", metavar="N", type=int, default=200, help="Base number of training steps to perform per iteration.")
	parser.add_argument("--training-steps-linear", metavar="N", type=int, default=50, help="We also apply an additional N steps for each additional iteration included in the training window.")
	parser.add_argument("--training-window", metavar="N", type=int, default=10, help="When training include games from the past N iterations.")
	parser.add_argument("--training-window-exclude", metavar="N", type=int, default=3, help="To help things get started faster we exclude games from the very first N iterations from later training game windows.")
	args = parser.parse_args()
	print "Arguments:", args

	current_model_number = 1

	while True:
		start = time.time()
		old_model = index_to_model_path(current_model_number)
		new_model = index_to_model_path(current_model_number + 1)

		if os.path.exists(new_model):
			print "Model already exists, skipping:", new_model
			current_model_number += 1
			continue

		print "=========================== Doing data generation for:", old_model
		print "Start time:", start
		generate_games(current_model_number)

		print "=========================== Doing training:", old_model, "->", new_model
		# Figure out the directories of games to train on.
		low_index = min(current_model_number, max(args.training_window_exclude + 1, current_model_number - args.training_window + 1))
		high_index = current_model_number
		games_paths = [
			index_to_games_path(i)
			for i in xrange(low_index, high_index + 1)
		]
		print "Game paths:", games_paths
		steps = args.training_steps_const + args.training_steps_linear * (len(games_paths) - 1)
		print "Steps:", steps
		subprocess.check_call([
			"python", "train.py",
				"--steps", str(steps),
				"--games"] + games_paths + [
				"--old-path", old_model,
				"--new-path", new_model,
		], close_fds=True)

		end = time.time()
		print "Total seconds for iteration:", end - start
		current_model_number += 1

