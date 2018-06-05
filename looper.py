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
	visit_count = VISIT_COUNT
	if model_number == 1:
		visit_count = FIRST_MODEL_VISIT_COUNT

	launch_proc = subprocess.Popen([
		"python", "game_generator.py",
			"--network", index_to_model_path(model_number),
			"--output-games", index_to_games_path(model_number),
			"--visits", str(visit_count),
	], close_fds=True)
	def _(launch_proc):
		atexit.register(lambda: kill(launch_proc))
	_(launch_proc)

	# We now periodically check up on how many games we have.
	game_counts = []
	while True:
		game_count = count_games([index_to_games_path(model_number)])
		game_counts.append(game_count)
		print "Game count:", game_count
		time.sleep(10)
		if game_count >= GAME_COUNT:
			break

	# Signal the process to die gracefully.
	os.kill(proc.pid, signal.SIGTERM)
	# Wait up to two seconds, then forcefully kill it.
	time.sleep(2)
	kill(launch_proc)
	print "Exiting."
	return True

def index_to_model_path(i):
	return os.path.join(path_prefix, "models", "model-%03i.npy" % i)

def index_to_games_path(i):
	return os.path.join(path_prefix, "games", "model-%03i.json" % i)

if __name__ == "__main__":
	import argparse
	VISIT_COUNT = 200
	FIRST_MODEL_VISIT_COUNT = 50
	GAME_COUNT = 500
	TRAINING_STEPS = 200
	BONUS_TRAINING_STEPS_PER_ROUND = 50
	TRAIN_ROUNDS_INCLUDED = 10
	TRAIN_ROUNDS_MINIMUM = 3

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
		result = generate_games(current_model_number)

		print "=========================== Doing training:", old_model, "->", new_model
		# Figure out the directories of games to train on.
		low_index = min(current_model_number, max(TRAIN_ROUNDS_MINIMUM, current_model_number - TRAIN_ROUNDS_INCLUDED + 1))
		high_index = current_model_number
		games_paths = [
			index_to_games_path(i)
			for i in xrange(low_index, high_index + 1)
		]
		print "Game paths:", games_paths
		steps = TRAINING_STEPS + BONUS_TRAINING_STEPS_PER_ROUND * (len(games_paths) - 1)
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

