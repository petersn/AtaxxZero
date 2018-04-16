#!/usr/bin/python

import os, glob, signal, subprocess, socket, atexit, time
import train

VISIT_COUNT = 200
FIRST_MODEL_VISIT_COUNT = 50
GAME_COUNT = 500
TRAINING_STEPS = 200
BONUS_TRAINING_STEPS_PER_ROUND = 100
TRAIN_ROUNDS_INCLUDED = 5
TRAIN_ROUNDS_MINIMUM = 3

# XXX: This is messy crap I don't like...
RESET_RUN_LENGTH = 100

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
		os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
	except Exception, e:
		print "ERROR in killpg:", e
	proc.kill()

def generate_games(model_name):
	game_directory = os.path.join("games", model_name)
	if not os.path.exists(game_directory):
		os.mkdir(game_directory)

	server_proc = subprocess.Popen(
		["python", "gpu_server.py", model_name, "6000"],
		stdout=open("/dev/null"),
		preexec_fn=os.setsid,
		close_fds=True,
	)
	def _(server_proc):
		atexit.register(lambda: kill(server_proc))
	_(server_proc)
	# Try to connect to our local server.
	for attempt in xrange(100):
		try:
			s = socket.create_connection(("localhost", 6000))
			s.close()
			break
		except socket.error:
			print "Couldn't connect..."
		time.sleep(0.25)
	else:
		print "ERROR: Couldn't connect to GPU server!"

	visit_count = VISIT_COUNT
	if model_name == "model-001":
		visit_count = FIRST_MODEL_VISIT_COUNT

	launch_proc = subprocess.Popen(
		["./launch.sh", model_name, visit_count],
		preexec_fn=os.setsid,
		close_fds=True,
	)
	def _(launch_proc):
		atexit.register(lambda: kill(launch_proc))
	_(launch_proc)

	# We now periodically check up on how many games we have.
	game_counts = []
	while True:
		game_count = count_games(glob.glob(os.path.join(game_directory, "*.json")))
		game_counts.append(game_count)
		print "Game count:", game_count
		time.sleep(10)
		if game_count >= GAME_COUNT:
			break
		if len(game_counts) > RESET_RUN_LENGTH:
			if game_counts[-RESET_RUN_LENGTH:] == game_counts[-1:] * RESET_RUN_LENGTH:
				print "BAD"
				kill(server_proc)
				kill(launch_proc)
				time.sleep(10)
				print "Yielding false..."
				return False

	# Create the signal file.
	print "Creating kill file."
	open(os.path.join(game_directory, "die"), "w").close()
	time.sleep(5)
	print "Sending kill signals."
	kill(server_proc)
	kill(launch_proc)
	print "Exiting."
	return True

#def train_model(game_dirs, old_name, new_name):

def index_to_model_name(i):
	return "model-%03i" % i

if __name__ == "__main__":
	current_model_number = 1
	while True:
		start = time.time()
		old_model = index_to_model_name(current_model_number)
		new_model = index_to_model_name(current_model_number + 1)

		# XXX: DRY with train.model_path?
		if os.path.exists(train.model_path(new_model)):
			print "Model already exists, skipping:", new_model
			current_model_number += 1
			continue

		print "=========================== Doing data generation for:", old_model
		print "Start time:", start
		while True:
			result = generate_games(old_model)
			if result:
				break
			print "BAD --- Restarting."
		print "=========================== Doing training:", old_model, "->", new_model
		# Figure out the directories of games to train on.
		low_index = min(current_model_number, max(TRAIN_ROUNDS_MINIMUM, current_model_number - TRAIN_ROUNDS_INCLUDED + 1))
		high_index = current_model_number
		game_dirs = [
			os.path.join("games", index_to_model_name(i))
			for i in xrange(low_index, high_index + 1)
		]
		print "Game directories:", game_dirs
		steps = TRAINING_STEPS + BONUS_TRAINING_STEPS_PER_ROUND * (len(game_dirs) - 1)
		print "Steps:", steps
		subprocess.check_call([
			"python", "train.py",
				"--steps", str(steps),
				"--games"] + game_dirs + [
				"--old-name", old_model,
				"--new-name", new_model,
		], close_fds=True)

		end = time.time()
		print "Total seconds:", end - start
		current_model_number += 1

		time.sleep(1)

