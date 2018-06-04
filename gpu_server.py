#!/usr/bin/python

## Workaround?
#import gevent.hub
#gevent.hub.Hub.backend = "poll"

import numpy
import Queue, threading, time, random, sys, hashlib
import gevent.event, gevent.queue, gevent.server
from mprpc import RPCServer
import engine
import train
import model

submit_queue = gevent.queue.Queue()

FEATURES_SHAPE = model.BOARD_SIZE, model.BOARD_SIZE, model.Network.INPUT_FEATURE_COUNT

class Processor:
	MAXIMUM_WAIT_TIME = 0.01
	MARSHALL_COUNT    = 8

	def __init__(self):
		self.empty()
		self.process()
		self.main_loop()

	def empty(self):
		self.accumulated_features = []
		self.accumulated_slots = []

	def process(self):
		if not self.accumulated_features:
			self.last_process_time = time.time()
			return
		assert len(self.accumulated_features) == len(self.accumulated_slots)

		batch_size = len(self.accumulated_features)
#		if random.random() < 0.001:
#			print "Evaluating batch of size:", batch_size

		# Do the actual processing here!
		posteriors, values = engine.sess.run(
			[engine.network.policy_output, engine.network.value_output],
			feed_dict={
				engine.network.input_ph: self.accumulated_features,
				engine.network.is_training_ph: False,
			},
		)
		assert posteriors.dtype == numpy.float32
		assert values.dtype == numpy.float32
		for slot, posterior, (value,) in zip(self.accumulated_slots, posteriors, values):
			posterior_string = posterior.tostring()
			assert len(posterior_string) == (7 * 7 * 17) * 4
			value = float(value)
			slot.set((posterior_string, value))

		self.empty()
		self.last_process_time = time.time()

	def main_loop(self):
		while True:
			now = time.time()
			# Compute the maximum amount of time we're allowed to wait.
			time_since_last_process = now - self.last_process_time
			allowed_sleep_time = max(0, self.MAXIMUM_WAIT_TIME - time_since_last_process)
			try:
				feature_string, result_slot = submit_queue.get(timeout=allowed_sleep_time)
				array = numpy.fromstring(feature_string, dtype=numpy.int8).reshape(FEATURES_SHAPE)
				self.accumulated_features.append(array)
				self.accumulated_slots.append(result_slot)
			except gevent.queue.Empty:
				self.process()
			if len(self.accumulated_features) >= self.MARSHALL_COUNT:
#				print "Processing due to full batch."
				self.process()

class NetworkServer(RPCServer):
	def network(self, feature_string):
		result_slot = gevent.event.AsyncResult()
		submit_queue.put((feature_string, result_slot))
		return result_slot.get()

if len(sys.argv) != 3:
	print "Usage: %s model_path port-to-host-on" % (sys.argv[0],)
	exit(1)

#model_path = train.model_path(sys.argv[1])
engine.initialize_model(sys.argv[1])

port = int(sys.argv[2])
print
print "Launching on port:", port

gevent.spawn(Processor)
server = gevent.server.StreamServer(("127.0.0.1", port), NetworkServer)
server.serve_forever()

