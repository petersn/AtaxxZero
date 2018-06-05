#!/usr/bin/python

import sys
import numpy as np
import tensorflow as tf

product = lambda l: reduce(lambda x, y: x * y, l, 1)

BOARD_SIZE = 7
MOVE_TYPES = 17

class Network:
	INPUT_FEATURE_COUNT = 4
	NONLINEARITY = [tf.nn.relu]
	FILTERS = 96
	CONV_SIZE = 3
	BLOCK_COUNT = 12
	VALUE_FILTERS = 1
	POLICY_OUTPUT_SHAPE = [None, BOARD_SIZE, BOARD_SIZE, MOVE_TYPES]
	VALUE_OUTPUT_SHAPE = [None, 1]

	total_parameters = 0

	def __init__(self, scope_name, build_training=False):
		self.scope_name = scope_name
		with tf.variable_scope(scope_name):
			# First we build a tower producing some number of features.
			self.build_tower()
			# Next we build the heads.
			self.build_policy_head()
			self.build_value_head()
		# Finally, we build the training components if requested.
		if build_training:
			self.build_training()

	def build_tower(self):
		# Construct input/output placeholders.
		self.input_ph = tf.placeholder(
			tf.float32,
			shape=[None, BOARD_SIZE, BOARD_SIZE, self.INPUT_FEATURE_COUNT],
			name="input_placeholder")
		self.desired_policy_ph = tf.placeholder(
			tf.float32,
			shape=self.POLICY_OUTPUT_SHAPE,
			name="desired_policy_placeholder")
		self.desired_value_ph = tf.placeholder(
			tf.float32,
			shape=self.VALUE_OUTPUT_SHAPE,
			name="desired_value_placeholder")
		self.learning_rate_ph = tf.placeholder(tf.float32, shape=[], name="learning_rate")
		self.is_training_ph = tf.placeholder(tf.bool, shape=[], name="is_training")
		# Begin constructing the data flow.
		self.parameters = []
		self.flow = self.input_ph
		# Stack an initial convolution.
		self.stack_convolution(self.CONV_SIZE, self.INPUT_FEATURE_COUNT, self.FILTERS)
		self.stack_nonlinearity()
		# Stack some number of residual blocks.
		for _ in xrange(self.BLOCK_COUNT):
			self.stack_block()
		# Stack a final 1x1 convolution transitioning to fully-connected features.
		#self.stack_convolution(1, self.FILTERS, self.OUTPUT_CONV_FILTERS, batch_normalization=False)

	def build_policy_head(self):
		weights = self.new_weight_variable([1, 1, self.FILTERS, MOVE_TYPES])
		self.policy_output = tf.nn.conv2d(self.flow, weights, strides=[1, 1, 1, 1], padding="SAME")
#		self.policy_output = tf.reshape(self.policy_output, [-1, BOARD_SIZE * BOARD_SIZE, MOVE_TYPES])
#		self.policy_output = tf.matrix_transpose(self.policy_output)

	def build_value_head(self):
		weights = self.new_weight_variable([1, 1, self.FILTERS, self.VALUE_FILTERS])
		value_layer = tf.nn.conv2d(self.flow, weights, strides=[1, 1, 1, 1], padding="SAME")
		value_layer = tf.reshape(value_layer, [-1, BOARD_SIZE * BOARD_SIZE * self.VALUE_FILTERS])
		fc_w = self.new_weight_variable([BOARD_SIZE * BOARD_SIZE, 1])
		fc_b = self.new_bias_variable([1])
		self.value_output = tf.matmul(value_layer, fc_w) + fc_b
		self.value_output = tf.nn.tanh(self.value_output)

	def build_training(self):
		# Make policy head loss.
#		assert self.desired_policy_ph.shape == self.policy_output.shape
		self.policy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
			labels=tf.reshape(self.desired_policy_ph, [-1, BOARD_SIZE * BOARD_SIZE * MOVE_TYPES]),
			logits=tf.reshape(self.policy_output, [-1, BOARD_SIZE * BOARD_SIZE * MOVE_TYPES]),
		))
		# Make value head loss.
		self.value_loss = tf.reduce_mean(tf.square(self.desired_value_ph - self.value_output))
		# Make regularization loss.
		regularizer = tf.contrib.layers.l2_regularizer(scale=0.0001)
		reg_variables = tf.trainable_variables(scope=self.scope_name)
		self.regularization_term = tf.contrib.layers.apply_regularization(regularizer, reg_variables)
		# Loss is the sum of these three.
		self.loss = 0.3 * self.policy_loss + self.value_loss + self.regularization_term

		# Associate batch normalization with training.
		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.scope_name)
		with tf.control_dependencies(update_ops):
			self.train_step = tf.train.MomentumOptimizer(
				learning_rate=self.learning_rate_ph, momentum=0.9).minimize(self.loss)

	def new_weight_variable(self, shape):
		self.total_parameters += product(shape)
		stddev = 0.2 * (2.0 / product(shape[:-1]))**0.5
		var = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
		self.parameters.append(var)
		return var

	def new_bias_variable(self, shape):
		self.total_parameters += product(shape)
		var = tf.Variable(tf.constant(0.1, shape=shape))
		self.parameters.append(var)
		return var

	def stack_convolution(self, kernel_size, old_size, new_size, batch_normalization=True):
		weights = self.new_weight_variable([kernel_size, kernel_size, old_size, new_size])
		self.flow = tf.nn.conv2d(self.flow, weights, strides=[1, 1, 1, 1], padding="SAME")
		if batch_normalization:
			self.flow = tf.layers.batch_normalization(
				self.flow,
				center=True,
				scale=True,
				training=self.is_training_ph)
		else:
			bias = self.new_bias_variable([new_size])
			self.flow = self.flow + bias # TODO: Is += equivalent?

	def stack_nonlinearity(self):
		self.flow = self.NONLINEARITY[0](self.flow)

	def stack_block(self):
		initial_value = self.flow
		# Stack the first convolution.
		self.stack_convolution(self.CONV_SIZE, self.FILTERS, self.FILTERS)
		self.stack_nonlinearity()
		# Stack the second convolution.
		self.stack_convolution(self.CONV_SIZE, self.FILTERS, self.FILTERS)
		# Add the skip connection.
		self.flow = self.flow + initial_value
		# Stack on the deferred non-linearity.
		self.stack_nonlinearity()

	def train(self, samples, learning_rate):
		self.run_on_samples(self.train_step.run, samples, learning_rate=learning_rate, is_training=True)

	def get_loss(self, samples):
		return self.run_on_samples(self.loss.eval, samples)

	def get_accuracy(self, samples):
		results = self.run_on_samples(self.final_output.eval, samples).reshape((-1, 64 * 64))
		#results = results.reshape((-1, 64 * 8 * 8))
		results = np.argmax(results, axis=-1)
		assert results.shape == (len(samples["features"]),)
		correct = 0
		for move, result in zip(samples["moves"], results):
			lhs = np.argmax(move.reshape((64 * 64,)))
			#assert lhs.shape == result.shape == (2,)
			correct += lhs == result #np.all(lhs == result)
		return correct / float(len(samples["features"]))

	def run_on_samples(self, f, samples, learning_rate=0.01, is_training=False):
		return f(feed_dict={
			self.input_ph:          samples["features"],
			self.desired_policy_ph: samples["policies"],
			self.desired_value_ph:  samples["values"],
			self.learning_rate_ph:  learning_rate,
			self.is_training_ph:    is_training,
		})

# XXX: This is horrifically ugly.
# TODO: Once I have a second change it to not do this horrible graph scraping that breaks if you have other things going on.
def get_batch_norm_vars(net):
	return [
		i for i in tf.global_variables(scope=net.scope_name)
		if "batch_normalization" in i.name and ("moving_mean:0" in i.name or "moving_variance:0" in i.name)
	]

def save_model(net, path):
	x_conv_weights = [sess.run(var) for var in net.parameters]
	x_bn_params = [sess.run(i) for i in get_batch_norm_vars(net)]
	np.save(path, [x_conv_weights, x_bn_params])
	print "\x1b[35mSaved model to:\x1b[0m", path

# XXX: Still horrifically fragile wrt batch norm variables due to the above horrible graph scraping stuff.
def load_model(net, path):
	x_conv_weights, x_bn_params = np.load(path)
	assert len(net.parameters) == len(x_conv_weights), "Parameter count mismatch!"
	operations = []
	for var, value in zip(net.parameters, x_conv_weights):
		operations.append(var.assign(value))
	bn_vars = get_batch_norm_vars(net)
	assert len(bn_vars) == len(x_bn_params), "Bad batch normalization parameter count!"
	for var, value in zip(bn_vars, x_bn_params):
		operations.append(var.assign(value))
	sess.run(operations)

if __name__ == "__main__":
	net = Network("net/")
	print get_batch_norm_vars(net)
	print net.total_parameters

