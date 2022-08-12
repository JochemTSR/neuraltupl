#!/usr/bin/env python

#
# tuplcomplex.py: More complex tUPL framework.
#
#     The more complex tUPL framework is suitable for modeling whilelem
#     loops with multiple conditions and serial codes. In this case, when
#     multiple conditions are True, one of the serial codes is to be
#     chosen arbitrarily. The loop terminates when no tuple exists for
#     which one of the conditions is true.
#
# Copyright (C) Harry Wijshoff, Kristian Rietveld
#

# For our Python3 friends.
from __future__ import print_function, division, absolute_import, unicode_literals

import collections

import os
import random


from learningcurveplot import LearningCurvePlot

class NeuralNetwork:
	def __init__(self, inputs, layers):
		self.graph, self.inputs, self.outputs, self.num_neurons = self.init_graph(inputs, layers)
		self.weights = self.init_weights()
		
	def init_graph(self, inputs, layers):
		graph = []
		neuron_index = 0
		
		# Inputs
		current_layer = []
		for _ in range(inputs):
			current_layer.append(neuron_index)
			neuron_index += 1
		inputs = current_layer
		
		# Hidden layers
		for layer_size in layers:
			prev_layer = current_layer
			current_layer = []
			
			# Add desired number of neurons to layer
			for _ in range(layer_size):
				current_layer.append(neuron_index)
				neuron_index += 1
				
			# Connect current layer to previous layer
			for input_neuron in prev_layer:
				for output_neuron in current_layer:
					graph.append(Tuple(input_neuron, output_neuron))	
		
		# Outputs (in our case just the last layer)
		outputs = current_layer
		
		num_neurons = neuron_index
		return graph, inputs, outputs, num_neurons

	def init_weights(self):
		weights = {}
		for neuron in self.graph:
			weights[neuron] = 1

		return weights

# We define our Tuple type here with fields "u" and "v".
Tuple = collections.namedtuple('Tuple', 'u v')

def init(network, inputs, labels):
	'''Initialize some tuple reservoirs (e.g. T) and necessary shared
	spaces (e.g. S). Shared spaces are usually modeled with a dictionary
	that is indexed with tuples.'''
	global G

	global WEIGHT
	global IS_OUTPUT
	global LABEL
	global SUM
	global ACT
	global OLD_EDGEVAL
	global OLD_ERROR
	global DELTA_SUM
	global OLD_DELTA
	global NEW_WEIGHT
	global OLD_DIFF_WEIGHT

	G = network.graph
	WEIGHT = network.weights
	IS_OUTPUT = {}
	LABEL = {}
	SUM = {}
	ACT = {}
	OLD_EDGEVAL = {}
	OLD_ERROR = {}
	DELTA_SUM = {}
	OLD_DELTA = {} # To determine whether a delta has changed
	NEW_WEIGHT = {} # Store updated weights until ready
	OLD_DIFF_WEIGHT = {}

	# Initialize neuron values
	for neuron in range(network.num_neurons):
		SUM[neuron] = 0
		ACT[neuron] = 0
		DELTA_SUM[neuron] = 0
		IS_OUTPUT[neuron] = False
		LABEL[neuron] = 0

	# Initialize weights
	for edge in G:
		NEW_WEIGHT[edge] = 0
		OLD_DIFF_WEIGHT[edge] = 0

	# Initialize edge values
	for edge in G:
		OLD_EDGEVAL[edge] = 0
		OLD_DELTA[edge] = 0

	# Set one-hot output encoding table
	for neuron in network.outputs:
		OLD_ERROR[neuron] = 0
		IS_OUTPUT[neuron] = True

	# Set input values (can be set in ACT of first layer neurons)
	for neuron in network.inputs:
		ACT[neuron] = inputs[neuron]

	# Set labels
	assert(len(labels) == len(network.outputs))
	pairs = zip(network.outputs, labels)
	for output, label in pairs:
		LABEL[output] = label

	print(LABEL)


###
# Definition of used activation function and its derivative
## 

def a_prime(value):
	'''Returns the derivative of the ReLU activation function'''
	# Since we are using ReLU as activation function, this computes its derivative w.r.t 'value'
	# Relu derivative: 1 if >=0, 0 otherwise
	if value < 0:
		return 0
	return 1
	
def activation(value):
	'''Returns the result of the ReLU activation function
	under the input value'''
	# We are using the ReLU activation function
	return max(0, value)


###
# Serial code 1: Forward Propagation
##

def cond1(e):
	'''Implement the condition for serial code 1, return a boolean'''
	edgeval = WEIGHT[e] * ACT[e.u]
	return edgeval != OLD_EDGEVAL[e]

def SC1body(e):
	'''The actual serial code to execute if cond1 is True'''
	edgeval = WEIGHT[e] * ACT[e.u]
	SUM[e.v] = SUM[e.v] - OLD_EDGEVAL[e] + edgeval
	OLD_EDGEVAL[e] = edgeval
	ACT[e.v] = activation(SUM[e.v])


###
# Serial code 2: Cost Computation
#

def cond2(e):
	'''Implement the condition for serial code 2, return a boolean'''
	error = (ACT[e.v] - LABEL[e.v]) ** 2
	return IS_OUTPUT[e.v] and error != OLD_ERROR[e.v]

def SC2body(e):
	'''The actual serial code to execute if cond2 is True'''
	error = (ACT[e.v] - LABEL[e.v]) ** 2
	DELTA_SUM[e.v] = 2 * (ACT[e.v] - LABEL[e.v])
	OLD_ERROR[e.v] = error


###
# Serial code 3: delta_sum computation
#

def cond3(e):
	'''Implement the condition for serial code 2, return a boolean'''
	delta = WEIGHT[e] * a_prime(SUM[e.v]) * DELTA_SUM[e.v]
	return delta != OLD_DELTA[e]

def SC3body(e):
	'''The actual serial code to execute if cond2 is True'''
	delta = WEIGHT[e] * a_prime(SUM[e.v]) * DELTA_SUM[e.v]
	DELTA_SUM[e.u] = DELTA_SUM[e.u] - OLD_DELTA[e] + delta
	OLD_DELTA[e] = delta


###
# Serial code 4: weight differentiation & update
#

def cond4(e):
	'''Implement the condition for serial code 2, return a boolean'''
	diff_weight = ACT[e.u] * a_prime(SUM[e.v]) * DELTA_SUM[e.v]
	return diff_weight != OLD_DIFF_WEIGHT[e]

def SC4body(e):
	'''The actual serial code to execute if cond2 is True'''
	diff_weight = ACT[e.u] * a_prime(SUM[e.v]) * DELTA_SUM[e.v]
	NEW_WEIGHT[e] = WEIGHT[e] - learn_rate * diff_weight
	OLD_DIFF_WEIGHT[e] = diff_weight


###
# Loop mechanics
##

def body(e):
	''' Return True when modification has been made '''
	#FIXME this can probably be done much easier, but stackoverflow is down.
	assert e in G

	# Determine if conditions hold
	c1 = cond1(e)
	c2 = cond2(e)
	c3 = cond3(e)
	c4 = cond4(e)

	# Make a list of all valid conditions
	true_conditions = []
	if c1:
		true_conditions.append("c1")
	if c2:
		true_conditions.append("c2")
	if c3:
		true_conditions.append("c3")
	if c4:
		true_conditions.append("c4")

	# Randomly pick one of the valid conditions
	choice = ""
	if true_conditions:
		choice = random.choice(true_conditions)

	# Execute body associated with picked condition
	if choice == "c1":
		SC1body(e)
		return
	elif choice == "c2":
		SC2body(e)
		return
	elif choice == "c3":
		SC3body(e)
		return
	elif choice == "c4":
		SC4body(e)
		return
	assert "Shouldn't be reached"


def checkConditions():
	'''Check if there is any tuple left for which at least one condition
	is True.'''
	tmp = False
	for e in G:
		tmp |= cond1(e)
		tmp |= cond2(e)
		tmp |= cond3(e)
		tmp |= cond4(e)
	return tmp


def loop(network):
	'''Implementation of the actual whilelem loop.'''

	# The whilelem loop is run until there is no tuple left for which at
	# least one of the conditions is True.
	changed = True
	while changed:
		changed = False

		# Execute loop body for each tuple in random order.
		# For testing it is sometimes useful to temporarily disable the
		# call to "shuffle".
		tmp = list(G)
		random.shuffle(tmp)
		for e in tmp:
			body(e)

		# Check if there's a tuple left for which at least one of the
		# conditions is True.
		changed |= checkConditions()
	
	# Update weights once whilelem loop terminates
	WEIGHT = NEW_WEIGHT
	network.weights = WEIGHT
	return [ACT[output] for output in network.outputs]

def initRandom():
	seed = 0
	for i, c in enumerate(bytearray(os.urandom(4))):
		seed += c << i * 8

	random.seed(seed)

	print("Initialized random number generator with seed: 0x{:x}\n".format(seed))


if __name__ == '__main__':
	initRandom()

	error_hist = []
	global learn_rate
	learn_rate = 0.0001

	net = NeuralNetwork(2, [3, 3, 1])
	for _ in range(50):
		a = random.randint(1, 10)
		b = random.randint(1, 10)
		label = a + b
		init(net, [a, b], [label])
		result = loop(net)
		result = result[0]
		print("Result: {}".format(result))
		error = abs(result-label)
		error_hist.append(error)

	plot = LearningCurvePlot(title="tUPL learning curve")
	plot.add_curve(error_hist, label='error')
	plot.save('tupl_curve.png')
