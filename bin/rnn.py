#! /usr/bin/python
# filename:			rnn.py
# author:			Joram Wessels
# date:				15-06-2017
# python versoin:	3.5
# dependencies:		tensorflow
# public functions:	train
# description:		Reads the prepared data from a file and trains the network
#
# This network is a single layer LSTM RNN aimed at recognizing music genres.

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from mgr_utils import log
from mgr_utils import log_exception
from mgr_utils import MGRException
from mgr_utils import trackExceptions

def RNN(x, weights, biases, n_hidden, n_steps):
	x = tf.unstack(x, n_steps, 1)
	lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
	outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
	return tf.matmul(outputs[-1], weights['out']) + biases['out']

def construct_model(x, y, n_classes, n_hidden, n_steps, lr):
	weights = {
		'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
	}
	biases = {
		'out': tf.Variable(tf.random_normal([n_classes]))
	}
	pred = RNN(x, weights, biases, n_hidden, n_steps)
	cost = tf.reduce_mean(\
			tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
	optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)
	correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
	accuracy = tf.reduce_mean(tf.cast(pred, tf.float32), name="accuracy")
	saver = tf.train.Saver(keep_checkpoint_every_n_hours=1)
	return pred, cost, optimizer, accuracy, saver

@trackExceptions
def train(log, data, dir=None, id=None, md=None, lr=.001, ti=2*10^5, ds=25):
	"""Trains a 1-layer LSTM RNN
	
	Args:
		log:	A logger object to track the progress
		data:	A Dataset object handling the data
		dir:	The directory to save the model in, if required
		id:		The name of the model to be trained
		md:		The maximum dataset size to test on, or inf if not given
		lr:		Learning rate. Defaults to 0.001
		ti:		Training iterations. Defaults to 200,000
		ds:		Display step. The interval at which to log the progress
	Returns:
		The path to the save file, if dir and id were given, None otherwise
	
	"""
	log.info(51*'=')
	log.info('Training network: "' + dir + str(id) + '" ...')
	log.info("Preparing parameters and network...")
	
	# Preparing parameters
	if (md): data_size = md
	else:    data_size = data.get_size()
	n_input   = data.get_data_dim()[0]
	n_steps   = data.get_data_dim()[1]
	n_classes = data.get_n_classes()
	n_hidden  = 1 # was 128
	
	x = tf.placeholder(tf.float32, [None, n_steps, n_input], name='x')
	y = tf.placeholder(tf.float32, [None, n_classes], name='y')
	
	# Constructing model
	pred, cost, optimizer, accuracy, saver = construct_model(x, y, n_classes, \
														n_hidden, n_steps, lr)
	init = tf.global_variables_initializer()
	
	log.info("Running session...")
	sess = tf.Session()
	sess.run(init)
	step = 1
	# Keep training until reach max iterations
	while (step * data.batch_size < ti):
		try:
			ids, batch_y, batch_x = data.next_batch()
			# Run optimization op (backprop)
			sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
			if step % ds == 0:
				# Calculate batch loss and accuracy
				loss, acc = sess.run([cost, accuracy],
						feed_dict={x: batch_x, y: batch_y})
				log.info("Iter " + str(step*batch_size) + ", Minibatch " \
					  + "Loss= {:.6f}".format(loss) + ", Training " \
					  + "Accuracy= {:.5f}".format(acc))
			step += 1
		except StopIteration as si:
			pass
		except Exception as e:
			global err
			err += 1
			log_exception(e)
			log_exception(MGRException(msg="previous exception happened " +\
				"during training of these files: " + ' --- '.join(ids)))
	log.info("Optimization Finished.")
	
	# Calculate accuracy for all test images
	log.info("Testing accuracy on training set: ")
	test_x = data.get_test_x().reshape((-1, n_steps, n_input))
	acc = sess.run(accuracy, feed_dict={x: test_x, y: data.get_test_y()})
	log.info("Acc on TRAIN set: " + str(acc))
		
	# Save the variables to storage
	if dir and id:
		save_path = saver.save(sess, dir + str(id))
		log.info("Model saved in file: %s" % save_path)
		return save_path
	sess.close()
	return