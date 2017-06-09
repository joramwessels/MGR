#! /usr/bin/python
# filename:			cnn.py
# author:			Joram Wessels
# date:				09-06-2017
# python versoin:	2.7
# dependencies:		tensorflow
# public functions:	train, test
# description:		Reads the prepared data from a file and trains the network
#
# This network contains of 3 convolutional layers with respectively 20, 41 and
# 41 3x3 filters. Each convolution is followed by a 2x4 maxpool. It uses zero-
# -padding and ReLu activation. The net is followed by a single softmax layer.
# No dropout

import numpy as np
import tensorflow as tf
from mgr_utils import log
from mgr_utils import log_exception
from mgr_utils import MGRException
from mgr_utils import trackExceptions

def train(log, dataset, dir=None, id=None, md=None, lr=.001, ti=2*10^5, ds=25):
	"""Trains a 3-layer ConvNet ending in a single Softmax layer.
	
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
	global err
	log.info("Training 3-layer CNN...")
	
	# Preparing parameters
	if (md): data_size = md
	else: data_size = dataset.get_size()
	n_input = dataset.get_data_dim()[0] * dataset.get_data_dim()[1]
	n_classes = dataset.get_n_classes()
	dropout = 0.75 # Dropout, probability to keep units (unused)
	x = tf.placeholder(tf.float32, [None, n_input])
	y = tf.placeholder(tf.int8, [None, n_classes])
	keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)
	
	# Constructing model
	pred, cost, optimizer, accuracy = construct_model(x, y, n_classes, lr)
	init = tf.global_variables_initializer()
	saver = tf.train.Saver(keep_checkpoint_every_n_hours=1)
	
	# Launch the graph
	with tf.Session() as sess:
		sess.run(init)
		step = 1
		# Keep training until reach max iterations
		while (step * dataset.batch_size < ti):
			try:
				ids, batch_y, batch_x = dataset.next_batch()
				# Run optimization op (backprop)
				sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
											   keep_prob: dropout})
				if step % ds == 0:
					# Calculate batch loss and accuracy
					loss, acc = sess.run([cost, accuracy],
							feed_dict={x: batch_x, y: batch_y, keep_prob: 1.})
					log.info("Iter " + str(step*batch_size) + ", Minibatch " \
						  + "Loss= {:.6f}".format(loss) + ", Training " \
						  + "Accuracy= {:.5f}".format(acc))
				step += 1
			except StopIteration as si:
				pass
			except Exception as e:
				err += 1
				log_exception(e)
				log_exception(MGRException(msg="previous exception happened " +\
					"during training of these files: " + ' --- '.join(ids)))
		log.info("Optimization Finished.")

		# Calculate accuracy for all test images
		log.info("Testing accuracy on training set: ")
		acc = sess.run(accuracy, feed_dict={x: dataset.get_test_x()[:data_size],
											y: dataset.get_test_y()[:data_size],
											keep_prob: 1.})
		log.info("Accuracy of " + str(id) + " on the TRAIN set: " + str(acc))
		# Save the variables to storage
		if dir and id:
			save_path = saver.save(sess, dir + str(id))
			log.info("Model saved in file: %s" % save_path)
			return save_path
	return

def test(log, data, filename):
	"""Loads and tests a model on the given cross validated dataset.
	
	Args:
		log:		A logger object to track the progress
		filename:	The path to the file with the model
		data:		The Dataset object with the test set test on
	Returns:
		The accuracy of the cross validated test
	
	"""
	log.info('Testing CNN "' + filename + '" ...')
	keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)
	pred = conv_net(x, weights, biases)
	correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
	saver = tf.train.Saver()
	with tf.Session() as sess:
		saver.restore(sess, filename)
		acc = sess.run(accuracy, feed_dict={x: data.get_test_x(),
											y: data.get_test_y(),
											keep_prob: 1.})
	log.info("The accuracy of " + filename + " on the TEST set: " + str(acc))
	return acc


def conv2d(x, F, bias, strides=1):
    x = tf.nn.conv2d(x, F, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, bias)
    return tf.nn.relu(x)

def maxpool2d(x, k1=2, k2=2):
    return tf.nn.max_pool(x, ksize=[1, k1, k2, 1], strides=[1, k1, k2, 1],
                          padding='SAME')

def conv_net(x, weights, biases):
	x = tf.reshape(x, shape=[-1, 96, 1280, 1])
	
	conv1 = conv2d(x, weights['wc1'], biases['bc1'])
	conv1 = maxpool2d(conv1, k1=2, k2=4)
	
	conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
	conv2 = maxpool2d(conv2, k1=2, k2=4)
	
	conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
	conv3 = maxpool2d(conv3, k1=2, k2=4)
	
	fc1 = tf.reshape(conv3, [-1, weights['wd1'].get_shape().as_list()[0]])
	fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
	fc1 = tf.nn.relu(fc1)
	
	out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
	return out

def construct_model(x, y, n_classes, lr):
	weights = {
		# 3x3 conv, 1 input, 20 outputs (i.e. 20 filters)
		'wc1': tf.Variable(tf.random_normal([3, 3, 1, 20])),
		'wc2': tf.Variable(tf.random_normal([3, 3, 20, 41])),
		'wc3': tf.Variable(tf.random_normal([3, 3, 41, 41])),
		'wd1': tf.Variable(tf.random_normal([12*20*41, 1024])),
		'out': tf.Variable(tf.random_normal([1024, n_classes]))
	}
	biases = {
		'bc1': tf.Variable(tf.random_normal([20])),
		'bc2': tf.Variable(tf.random_normal([41])),
		'bc3': tf.Variable(tf.random_normal([41])),
		'bd1': tf.Variable(tf.random_normal([1024])),
		'out': tf.Variable(tf.random_normal([n_classes]))
	}
	pred = conv_net(x, weights, biases)
	cost = tf.reduce_mean(\
			tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
	optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)
	correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
	return pred, cost, optimizer, accuracy