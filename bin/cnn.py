#! /usr/bin/python
# filename:			cnn.py
# author:			Joram Wessels
# date:				09-06-2017
# python versoin:	3.5
# dependencies:		tensorflow
# public functions:	train, test
# description:		Reads the prepared data from a file and trains the network
#
# This network contains of 3 convolutional layers with respectively 20, 41 and
# 41 3x3 filters. Each convolution is followed by a 2x4 maxpool. It uses zero-
# -padding and ReLu activation. The net is followed by a single softmax layer.
# No dropout

import sys
import numpy as np
import tensorflow as tf
from mgr_utils import log
from mgr_utils import log_exception
from mgr_utils import MGRException
from mgr_utils import trackExceptions

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
	
	out = tf.add(tf.matmul(fc1, weights['wout']), biases['bout'], name="pred")
	return out

def construct_model(x, y, n_classes, lr):
	weights = {
		# 3x3 conv, 1 input, 20 outputs (i.e. 20 filters)
		'wc1': tf.Variable(tf.random_normal([3, 3, 1, 20]), name='wc1'),
		'wc2': tf.Variable(tf.random_normal([3, 3, 20, 41]), name='wc2'),
		'wc3': tf.Variable(tf.random_normal([3, 3, 41, 41]), name='wc3'),
		# 12x20 is the output frame dimension, 41 is the last layer's output
		'wd1': tf.Variable(tf.random_normal([12*20*41, 1024]), name='wd1'),
		'wout': tf.Variable(tf.random_normal([1024, n_classes]), name='wout')
	}
	biases = {
		'bc1': tf.Variable(tf.random_normal([20]), name='bc1'),
		'bc2': tf.Variable(tf.random_normal([41]), name='bc2'),
		'bc3': tf.Variable(tf.random_normal([41]), name='bc3'),
		'bd1': tf.Variable(tf.random_normal([1024]), name='bd1'),
		'bout': tf.Variable(tf.random_normal([n_classes]), name='bout')
	}
	pred = conv_net(x, weights, biases)
	cost = tf.reduce_mean(\
			tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=y))
	optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)
	#correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
	correct_conf = tf.where(tf.cast(y, tf.bool), pred, tf.zeros(tf.shape(y)))
	correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(correct_conf, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name="accuracy")
	saver = tf.train.Saver(keep_checkpoint_every_n_hours=1)
	return pred, cost, optimizer, accuracy, saver

@trackExceptions
def train(log, data, dir=None, id=None, md=None, lr=.001, ti=200000, ds=25):
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
	log.info(51*'=')
	log.info('Training network: "' + dir + str(id) + '" ...')
	log.info("Preparing parameters and model...")
	
	# Preparing parameters
	if (md): data_size = md
	else: data_size = data.get_size()
	n_input = data.get_data_dim()[0] * data.get_data_dim()[1]
	n_classes = data.get_n_classes()
	x = tf.placeholder(tf.float32, [None, n_input], name='x')
	y = tf.placeholder(tf.float32, [None, n_classes], name='y')
	b_s = data.batch_size
	
	# Constructing model
	pred, cost, optimizer, accuracy, saver = construct_model(x, y, n_classes, lr)
	init = tf.global_variables_initializer()
	
	# Launch the graph
	log.info("Running session...")
	sess = tf.Session()
	sess.run(init)
	step = 1
	# Keep training until reach max iterations
	while (step * b_s < ti):
		try:
			ids, batch_y, batch_x = data.next_batch()
			# Run optimization op (backprop)
			sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
			if step % ds == 0:
				# Calculate batch loss and accuracy
				loss, acc = sess.run([cost, accuracy],
						feed_dict={x: batch_x, y: batch_y})
				log.info("Iter " + str(step*b_s) + ", Minibatch " +\
						 "Loss= {:.6f}".format(loss) + ", Training " +\
						 "Accuracy= {:.5f}".format(acc))
			step += 1
		except StopIteration as si:
			log.info("All " + str(step) + " batches were used for training.")
			break
		except Exception as e:
			global err
			err += 1
			log_exception(e)
			log_exception(MGRException(msg="previous exception happened " +\
				"during training of these files: " + ' --- '.join(ids)))
	log.info("Optimization Finished.")
	
	# Calculate accuracy for all test images
	log.info("Testing accuracy on training set: ")
	acc = sess.run(accuracy, feed_dict={x: data.get_test_x()[:data_size],
										y: data.get_test_y()[:data_size]})
	log.info("Acc on TRAIN set: " + str(acc))
	# Save the variables to storage
	if dir and id:
		save_path = saver.save(sess, dir + str(id))
		log.info("Model saved in file: %s" % save_path)
		return save_path
	sess.close()
	return
'''
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
	sess = tf.Session()
	tf.reset_default_graph()
	saver = tf.train.import_meta_graph(filename + '.meta')
	saver.restore(sess, filename)#tf.train.latest_checkpoint('./'))
	graph = tf.get_default_graph()
	#print([t.name for t in graph.as_graph_def().node])
	"""weights = {
		'wc1': graph.get_tensor_by_name("wc1:0"),
		'wc2': graph.get_tensor_by_name("wc2:0"),
		'wc3': graph.get_tensor_by_name("wc3:0"),
		'wd1': graph.get_tensor_by_name("wd1:0"),
		'wout': graph.get_tensor_by_name("wout:0"),
	}
	biases = {
		'bc1': graph.get_tensor_by_name("bc1:0"),
		'bc2': graph.get_tensor_by_name("bc2:0"),
		'bc3': graph.get_tensor_by_name("bc3:0"),
		'bd1': graph.get_tensor_by_name("bd1:0"),
		'bout': graph.get_tensor_by_name("bout:0"),
	}"""
	#x = graph.get_tensor_by_name("x:0")
	#y = graph.get_tensor_by_name("y:0")
	#accuracy = graph.get_tensor_by_name("accuracy:0")
	acc = sess.run("accuracy:0", feed_dict={"x:0": data.get_test_x(),
										"y:0": data.get_test_y()})
	log.info("The accuracy of " + filename + " on the TEST set: " + str(acc))
	return acc'''