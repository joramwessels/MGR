#! /usr/bin/python
# filename:			Choi2016.py
# author:			Joram Wessels, Aymeric Damien
# date:				25-05-2017
# python versoin:	2.7
# source:			https://github.com/aymericdamien/TensorFlow-Examples/
# dependencies:		numpy, tensorflow
# public functions:	train, test
# description:		A tensorflow network modeled after the fully convolutional
#					k2c2 network described in (Choi, 2016)

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
	
    conv4 = conv2d(conv3, weights['wc4'], biases['bc4'])
    conv4 = maxpool2d(conv4, k1=3, k2=5)
	
    conv5 = conv2d(conv4, weights['wc5'], biases['bc5'])
    conv5 = maxpool2d(conv5, k1=4, k2=4)
	
    out = tf.add(tf.matmul(conv5, weights['out']), biases['out'])
    return out

def prepare_variables(n_classes):
	weights = {
		# 3x3 conv, 1 input, 20 outputs (i.e. 20 filters)
		'wc1': tf.Variable(tf.random_normal([3, 3, 1, 20])),
		'wc2': tf.Variable(tf.random_normal([3, 3, 20, 41])),
		'wc3': tf.Variable(tf.random_normal([3, 3, 41, 41])),
		'wc4': tf.Variable(tf.random_normal([3, 3, 41, 62])),
		'wc5': tf.Variable(tf.random_normal([3, 3, 62, 83])),
		'out': tf.Variable(tf.random_normal([1, 1, 83, n_classes]))
	}
	biases = {
		'bc1': tf.Variable(tf.random_normal([20])),
		'bc2': tf.Variable(tf.random_normal([41])),
		'bc3': tf.Variable(tf.random_normal([41])),
		'bc4': tf.Variable(tf.random_normal([62])),
		'bc5': tf.Variable(tf.random_normal([83])),
		'out': tf.Variable(tf.random_normal([n_classes]))
	}
	return weights, biases

def construct_model(n_input, n_classes):
	x = tf.placeholder(tf.float32, [None, n_input], name='x')
	y = tf.placeholder(tf.float32, [None, n_classes], name='y')
	keep_prob = tf.placeholder(tf.float32, name='keep_prob')
	learn_r = tf.placeholder(tf.float32, name='learn_r')
	# defining the model
	weights, biases = prepare_variables(n_classes)
	pred = conv_net(x, weights, biases, keep_prob, name='pred')
	#z_pred = tf.reduce_sum(y, 0)
	# the cost function reduces the sigmoid cross entropy error
	cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred,
																  labels=y),
																  name='cost')
	# using an Adam optimizer to minimize the cost function
	optimizer = tf.train.AdamOptimizer(learning_rate=learn_r).minimize(cost)
	# selecting the probabilities that ought to be highest
	correct_prob = tf.where(tf.cast(y, tf.bool), pred, tf.zeros(tf.shape(y)))
	# testing if the max value of this selection is equal to the original
	corr_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(correct_prob, 1))
	# taking the mean of the resulting binary array
	accuracy = tf.reduce_mean(tf.cast(corr_pred, tf.float32), name="accuracy")
	vars = {v.name:v for v in (list(weights.values()) + list(biases.values()))}
	saver = tf.train.Saver(vars)#, keep_checkpoint_every_n_hours=1)
	return optimizer, saver, tf.get_default_graph()

@trackExceptions
def train(log, data, dir=None, id=None, a=.001, ti=200000, ds=25, do=1.0):
	"""Trains a 3-layer ConvNet ending in a single Softmax layer.
	
	Args:
		log:	A logger object to track the progress
		data:	A Dataset object handling the data
		dir:	The directory to save the model in, if required
		id:		The name of the model to be trained
		a:		Learning rate. Defaults to 0.001
		ti:		Training iterations. Defaults to 200,000
		ds:		Display step. The interval at which to log the progress
		do:		The dropout rate. Must be a float between 0 and 1. Default=1.0
	Returns:
		The path to the save file, if dir and id were given, None otherwise
	
	"""
	log.info('Training network: "' + dir + str(id) + '" ...')
	log.info("Preparing parameters and model...")
	
	# Preparing parameters
	tf.reset_default_graph()
	n_input = data.get_data_dim()[0] * data.get_data_dim()[1]
	n_classes = data.get_n_classes()
	b_s = data.batch_size
	
	# Constructing model
	opt, sav, graph = construct_model(n_input, n_classes)
	init = tf.global_variables_initializer()
	
	# Launch the graph
	log.info("Running session...")
	sess = tf.Session()
	sess.run(init)
	optimize(sess, data, graph, opt, b_s, a, ti, ds, do)
	log.info("Optimization Finished.")
	
	# Calculate accuracy for all test images
	log.info("Testing accuracy on training set: ")
	score = accuracy(sess, graph, data.get_train_x(), data.get_train_y())
	log.info("Acc on TRAIN set: " + str(score))
	
	# Save the variables to storage
	if dir and id: return save(sess, sav, dir + str(id))
	sess.close()
	return

def optimize(sess, data, graph, opt, b_s, a, ti, ds, do):
	x = graph.get_tensor_by_name('x:0')
	y = graph.get_tensor_by_name('y:0')
	kp = graph.get_tensor_by_name('keep_prob:0')
	lr = graph.get_tensor_by_name('learn_r:0')
	step = 1
	data_left = True
	# Keep training until reach max iterations
	while (step * b_s < ti and data_left):
		try:
			ids, batch_y, batch_x = data.next_batch()
			sess.run(opt, feed_dict={x: batch_x, y: batch_y, kp: do, lr: a})
			if step % ds == 0:
				score = accuracy(sess, graph, data.get_test_x(), data.get_test_y())
				log.info("Iter %i: TRAIN acc = %.5f" %((step*b_s), score))
			step += 1
		except StopIteration:
			log.info("All " + str(step) + " batches were used for training.")
			data_left = False
		except Exception as e:
			global err
			err += 1
			try:
				log_exception(e, msg="previous exception happened " +\
				"during training of these files: " + ' --- '.join(ids))
			except:
				log_exception(e)
	return

def accuracy(sess, graph, xfeed, yfeed):
	x = graph.get_tensor_by_name('x:0')
	y = graph.get_tensor_by_name('y:0')
	kp = graph.get_tensor_by_name('keep_prob:0')
	lr = graph.get_tensor_by_name('learn_r:0')
	acc = graph.get_tensor_by_name('accuracy:0')
	feed_dict = {x: xfeed, y: yfeed, kp: 1.0, lr: 1.0}
	score = sess.run(acc, feed_dict=feed_dict)
	return score

def save(sess, sav, filename):
	savefile = sav.save(sess, filename)
	log.info("Model saved in file: %s" % savefile)
	return savefile
