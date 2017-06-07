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

# The maximum dataset size
data_size = 256

# Parameters
learning_rate = 0.001
training_iters = 200000
display_step = 25

# Network Parameters
n_input = 122880 # data input (img shape: 96*1280)
n_classes = 9 # total genres
dropout = 0.75 # Dropout, probability to keep units (unused)

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.int8, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

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
# Construct model
pred = conv_net(x, weights, biases)
# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
# Initializing the variables
init = tf.global_variables_initializer()
# Add ops to save and restore all the variables
saver = tf.train.Saver()

def construct_model(x, y, n_classes, lr):
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
	pred = conv_net(x, weights, biases)
	cost = tf.reduce_mean(\
			tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
	optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)
	correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
	return pred, cost, optimizer, accuracy

@trackExceptions
def train(log, dataset, dir=None, id=None, md=None, lr=.001, ti=2*10^5, ds=25):
	"""Trains a k2c2 network as described in (Choi, 2016).
	
	Args:
		log:		A logger object to track the progress
		dataset:	A Dataset object handling the data
		dir:		The directory to save the model in, if required
		id:			The name of the model to be trained
		md:			The maximum dataset size to test on, or inf if not given
		lr:			Learning rate. Defaults to 0.001
		ti:			Training iterations. Defaults to 200,000
		ds:			Display step. The interval at which to log the progress
	Returns:
		The path to the save file, if dir and id were given, None otherwise
	
	"""
	global err
	log.info("Training Choi2016 k2c2 network...")
	
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
	saver = tf.train.Saver()
	
	# Launch the graph
	with tf.Session() as sess:
		sess.run(init)
		step = 1
		# Keep training until reach max iterations
		while (step * dataset.batch_size < ti):
			try:
				batch_y, batch_x = dataset.next_batch()
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
		log.info("Optimization Finished.")

		# Calculate accuracy for all test images
		log.info("Testing accuracy on training set:", \
			sess.run(accuracy, feed_dict={x: dataset.get_test_x()[:data_size],
										  y: dataset.get_test_y()[:data_size],
										  keep_prob: 1.}))
	# Save the variables to storage
	if dir and id:
		save_path = saver.save(sess, dir + str(id) + '.ckpt')
		log.info("Model saved in file: %s" % save_path)
		return save_path
	else: return

def test(log, dataset, filename):
	"""Loads and tests a model on the given cross validated dataset.
	
	Args:
		log:			A logger object to track the progress
		filename:		The path to the file with the model
		dataset:		The Dataset object with the test set test on
	Returns:
		The accuracy of the cross validated test
	
	"""
	log.info("Testing " + filename + "...")
	with tf.Session() as sess:
		saver.restore(sess, filename)
		acc = sess.run(accuracy, feed_dict={x: dataset.get_test_x()[:data_size],
											y: dataset.get_test_y()[:data_size],
											keep_prob: 1.})
	log.info("The acc of " + filename + " is " + str(acc))
	return acc