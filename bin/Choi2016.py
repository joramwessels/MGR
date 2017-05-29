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
from mgr_utils import MGRException
from mgr_utils import trackExceptions

# The maximum dataset size
data_size = 256

# Parameters
learning_rate = 0.001
training_iters = 200000
batch_size = 128
display_step = 25

# Network Parameters
n_input = 131136 # data input (img shape: 96*1366)
n_classes = 10 # total classes (0-9 digits)
dropout = 0.75 # Dropout, probability to keep units (unused)

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.string, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

def conv2d(x, F, bias, strides=1):
    x = tf.nn.conv2d(x, F, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, bias)
    return tf.nn.relu(x)

def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')

def conv_net(x, weights, biases):
    x = tf.reshape(x, shape=[-1, 96, 1366, 1])

    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    conv1 = maxpool2d(conv1, k=2)

    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    conv2 = maxpool2d(conv2, k=2)
	
    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
    conv3 = maxpool2d(conv3, k=2)
	
    conv4 = conv2d(conv3, weights['wc4'], biases['bc4'])
    conv4 = maxpool2d(conv4, k=2)

    out = tf.add(tf.matmul(conv4, weights['out']), biases['out'])
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

@trackExceptions
def train(log, dataset, dir=None, id=None):
	"""Trains a k2c2 network as described in (Choi, 2016).
	
	Args:
		log:		A logger object to track the progress
		dataset:	A Dataset object handling the data
		dir:		The directory to save the model in, if required
		id:			The name of the model to be trained
	Returns:
		The path to the save file, if dir and id were given, None otherwise
	
	"""
	global err
	log.info("Training Choi2016 k2c2 network...")
	# Launch the graph
	with tf.Session() as sess:
		sess.run(init)
		step = 1
		# Keep training until reach max iterations
		while (step * dataset.batch_size < training_iters):
			try:
				batch_y, batch_x = dataset.next_batch()
				# Run optimization op (backprop)
				sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
											   keep_prob: dropout})
				if step % display_step == 0:
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
				log.error(str(MGRException(ex=e)))
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
	info.log("Testing " + filename + "...")
	with tf.Session() as sess:
		saver.restore(sess, filename)
		acc = sess.run(accuracy, feed_dict={x: dataset.get_test_x()[:data_size],
											y: dataset.get_test_y()[:data_size],
											keep_prob: 1.})
	log.info("The acc of " + filename + " is " + str(acc))
	return acc