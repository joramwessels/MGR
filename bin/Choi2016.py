#! /usr/bin/python
# filename:			Choi2016.py
# author:			Joram Wessels, Yann LeCunn
# date:				25-05-2017
# python versoin:	2.7
# dependencies:		numpy, tensorflow
# public functions:	
# description:		

import numpy as np
import tensorflow as tf

def conv2d(x, F, bias, strides=1):
    x = tf.nn.conv2d(x, F, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, bias)
    return tf.nn.relu(x)

def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')

def conv_net(x, weights, biases, dropout):
    x = tf.reshape(x, shape=[96, 1366])

    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    conv1 = maxpool2d(conv1, k=2)

    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    conv2 = maxpool2d(conv2, k=2)
	
    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
    conv3 = maxpool2d(conv3, k=2)
	
    conv4 = conv2d(conv3, weights['wc4'], biases['bc4'])
    conv4 = maxpool2d(conv4, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv4, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
	
    fc2 = tf.reshape(fc1, [-1, weights['wd2'].get_shape().as_list()[0]])
    fc2 = tf.add(tf.matmul(fc2, weights['wd2']), biases['bd2'])
    fc2 = tf.nn.relu(fc2)

    # Output, class prediction
    out = tf.add(tf.matmul(fc2, weights['out']), biases['out'])
    return out

weights = {
    # 3x3 conv, 1 input, 20 outputs (i.e. 20 filters)
    'wc1': tf.Variable(tf.random_normal([3, 3, 1, 20])),
    'wc2': tf.Variable(tf.random_normal([3, 3, 20, 20*41])),
    'wc3': tf.Variable(tf.random_normal([3, 3, 20*41, 20*41*41])),
    'wc4': tf.Variable(tf.random_normal([3, 3, 20*41*41, 20*41*41*62])),
    'wc5': tf.Variable(tf.random_normal([3, 3, 20*41*41*62, 20*41*41*62*83])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([20*41*41*62*83, n_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([20])),
    'bc2': tf.Variable(tf.random_normal([20*41])),
    'bc3': tf.Variable(tf.random_normal([20*41*41])),
    'bc4': tf.Variable(tf.random_normal([20*41*41*62])),
    'bc5': tf.Variable(tf.random_normal([20*41*41*62*83])),
    'bd1': tf.Variable(tf.random_normal([20*41*41*62*83])),
    'bd2': tf.Variable(tf.random_normal([20*41*41*62*83])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = conv_net(x, weights, biases, keep_prob)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()
# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
                                       keep_prob: dropout})
        if step % display_step == 0:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                              y: batch_y,
                                                              keep_prob: 1.})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")

    # Calculate accuracy for 256 mnist test images
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: mnist.test.images[:256],
                                      y: mnist.test.labels[:256],
                                      keep_prob: 1.}))