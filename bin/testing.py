#! usr/bin/python
# filename:			testing.py
# author:			Joram Wessels
# date:				12-06-2017
# python versoin:	3.5
# dependencies:		tensorflow
# public functions:	test, test_on_file
# description:		Reads the data from storage and tests the network

import sys, argparse
import numpy as np
import tensorflow as tf
from dataset import Dataset
from mgr_utils import log
from mgr_utils import trackExceptions
from mgr_utils import log_exception

def main(argv):
	p = parser.parse_args(argv[1:])
	if (p.msg):
		log.info(51*'=')
		log.info(p.msg)
	log.info(51*'=')
	test_on_file(p.model, p.dataset, abs=p.abs, seed=p.seed)
	log.info(51*'=' + '\n')

@trackExceptions
def test_on_file(model, dataset, abs='all', seed=None):
	"""Loads the preprocessed dataset and tests the model on it
	
	This function assumes that the dataset was not involved in training. It also
	requires the training procedure to have named the accuracy, x and y tensors
	with these respective names before saving them.
	
	Args:
		model:	The base name of the save files (without extension)
		dataset:	The preprocessed dataset as a txt file
		abs:	The evaluation abstraction: '1', '2', '3', '<1', or 'leafs'
		seed:		If required, a random gen seed to replicate CV results
	Returns:
		The accuracy on the total dataset
	
	"""
	log.info('Testing on seperate dataset: "' + dataset +'".')
	data = Dataset(dataset, 1, 2, seed=seed)
	acc = test(log, model, data, abs=abs)
	return acc

def test(log, model, data, abs=None):
	"""Tests a model on the given cross validated dataset.
	
	It requires the training procedure to have named the accuracy, x and y
	tensors with these respective names before saving them.
	
	Args:
		log:	A logger object to track the progress
		model:	The base name of the save files (without extension)
		data:	The Dataset object with the test set test on
		abs:	The evaluation abstraction: '1', '2', '3', '<1', or 'leafs'
	Returns:
		The accuracy of the cross validated test
	
	"""
	log.info('Testing network:  "' + str(model) + '" with abs=%s ...' %abs)
	tf.reset_default_graph()
	sess = tf.Session()
	saver = tf.train.import_meta_graph(model + '.meta')
	saver.restore(sess, model)
	data.new_batch_generator('test', ev_abs=abs)
	A = []
	for i, y, x in data.batch_gen:
		A.append(sess.run('accuracy:0', feed_dict={"x:0": x,
												   "y:0": y,
												   "keep_prob:0": 1.0,
												   "learn_r:0":1.0}))
	sess.close()
	log.info('Finished testing: "' + model + '".')
	log.info("Acc on TEST set:  " + str(np.mean(A)))
	try:
		tf.gfile.Remove(model + '.meta')
		tf.gfile.Remove(model + '.data-00000-of-00001')
		tf.gfile.Remove(model + '.index')
	except:
		pass
	return np.mean(A)

parser = argparse.ArgumentParser(prog="testing.py",
		description="Tests a model given a preprocessed dataset file.")
parser.add_argument('-n','--network-model',
					type=str,
					required=True,
					metavar='N',
					dest='model',
					help="The path to the model file (without extension)")
parser.add_argument('-d','--dataset',
					type=str,
					required=True,
					metavar='D',
					dest='dataset',
					help="The path to the preprocessed dataset txt file")
parser.add_argument('-s', '--seed',
					type=int,
					required=False,
					metavar='S',
					dest='seed',
					help="A cross validation seed to replicate results")
parser.add_argument('-t', '--abstraction',
					type=str,
					required=False,
					metavar='T',
					dest='abs',
					help="The taxonomical abstraction for the target labels")
parser.add_argument('-m', '--message',
					type=str,
					required=False,
					metavar='M',
					dest='msg',
					default=None,
					help="A message describing the purpose of this run, \
						  which will be logged before the program execution")

if __name__ == "__main__":
	main(sys.argv)
