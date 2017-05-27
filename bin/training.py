#! usr/bin/python
# filename:			training.py
# author:			Joram Wessels
# date:				11-05-2017
# python versoin:	2.7
# dependencies:		tensorflow
# public functions:	train
# description:		Reads the prepared data from a file and trains the network

import sys, json, logging
import numpy as np
import Choi2016

# Creates a logger
logging.basicConfig(filename='../logs/training.log', level=logging.DEBUG,
	format="%(asctime)s.%(msecs)03d: %(levelname)s: %(module)s."
	+ "%(funcName)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger("training")

def main(argv):
	data = read_from_file(argv[1])
	CV = cross_validate(data, 5)
	(train, test) = CV(1)
	coefficients = train_Choi2016(train)
	save_to_file(argv[2], coefficients)

def train_Choi2016(data):
	return

def cross_validate(data, k, seed=None):
	"""Divides the data into k partitions for k-fold cross validation
	If the data is not a multiple of k, the last partition gets the leftovers
	
	Args:
		data:		The entire preprocessed dataset
		k:			The amount of partitions to divide the data into
		seed:		If required, a random gen seed to replicate the results
	Returns:
		A closure that returns a (train, test) tuple for the given fold
			Args:
				fold:	The required fold
			Returns:
				A tuple of train and test sets
			Raises:
				Exception:	If 'fold' is invalid
	Raises:
		Exception:	If 'k' is invalid
	
	"""
	if (len(data) < k): raise Exception("Length of data < k")
	if (seed): np.random.seed(seed=seed)
	np.random.shuffle(data)
	l = len(data)/k
	folds = [(data[l*i-l:l*i] if i<k else data[l*i:]) for i in range(1,k+1)]
	def func(fold):
		if (fold > k or fold < 1):
			raise Exception(str(fold) +" fold does not exist")
		train = [e for f in (folds[:fold-1] + folds[fold:]) for e in f]
		return (train, folds[fold-1])
	return func

def read_from_file(filename):
	"""Reads out the dataset from storage
	
	Args:
		filename:	The path to the dataset file
	Returns:
		A list of tuples with a target (str) and a spectrogram (numpy.array)
	
	"""
	file = open(filename, 'r')
	data = []
	for line in file:
		try:
			l = line.split(';')
			data.append([l[0], np.array(json.loads(l[1]))])
		except Exception as e:
			log.error(str(e))
	return data

def save_to_file(filename, coefficients):
	"""Saves the results of the training to the specified output file
	
	Args:
		filename:		The path to the file that will contain the data
		coefficients:	The results of the training
	
	"""
	out = open(filename, 'w')
	# don't know what the results will look like yet
	out.close()

if __name__ == "__main__":
	main(sys.argv)