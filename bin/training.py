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
import tensorflow as tf
from mgr_utils import log
from mgr_utils import MGRException
from mgr_utils import trackExceptions
import Choi2016		# The K2C2 Tensorflow implementation

def main(argv):
	train_Choi2016(argv[1], batch_size=argv[2], k=argv[3])

@trackExceptions
def train_Choi2016(filename, batch_size=50, k=1, savedir="./models/"):
	global err
	log.info("Started training of dataset: " + filename)
	data = read_from_file(filename)
	CV = cross_validate(data, k)
	acc = k*[0.0]
	for fold in range(1,k):
		try:
			(trainset, testset) = CV(1)
			train_batch_gen = batch_generator(trainset, batch_size)
			savefile = Choi2016.train(log, train_batch_gen, dir=savedir, id=1)
			test_batch_gen = batch_generator(testset, batch_size)
			acc[k-1] = Choi2016.test(log, savefile, test_batch_gen)
		except Exception as e:
			err += 1
			log.error(str(MGRException(ex=e)))
	log.info("=========================================")
	log.info("Training complete. K2C2 Choi2016 network trained on the " + \
			 filename + " dataset(" + str(len(data)) + " samples), using " + \
			 str(k) + "-fold cross validation. " + str(err) + " error(s) " + \
			 "were caught and logged. The average cross validated accuracy " + \
			 "is " + np.mean(acc))
	log.info("=========================================")

class Dataset:
	"""Handles everything surrounding the dataset
	
	Args:
		filename:	The path to the preprocessed dataset
		batch_size:	The size of each batch fed to the network
		k:			The amount of cross validation partitions
		seed:		If required, a random gen seed to replicate CV results
	Attributes:
		data:		The complete dataset as a numpy matrix
		filename:	The path to the preprocessed dataset
		batch_size:	The size of each batch fed to the network
		k:			The amount of cross validation partitions
		fold:		The current cross validation fold
		folds:		The 
		train:		The
		test:		The
	Raises:
		MGRException:	
	
	"""
	def __init__(self, filename, batch_size=50, k=1, seed=None):
		if (not(sys.path.isfile(filename))):
			raise MGRException(msg="File does not exist: " + str(filename))
		if (not(type(batch_size) is int) or batch_size < 1):
			raise MGRException(msg="Invalid batch size: " + str(batch_size))
		if (k < 1 or not(type(k) is int)):
			raise MGRException(msg="Invalid k value: " + str(k))
		if (not(type(seed) is int or type(seed) is Nonetype) or seed < 1):
			raise MGRException(msg="Invalid seed: " + str(seed))
		self.filename = filename
		self.batch_size = batch_size
		self.k = k
		self.data = self.read_from_file()
		if (k>1): self.cross_validate()
		else:
			self.fold = 1
			self.train = self.data
			self.test = self.data
		self.mode = 'train'
	
	def cross_validate(self):
		"""Divides the data into k partitions for k-fold cross validation
		
		If the data is not perfectly divisable by k, the last partition gets
		the leftovers.
		
		Returns:
			A closure that returns a (train, test) tuple for the given fold
				Args:
					fold:	The required fold (1-k)
				Returns:
					A tuple of train and test sets
				Raises:
					MGRException:	If 'fold' is invalid
		Raises:
			MGRException:	If 'k' is invalid
		
		"""
		if (self.seed): np.random.seed(seed=seed)
		np.random.shuffle(self.data)
		if (len(self.data) < self.k): raise MGRException(msg="Length of data < k")
		l = len(self.data)/self.k
		self.folds = [(self.data[l*i-l:l*i] if i<self.k else \
						self.data[l*i:]) for i in range(1,k+1)]
		self.fold = 0
		self.next_fold()
	
	def next_fold(self):
		"""Sets the internal data representation to the next CV fold
		
		Raises:
			MGRException: If there are no more folds left
		
		"""
		if (self.fold >= k): raise MGRException(msg="There is no next fold")
		self.fold += 1
		self.train = [e for f in (self.folds[:self.fold-1] + \
						self.folds[self.fold:]) for e in f]
		self.test = self.folds[self.fold-1]
	
	def next_batch(self):
		"""Generates the next batch and yields it as a tuple
		
		Yields:
			A (labels_batch, images_batch) tuple
		
		"""
		if (self.mode == 'train'): data = self.train
		elif (self.mode == 'test'): data = self.test
		for batch_id in range(0, len(data), self.batch_size):
			labels_batch = data[batch_id : batch_id + self.batch_size][:,1]
			images_batch = data[batch_id : batch_id + self.batch_size][:,2]
			yield (labels_batch, images_batch.astype("float32"))
	
	@trackExceptions
	def read_from_file(self):
		"""Reads out the dataset from storage
		
		Returns:
			A list of tuples with a target (str) and a spectrogram (numpy.array)
		
		"""
		global err
		file = open(self.filename, 'r')
		data = []
		for line in file:
			try:
				l = line.split(';')
				data.append([l[0], np.array(json.loads(l[1]))])
			except Exception as e:
				err += 1
				log.error(str(MGRException(ex=e)))
		return data
	
	def get_train_ids(self):
		return self.train[:,0]
	
	def get_train_y(self):
		return self.train[:,1]
	
	def get_train_x(self):
		return self.train[:,2]
	
	def get_test_ids(self):
		return self.test[:,0]
	
	def get_test_y(self):
		return self.test[:,1]
	
	def get_test_x(self):
		return self.test[:,2]
	
	def set_data_mode(self, mode):
		"""Sets the dataset required from now on
		
		Args:
			mode:	Either 'train' or 'test'
		Raises:
			MGRException: If mode is invalid
		
		"""
		if (mode == 'train' or mode == 'test'): self.mode = mode
		else: raise MGRException(msg="Invalid data type: " + mode)

if __name__ == "__main__":
	main(sys.argv)