#! usr/bin/python
# filename:			dataset.py
# author:			Joram Wessels
# date:				12-06-2017
# python versoin:	3.5
# dependencies:		numpy
# description:		The Dataset class reads, processes, and distributes all data

import os, json
import numpy as np
from mgr_utils import log_exception
from mgr_utils import MGRException
from mgr_utils import trackExceptions

class Dataset:
	"""Handles everything surrounding the dataset
	
	Args:
		filename:	The path to the preprocessed dataset
		batch_size:	The size of each batch fed to the network
		k:			The amount of cross validation partitions
		seed:		If required, a random gen seed to replicate CV results
	Attributes:
		data:		The complete dataset as a list of (label, np.matrix) tuples
		filename:	The path to the preprocessed dataset
		batch_size:	The size of each batch fed to the network
		k:			The amount of cross validation partitions
		fold:		The current cross validation fold index (i.e. fold - 1 )
		folds:		The partitioned dataset
		train:		The training partition of the current fold
		test:		The testing partition of the current fold
		batch_gen:	The batch generator closure
		decoder:	The dictionary keeping track of class label codation
		dec_iter:	The iterator keeping track of the number of class labels
	Raises:
		MGRException:	
	
	"""
	def __init__(self, filename, batch_size, k, seed=None):
		if (not(os.path.isfile(filename))):
			raise MGRException(msg="File does not exist: " + str(filename))
		if (not(type(batch_size) is int) or batch_size < 1):
			raise MGRException(msg="Invalid batch size: " + str(batch_size))
		if (k < 1 or not(type(k) is int)):
			raise MGRException(msg="Invalid k value: " + str(k))
		if (seed and (not(type(seed) is int) or seed < 1)):
			raise MGRException(msg="Invalid seed: " + str(seed))
		self.filename = filename
		self.batch_size = batch_size
		self.decoder = {}
		self.dec_iter = 0
		self.k = k
		self.data = read_from_file(self)
		if (seed): np.random.seed(seed=seed)
		np.random.shuffle(self.data)
		self.cross_validate()
	
	def cross_validate(self):
		"""Collects references to the data for each fold
		
		If the data is not perfectly divisable by k, the last partition gets
		the leftovers. The resulting datastructure is saved in self.folds as a
		4D [fold, train/test, partition, start/end] sequence, along with the
		current fold in self. fold.
		
		Raises:
			MGRException:	If 'k' is invalid
		
		"""
		if (self.k > 1):
			if (len(self.data) < self.k):
				raise MGRException(msg="Length of data < k")
			l = int(len(self.data)/self.k)
			overflow = int(len(self.data)%self.k)
			part = [(l*f-l,l*f) if f<self.k else (l*f-l, l*f+overflow) \
							for f in range(1, self.k+1)]
			self.folds = [([(part[p] if (p < f) else part[p+1]) \
				for p in range(self.k-1)],[part[f]]) for f in range(self.k)]
		else:
			self.folds = [([(0, len(self.data))], [(0, len(self.data))])]
		self.fold = 0
	
	def next_fold(self):
		"""Sets the internal data representation to the next CV fold
		
		Raises:
			MGRException: If there are no more folds left
		
		"""
		if (self.fold >= self.k):
			raise MGRException(msg="There is no next cross validation fold")
		self.fold += 1
	
	def new_batch_generator(self, mode):
		"""(Re)sets the batch generator using the given mode
		
		The resulting generator in self.batch_gen yields
		(id, labels_batch, images_batch) tuples ready for training/testing. The
		generator can be run by calling next_batch() on the Dataset object. It
		goes through the trouble of referencing the self.data object in order
		to avoid holding the entire dataset in memory numerous times. This way,
		only the original self.data object and the current batch are held in
		memory.
		
		Args:
			mode:	Either 'train' or 'test'
		Raises:
			MGRException: If mode is invalid
		
		"""
		if (not(mode == 'train' or mode == 'test')):
			raise MGRException(msg="Invalid batch generator mode: " + str(mode))
		def gen():
			if (mode == 'train'): folds = self.folds[self.fold][0]
			elif (mode == 'test'): folds = self.folds[self.fold][1]
			partition = 0
			rng = folds[partition]
			n = 0
			batch = self.batch_size*[0]
			residue = self.batch_size*[0]
			extras = self.batch_size*[0]
			while (partition < len(folds)-1 or n < rng[1] - rng[0]):
				loc = rng[0] + n
				dst = loc + self.batch_size
				if (dst <= rng[1]):
					batch = self.data[loc:dst]
				else:
					residue = self.data[loc:rng[1]]
					if (partition < len(folds)-1):
						n = 0
						dst -= rng[1]
						partition += 1
						rng = folds[partition]
						dst += rng[0]
						extras = self.data[rng[0]:dst]
						batch = residue + extras
					else:
						batch = residue
				n = dst-rng[0]
				ids = [s[0] for s in batch]
				labels = [[(1 if i in l else 0) for i in range(self.dec_iter)] \
												for l in [s[1] for s in batch]]
				yield (ids, labels, [np.reshape(s[2], -1) for s in batch])
		self.batch_gen = gen()
	
	def next_batch(self):
		"""Executes the batch generator closure and returns the next batch
		
		Returns:
			A (labels <list>, images <np.array>) tuple of size batch_size
		
		"""
		return self.batch_gen.__next__()
	
	def encode_label(self, label):
		"""Encodes a class label string to an int and adds the key to the dict
		
		Args:
			label:	The label to encode
		Returns:
			The encoded label integer
		
		"""
		if not(label in self.decoder):
			self.decoder[label] = self.dec_iter
			self.decoder[str(self.dec_iter)] = label
			self.dec_iter += 1
		return self.decoder[label]
	
	def decode_label(self, code):
		"""Decodes a class label code to its original string
		
		Args:
			code:	The integer code returned by the network
		Returns:
			The original class label string represented by the code
		
		"""
		if not(str(code) in self.decoder):
			raise MGRException(msg="Unknown label code: " + str(code))
		else:
			return self.decoder[str(code)]
	
	def get_n_classes(self):
		return self.dec_iter
	
	def get_size(self):
		return len(self.data)
	
	def get_data_dim(self):
		return (len(self.data[0][2]), len(self.data[0][2][0]))
	
	def get_train_ids(self):
		p = self.folds[self.fold][0]
		return [l[0] for (b,e) in p for l in self.data[b:e]]
	
	def get_train_y(self):
		# Retrieves the begin and end of all train partitions, loops through all
		# their targets (index 1), creates an array of length dec_iter\
		# (n_classes) and adds a one to those indices that occur in the target
		# array, such that the output for target [1,4] is [0,1,0,0,1,0,...,0].
		p = self.folds[self.fold][0]
		return [[(1 if i in l else 0) for i in range(self.dec_iter)] \
						for l in [s[1] for (b,e) in p for s in self.data[b:e]]]
	
	def get_train_x(self):
		p = self.folds[self.fold][0]
		return np.asarray([np.reshape(l[2], -1) for (b,e) in p for l in self.data[b:e]], \
							dtype=np.float32)
	
	def get_test_ids(self):
		(b, e) = self.folds[self.fold][1][0]
		return [l[0] for l in self.data[b:e]]
	
	def get_test_y(self):
		# Retrieves the begin and end of the test partition, loops through all
		# its targets (index 1), creates an array of length dec_iter (n_classes)
		# and adds a one to those indices that occur in the target array, such
		# that the output for target [1,4] is [0,1,0,0,1,0,...,0].
		(b, e) = self.folds[self.fold][1][0]
		return [[(1 if i in l else 0) for i in range(self.dec_iter)] \
									  for l in [l[1] for l in self.data[b:e]]]
	
	def get_test_x(self):
		(b, e) = self.folds[self.fold][1][0]
		return np.asarray([np.reshape(l[2], -1) for l in self.data[b:e]], \
							dtype=np.float32)

@trackExceptions
def read_from_file(dataset):
	"""Reads out the dataset from storage
	
	Args:
		dataset:	The Dataset object that will manage the data
	Returns:
		A list of tuples with a target (str) and a spectrogram (numpy.array)
	
	"""
	#global err
	file = open(dataset.filename, 'r')
	data = []
	for line in file:
		l = line.strip().split(';')
		if (len(l) == 3):
			try:
				targets = [dataset.encode_label(c) for c in l[1].split('/')]
				data.append([l[0], targets, json.loads(l[2])])
			except Exception as e:
				log_exception(e)
	return data