#! usr/bin/python
# filename:			dataset.py
# author:			Joram Wessels
# date:				12-06-2017
# python versoin:	3.5
# dependencies:		numpy
# description:		The Dataset class reads, processes, and distributes all data

import os, json
import numpy as np
from mgr_utils import log
from mgr_utils import log_exception
from mgr_utils import MGRException
from mgr_utils import trackExceptions

class Dataset:
	"""Handles everything surrounding the dataset
	
	Args:
		filename:	The path to the preprocessed dataset
		batch_size:	The size of each batch fed to the network
		k:			The amount of cross validation partitions
		abs:		The genre abstraction: '1', '2', '3', '<1', or 'leafs'
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
	def __init__(self, filename, batch_size, k, abs='all', seed=None):
		log.info('Preparing dataset: "' + str(filename) + '" ...')
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
		self.data = read_from_file(self, abs=abs)
		if (seed): np.random.seed(seed=seed)
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
		np.random.shuffle(self.data)
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
	
	def new_batch_generator(self, mode, ev_abs=None):
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
				if (mode == 'test' and ev_abs):
					labels = [k_hot(self, resolve_targets(self, \
							self.decode_label(s[1][0]), ev_abs)) for s in batch]
				else:
					labels = [k_hot(self, l) for l in [s[1] for s in batch]]
				yield (ids, labels, [np.reshape(s[2], -1) for s in batch])
		self.batch_gen = gen()
	
	def next_batch(self):
		"""Executes the batch generator closure and returns the next batch
		
		Returns:
			A (labels <list>, images <np.array>) tuple of size batch_size
		
		"""
		return next(self.batch_gen)
	
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
		if not(str(int(code)) in self.decoder):
			raise MGRException(msg="Unknown label code: " + str(code))
		else:
			return self.decoder[str(int(code))]
	
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
		# their targets (index 1) and returns them in k-hot encoding.
		p = self.folds[self.fold][0]
		return [k_hot(self, s[1]) for (b,e) in p for s in self.data[b:e]]
	
	def get_train_x(self):
		p = self.folds[self.fold][0]
		return np.asarray([np.reshape(l[2], -1) for (b,e) in p \
									for l in self.data[b:e]], dtype=np.float32)
	
	def get_test_ids(self):
		(b, e) = self.folds[self.fold][1][0]
		return [l[0] for l in self.data[b:e]]
	
	def get_test_y(self):
		# Retrieves the begin and end of the test partition, loops through all
		# its targets (index 1) and returns them in k-hot encoding.
		(b, e) = self.folds[self.fold][1][0]
		return [k_hot(self, l[1]) for l in self.data[b:e]]
	
	def get_eval_y(self, abs='all'):
		"""Returns the correct y labels for evaluation.
		
		In addition to get_test_y() this function also resolves the correct
		level of genre abstraction. This way, you can train on e.g. leaf nodes
		and evaluate using the highest abstraction.
		
		Args:
			abs:	The abstraction flag. default='all'
		
		Returns:
			The test labels in 'k-hot' format to evaluate the model with
		
		"""
		(b, e) = self.folds[self.fold][1][0]
		return [k_hot(self, resolve_targets(self, self.decode_label(l[1][0]), abs))
				for l in self.data[b:e]]
	
	def get_test_x(self):
		(b, e) = self.folds[self.fold][1][0]
		return np.asarray([np.reshape(l[2], -1) for l in self.data[b:e]], \
							dtype=np.float32)

@trackExceptions
def read_from_file(dataset, abs='all'):
	"""Reads out the dataset from storage
	
	Args:
		dataset:	The Dataset object that will manage the data
		abs:		The taxonomical abstraction of genres used. default='all'
	Returns:
		A list of tuples with a target (str) and a spectrogram (numpy.array)
	
	"""
	file = open(dataset.filename, 'r')
	data = []
	for line in file:
		l = line.strip().split(';')
		if (len(l) == 3):
			try:
				tgts = resolve_targets(dataset, l[1].split('/')[0], abs)
				data.append([l[0], tgts, json.loads(l[2])])
			except Exception as e:
				global err
				err += 1
				try:
					log_exception(e, msg='File=' + l[0])
				except:
					log_exception(e)
	return data

def k_hot(data, labels):
	"""Converts a list of class labels to k-hot encoding
	
	Args:
		data:	The dataset object issueing the labels
		labels:	A list of integer encoded labels
	Returns
		A list of floats, the k-hot encoded target array
	
	"""
	return [(1.0 if i in labels else 0.0) for i in range(data.get_n_classes())]

def resolve_targets(dataset, tag, abs):
	"""Prepares the right level of taxonomical abstraction for the target(s)
	
	Args:
		dataset:	The Dataset object that manages the encoding
		tag:		The genre found in the ID3 'TCON' frame
		abs:		The taxonomical abstraction of genres used. This can by any
					of: '1', '2', '3', '<1', or 'leafs'. default='all'
	
	Returns:
		A list of encoded target variables
	
	"""
	targets_used = []
	t = tag
	if (abs == 'all'):
		while (not t == None):
			targets_used.append(t)
			t = parent_of[t]
	elif (abs == '1'):
		targets_used = [t]
		while (not t == None):
			targets_used[0] = t
			t = parent_of[t]
	elif (abs == '2'):
		targets_used = [t, t]
		if (parent_of[t] == None):
			raise MGRException(msg="Abstraction of target higher than %s" %abs)
		while (not t == None):
			targets_used[0] = targets_used[1]
			targets_used[1] = t
			t = parent_of[t]
		targets_used.pop()
	elif (abs == '3'):
		targets_used = [t, t, t]
		if (parent_of[t] == None or parent_of[parent_of[t]] == None):
			raise MGRException(msg="Abstraction of target higher than %s" %abs)
		while (not t == None):
			targets_used[0] = targets_used[1]
			targets_used[1] = targets_used[2]
			targets_used[2] = t
			t = parent_of[t]
		targets_used.pop()
		targets_used.pop()
	elif (abs == 'leafs'):
		targets_used = [t]
	elif (abs == '<1'):
		if (parent_of[t] == None):
			raise MGRException(msg="Abstraction of target higher than %s" %abs)
		while (not t == None):
			targets_used.append(t)
			t = parent_of[t]
		targets_used.pop()
	elif (abs == 'ev1'):
		pr = t
		while (not t == None):
			pr = t
			t = parent_of[t]
		targets_used = [pr] + children[pr]
		for l in children[pr]:
			for c in children_of(l):
				targets_used += c
		targets_used = [l for l in targets_used if l in dataset.decoder]
	else:
		return resolve_targets(dataset, tag, 'all')
	return [dataset.encode_label(t) for t in targets_used]

"""The genre taxonomy, used to determine the training targets in the dataset
   initalization, and to compute the accuracy in the network module.
"""
parent_of = {'Trap':'Hip Hop', 'Chillhop':'Hip Hop','Future Bass':'Trap',
			 'Complextro':'Electro House', 'Big Room House':'Electro House',
			 'Electro House':'House', 'Progressive House':'House',
			 'Deep House':'House','Future House':'Deep House',
			 'Happy Hardcore':'Hardcore',
			 'Hardcore':'Hard Dance','Hardstyle':'Hard Dance',
			 'Liquid Funk':'Drum & Bass','Neurofunk':'Drum & Bass',
			 'Hip Hop':None, 'House':None, 'Hard Dance':None,
			 'Dubstep':'Bass','Drum & Bass':'Bass','Bass':None}

children = {'Hip Hop':['Trap','Chillhop'],'Trap':['Future Bass'],
			'House':['Electro House','Progressive House','Deep House'],
			'Electro House':['Complextro','Big Room House'],
			'Deep House':['Future House'],
			'Hard Dance':['Hardcore','Hardstyle'],'Hardcore':['Happy Hardcore'],
			'Bass':['Drum & Bass','Dubstep'],
			'Drum & Bass':['Liquid Funk','Neurofunk']}

def children_of(label):
	if (label not in children): return []
	ch = children[label]
	for l in children[label]:
		for c in children_of(l):
			ch += c
	return children[label] + [c for l in children[label] for c in children_of(l)]
