#! usr/bin/python
# filename:			training.py
# author:			Joram Wessels
# date:				11-05-2017
# python versoin:	3.5
# dependencies:		tensorflow
# public functions:	train
# description:		Reads the prepared data from a file and trains the network

import sys, os, argparse
import numpy as np
from dataset import Dataset
import mgr_utils
from mgr_utils import log
from mgr_utils import log_exception
from mgr_utils import trackExceptions
import testing
import k2c2		# The (Choi, 2016) K2C2 implementation
import cnn		# The 3-layer CNN implementation
import rnn		# The single layer LSTM RNN implementation

network_types = {'cnn':cnn, 'k2c2':k2c2, 'rnn':rnn}

def main(argv):
	p = parser.parse_args(argv[1:])
	log.info('\n')
	if (p.msg):
		log.info(51*'=')
		log.info(p.msg)
	log.info(51*'=')
	network = network_types[p.network]
	train(network, p.data, batch_size=p.bs, k=p.k, id=p.id, savedir=p.output, tr_abs=p.tra, ev_abs=p.eva, seed=p.seed)
	log.info(51*'=' + '\n')

@trackExceptions
def train(network,
		  dataset,
		  batch_size=1,
		  k=1,
		  id=None,
		  savedir="./models/",
		  tr_abs='all',
		  ev_abs='all',
		  lr=.05, #was .001
		  do=0.75,
		  seed=None,
		  data=None):
	"""Trains a tensorflow network and tests it using k-fold cross validation
	
	Args:
		network:	A module that trains a model and returns the save filename
		dataset:	The path to the txt file containing the preprocessed dataset
		batch_size:	The batch size used during training (defaults to stochastic)
		k:			The amount of folds used in cross validation (defaults to 1)
		id:			An optional name for the model to help find the files later
		savedir:	The path to the directory in which to save all model files
		tr_abs:		The training abstraction: '1', '2', '3', '<1', or 'leafs'
		ev_abs:		The evaluation abstraction: '1', '2', '3', '<1', or 'leafs'
		seed:		A cross validation seed to replicate the results
	
	"""
	log.info('Started training on dataset: "%s", ' %dataset \
			+ 'network=%s, batch_size=%i, ' %(network.__name__, batch_size) \
			+ 'k=%i, id=%s, savedir="%s", ' %(k, str(id), savedir) \
			+ 'tr_abs=%s, ev_abs=%s, seed=%s' %(tr_abs, ev_abs, str(seed)))
	dir = os.path.dirname(savedir) + os.path.sep
	if (not os.path.exists(dir)): os.mkdir(dir)
	id = (str(id) + '-' + network.__name__ if id else network.__name__) + '-%i'
	if (not data): data = Dataset(dataset, batch_size, k, abs=tr_abs, seed=seed)
	acc = []
	for fold in range(k):
		try:
			data.new_batch_generator('train')
			log.info(51*'=')
			savefile = network.train(log, data, dir=dir, id=(id %fold), a=lr, do=do)
			log.info(51*'=')
			a = None
			if (savefile): a = testing.test(log, savefile, data, abs=ev_abs)
			if (a): acc.append(a)
			data.next_fold()
		except Exception as e:
			global err
			err += 1
			log_exception(e)
	log.info(51*'=')
	log.info(17*'=' + "Training Complete" + 17*'=')
	log.info("Network:     " + network.__name__)
	log.info("learn_r:     " + str(lr))
	log.info("dropout:     " + str(do))
	log.info("train_abs:   " + tr_abs)
	log.info("Dataset:     " + dataset)
	log.info("Samples:     " + str(data.get_size()))
	log.info("Validation:  " + str(k) + "-fold cross validation")
	log.info("Accuracy:    " + str(acc))
	log.info("Average:     " + str(np.mean(acc)))
	log.info(str(mgr_utils.err_total) + " error(s) caught during runtime")
	log.info(51*'=')
	return np.mean(acc), np.var(acc)

parser = argparse.ArgumentParser(prog="training.py",
		description="Trains a model given a preprocessed dataset file.")
parser.add_argument('-d','--data',
					type=str,
					required=True,
					metavar='D',
					dest='data',
					help="The path to the preprocessed dataset txt file")
parser.add_argument('-n','--network',
					type=str,
					required=True,
					metavar='N',
					choices=network_types,
					dest='network',
					help="The type of network to train")
parser.add_argument('-id', '--id',
					type=str,
					required=False,
					metavar='ID',
					dest='id',
					help="An identifier for the output files to help find them")
parser.add_argument('-o','--output',
					type=str,
					required=False,
					default='.' + os.path.sep + 'models' + os.path.sep,
					metavar='O',
					dest='output',
					help="The output folder for the model files")
parser.add_argument('-b','--batch-size',
					type=int,
					required=False,
					default=1,
					metavar='B',
					dest='bs',
					help="The standard size of each batch")
parser.add_argument('-f','--folds',
					type=int,
					required=False,
					default=1,
					metavar='F',
					dest='k',
					help="The amount of folds in the k-fold cross validation")
parser.add_argument('-s', '--seed',
					type=int,
					required=False,
					metavar='S',
					dest='seed',
					help="A cross validation seed to replicate results")
parser.add_argument('-tra', '--train_abstraction',
					type=str,
					required=False,
					metavar='trA',
					dest='tra',
					help="The train abstraction for the target labels")
parser.add_argument('-eva', '--eval_abstraction',
					type=str,
					required=False,
					metavar='evA',
					dest='eva',
					help="The evaluation abstraction for the target labels")
parser.add_argument('-m', '--message',
					type=str,
					nargs='?',
					required=False,
					metavar='M',
					dest='msg',
					default=None,
					help="A message describing the purpose of this run, \
						  which will be logged before the program execution")

if __name__ == "__main__":
	main(sys.argv)