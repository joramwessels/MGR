#! usr/bin/python
# filename:			predicting.py
# author:			Joram Wessels
# date:				13-06-2017
# python versoin:	3.5
# dependencies:		tensorflow
# description:		unfinished untested

import sys, os, argparse
import tensorflow as tf
from dataset import Dataset
from mgr_utils import log
from mgr_utils import trackExceptions

def main(argv):
	p = parser.parse_args(argv[1:])
	if (p.msg):
		log.info(51*'=')
		log.info(p.msg)
	log.info(51*'=')
	if (os.path.isfile(p.data)): predict_file(p.model, p.data)
	elif (os.path.isdir(p.data)): predict_folder(p.model, p.data)
	else: raise MGRException(msg="Path does not exist: " + str(p.data))
	log.info(51*'=' + '\n')

@trackExceptions
def predict_folder(model, data_dir):
	"""Preprocesses the mp3 files in the data directory and predicts the labels.
	
	It requires the training procedure to have named the accuracy, x and y
	tensors with these respective names before saving them. The preprocessing
	will not be saved to storage, so repeated calls to the same data should use
	preprocessing.py and predict_file to save preprocessing time.
	
	Args:
		model:		The base name of the save files (without extension)
		data_dir:	The path to the root directory containing the mp3 files
	
	"""
	return

@trackExceptions
def predict_file(model, data_file, output='stdout'):
	"""Loads the preprocessed data file and predicts the labels.
	
	It requires the training procedure to have named the pred, x and y
	tensors with these respective names before saving them.
	
	Args:
		model:		The base name of the save files (without extension)
		data_file:	The preprocessed txt file containing the data
		output:		The location to write the predicted labels to
	
	"""
	log.info(51*'=')
	log.info('predicting the labels to: "' + data_file +'".')
	data = Dataset(data_file, 1, 1)
	sess = tf.Session()
	tf.reset_default_graph()
	saver = tf.train.import_meta_graph(model + '.meta')
	saver.restore(sess, model)
	p = sess.run("pred:0", feed_dict={"x:0": data.get_test_x()}) # does dis work?
	if (output == 'stdout'): print(p)
	sess.close()
	log.info('Finished predicting: "' + model + '".')
	return p

parser = argparse.ArgumentParser(prog="predicting.py",
		description="Predicts the labels to the data given a trained model.")
parser.add_argument('-n','--network-model',
					type=str,
					required=True,
					metavar='N',
					dest='model',
					help="The path to the model file (without extension)")
parser.add_argument('-d','--data',
					type=str,
					required=True,
					metavar='D',
					dest='dataset',
					help="The path to the preprocessed dataset txt file or " + \
						 "to the root directory containing the mp3 files")
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