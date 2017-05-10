#! usr/bin/python
# filename:			preprocessing.py
# author:			Joram Wessels
# date:				10-05-2017
# python versoin:	2.7
# dependencies:		librosa
# public functions:	
# description:		

import sys, logging
from os import listdir
from os.path import isfile, join
import librosa

logging.basicConfig(filename='../logs/preprocessing.log')
log = logging.getLogger(preprocessing)

def main(argv):
	prepare_data(argv[1])

def prepare_data(folder):
	"""Processes the raw data into trainable datapoints
	
	Args:
		folder:		The path to the folder containing all mp3 files
	Returs:
		A list with tuples containing spectrograms (numpy.array) and targets (str)
	"""
	raw_data = []
	for (dirpath, dirnames, filenames) in walk(main_data_folder):
		raw_data.extend([f for f in filenames if f.endswith(".mp3")])
	processed = []
	for filename in raw_data:
		try:
			target = str(ID3(filename)["TCON"].text[0])
			spectrogram = to_spectrogram(filename)
			processed.append([spectrogram, target])
		except e:
			logging.warning("Exception caught in preprocessing")
			log.error(str(e))
	return processed

def to_spectrogram(filename):
	"""Computes the Mel Spectrogram of a given mp3 file
	
	Args:
		filename:	The path to the mp3 file
	Returns:
		A numpy.array representing the Mel spectrogram
	
	"""
	return

if __name__ == "__main__":
	main(sys.argv)