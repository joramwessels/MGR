#! usr/bin/python
# filename:			preprocessing.py
# author:			Joram Wessels
# date:				10-05-2017
# python versoin:	2.7
# dependencies:		librosa
# public functions:	prepare_data
# description:		Prepares the raw data for the training

import sys, logging
from os import walk
import librosa
import numpy as np
from mutagen.id3 import ID3

# Creates a logger
logging.basicConfig(filename='../logs/preprocessing.log', level=logging.DEBUG,
	format="%(asctime)s.%(msecs)03d: %(levelname)s: %(module)s."
	+ "%(funcName)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger("preprocessing")

def main(argv):
	prepare_data(argv[1])

def prepare_data(folder):
	"""Processes the raw data into trainable datapoints
	
	Args:
		folder:		The path to the folder containing all mp3 files
	Returs:
		A list with tuples containing spectrograms (numpy.array) and targets (str)
	"""
	err = False
	raw_data = []
	for (dirpath, dirnames, filenames) in walk(folder):
		raw_data.extend([dirpath + f for f in filenames if f.endswith(".mp3")])
	processed = []
	for filename in raw_data:
		try:
			target = str(ID3(filename)["TCON"].text[0])
			spectrogram = to_spectrogram(filename)
			processed.append([spectrogram, target])
		except Exception as e:
			err = True
			log.error(str(e))
	if (err): print("Exception(s) caught and logged during preprocessing")
	return processed

def to_spectrogram(filename, n_mels=128):
	"""Computes the Mel Spectrogram for the first minute of a given mp3 file
	
	Args:
		filename:	The path to the mp3 file
	Returns:
		A numpy.array representing the Mel spectrogram
	
	"""
	y, sr = librosa.load(filename, offset=0, duration=60)
	S = librosa.feature.melspectrogram(y, sr=sr, n_mels=n_mels)
	mel_spectrogram = librosa.logamplitude(S, ref_power=np.max)
	return mel_spectrogram

if __name__ == "__main__":
	main(sys.argv)