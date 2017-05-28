#! usr/bin/python
# filename:			preprocessing.py
# author:			Joram Wessels
# date:				10-05-2017
# python versoin:	2.7
# dependencies:		numpy, librosa, mutagen
# public functions:	prepare_data
# description:		Prepares the raw data for the training

import sys, logging
from os import walk
import numpy as np
from librosa import load
from librosa.core.spectrum import power_to_db
from librosa.feature import melspectrogram
from mutagen.id3 import ID3

# Creates a logger
logging.basicConfig(filename='../logs/preprocessing.log', level=logging.DEBUG,
	format="%(asctime)s.%(msecs)03d: %(levelname)s: %(module)s."
	+ "%(funcName)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger("preprocessing")

def main(argv):
	prepare_data(argv[1], argv[2])

def prepare_data(folder, destination, n_fft=2048, stride=512, n_mels=128):
	"""Processes the raw data into trainable datapoints
	
	Args:
		folder:			The path to the folder containing all mp3 files
		destination:	The path to the output file that will store the data
		n_fft:			The FFT window size
		stride:			The FFT windows step size
		n_mels:			The amount of Mel frequency bins
	
	"""
	log.info("Started preprocessing of folder: " + folder)
	err = False
	count = 0
	out = open(destination, 'w')
	raw_data = []
	for (dirpath, dirnames, filenames) in walk(folder):
		raw_data.extend([dirpath + f for f in filenames if f.endswith(".mp3")])
	for filename in raw_data:
		try:
			target = str(ID3(filename)["TCON"].text[0])
			spectrogram = to_spectrogram(filename, n_fft, stride, n_mels)
			out.write(filename + '; ' + target + '; '
						+ str(spectrogram.tolist()) + '\n')
		except Exception as e:
			err = True
			log.error(str(e))
		count = count+1
		if (count % 500 == 0): log.info(count, "files processed...")
	out.close()
	log.info("Preprocessing complete. " + str(count) + " files processed.")
	if (err): print("Exception(s) caught and logged during preprocessing")

def to_spectrogram(filename, n_fft, stride, n_mels):
	"""Computes the Mel Spectrogram for the first minute of a given mp3 file
	
	Args:
		filename:	The path to the mp3 file
	Returns:
		A numpy.array representing the Mel spectrogram
	
	"""
	y, sr = load(filename, offset=0, duration=60)
	S = melspectrogram(y, sr=sr, n_fft=n_fft, hop_length=stride, n_mels=n_mels)
	mel_spectrogram = power_to_db(S, ref=np.max)
	return mel_spectrogram

if __name__ == "__main__":
	main(sys.argv)