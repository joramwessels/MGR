#! usr/bin/python
# filename:			preprocessing.py
# author:			Joram Wessels
# date:				10-05-2017
# python versoin:	2.7
# dependencies:		numpy, librosa, mutagen
# public functions:	prepare_data
# description:		Prepares the raw data for the training

import sys, os
import numpy as np
from json import dumps
from librosa import load
from librosa.core.spectrum import power_to_db
from librosa.feature import melspectrogram
from mutagen.id3 import ID3
from mutagen.mp3 import MP3
from mgr_utils import MGRException
from mgr_utils import trackExceptions
from mgr_utils import log
from mgr_utils import log_exception

# Sample rate (Hz) and duration (s) of music samples
duration = 29
sample_rate = 12000

def main(argv):
	prepare_data(argv[1], argv[2], 1280)

# n_fft=2048, stride=512, n_mels=128
@trackExceptions
def prepare_data(folder, destination, dim, n_fft=2048, stride=256, n_mels=96):
	"""Processes the raw data into trainable datapoints
	
	Args:
		folder:			The path to the folder containing all mp3 files
		destination:	The path to the output file that will store the data
		dim:			The required output dimension of audio samples
		n_fft:			The FFT window size
		stride:			The FFT windows step size
		n_mels:			The amount of Mel frequency bins
	
	"""
	global err
	folder = folder.replace('/', os.sep).replace('\\', os.sep)
	log.info("Started preprocessing of folder: " + folder)
	count = 0
	out = open(destination, 'w')
	raw_data = []
	for (dirpath, dirnames, filenames) in os.walk(folder):
		raw_data.extend([(os.path.join(dirpath, f),f) for f in filenames
							if f.endswith(".mp3")])
	for path, filename in raw_data:
		try:
			target = str(ID3(path)["TCON"].text[0])
			length = MP3(path).info.length
			spectrogram = to_spectrogram(path, (length/2)-(duration/2),
										duration, n_fft, stride, n_mels)
			if (len(spectrogram[0])<dim):
				raise MGRException(msg="Preprocessing of " + filename + " resu"\
				"lted in fewer than the requested " + str(dim) + " samples")
			elif(len(spectrogram[0])>dim):
				spectrogram = [list(l[:dim]) for l in spectrogram]
			else: spectrogram = list(spectrogram)
			out.write(filename + ';' + target + ';' + dumps(spectrogram) + '\n')
		except Exception as e:
			err += 1
			log_exception(e)
		count = count+1
		if (count % 500 == 0): log.info(count, "files processed...")
	out.close()
	log.info("Preprocessing complete. " + str(count) + " files processed." \
			 + str(err) + " errors caught and logged.")

def to_spectrogram(filename, ofs, dur, n_fft, stride, n_mels):
	"""Computes the Mel Spectrogram for the first minute of a given mp3 file
	
	Args:
		filename:	The path to the mp3 file
		ofs:		The offset in seconds
		dur:		The duration of the excerpt in seconds
		n_fft:		The FFT window size
		stride:		The FFT windows step size
		n_mels:		The amount of Mel frequency bins
	Returns:
		A numpy.array representing the Mel spectrogram
	
	"""
	y, sr = load(filename, sr=sample_rate, offset=ofs, duration=dur)
	S = melspectrogram(y, sr=sr, n_fft=n_fft, hop_length=stride, n_mels=n_mels)
	mel_spectrogram = power_to_db(S, ref=np.max)
	return mel_spectrogram

if __name__ == "__main__":
	main(sys.argv)