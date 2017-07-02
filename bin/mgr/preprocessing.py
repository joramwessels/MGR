#! usr/bin/python
# filename:			preprocessing.py
# author:			Joram Wessels
# date:				10-05-2017
# python versoin:	2.7
# dependencies:		numpy, librosa, mutagen
# public functions:	prepare_data
# description:		Prepares the raw data for the training

import sys, os, argparse
from numpy import max
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
	p = parser.parse_args(argv[1:])
	if (p.msg):
		log.info(51*'=')
		log.info(p.msg)
	log.info(51*'=')
	prepare_data(p.data_folder, p.output_file, p.max_dim,
				 n_fft=p.n_fft, stride=p.stride, n_mels=p.n_mels)
	log.info(51*'=' + '\n')

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
		if (count % 50 == 0): log.info(str(count) + "files processed...")
	out.close()
	log.info("Preprocessing complete. " + str(count) + " files processed." \
			 + str(err) + " errors caught and logged.")
	return

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
	mel_spectrogram = power_to_db(S, ref=max)
	return mel_spectrogram

parser = argparse.ArgumentParser(prog="preprocessing.py",
		description="Preprocesses a dataset file hierarchy into a txt file.")
parser.add_argument('-d','--dataset',
					type=str,
					required=True,
					metavar='D',
					dest='data_folder',
					help="The root directory of the dataset file hierarchy")
parser.add_argument('-o','--output',
					type=str,
					required=True,
					metavar='O',
					dest='output_file',
					help="The output file created by the preprocessing")
parser.add_argument('-x','--xdim',
					type=int,
					required=False,
					default=1280,
					metavar='X',
					dest='max_dim',
					help="The maximum frame length in the time dimension")
parser.add_argument('-w','--window',
					type=int,
					required=False,
					default=2048,
					metavar='W',
					dest='n_fft',
					help="The window size of the fast Fourier transform")
parser.add_argument('-s','--stride',
					type=int,
					required=False,
					default=256,
					metavar='S',
					dest='stride',
					help="The hop size of the fast Fourier transform")
parser.add_argument('-b','--melbins',
					type=int,
					required=False,
					default=96,
					metavar='M',
					dest='n_mels',
					help="The amount of Mel bins of the fast Fourier transform")
parser.add_argument('-m', '--message',
					type=str,
					required=False,
					metavar='M',
					default=None,
					dest='msg',
					help="A message describing the purpose of this run, \
						  which will be logged before the program execution")

if __name__ == "__main__":
	main(sys.argv)