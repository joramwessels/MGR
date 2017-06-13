#! usr/bin/python
# filename:			mgr.py
# author:			Joram Wessels
# date:				12-06-2017
# python versoin:	3.5
# dependencies:		numpy, json, tensorflow, mutagen, ffmpeg, librosa
# description:		Handles command line and executes package modules

import sys, argparse
import preprocessing, training, testing, predicting

actions = {'preprocess':preprocessing,'train':training, \
			'test':testing, 'predict':predicting}

def main(argv):
	#actions = [f.split(' ') for f in ' '.join(argv).split('-a')][1:]
	p = parser.parse_args(sys.argv[1:])
	for i in range(len(p.a)):
		param = (p.p[i] if len(p.p) > i else '')
		actions[p.a[i]].parser.parse_args(param)
	for i in range(len(p.a)):
		param = (p.p[i] if len(p.p) > i else '')
		actions[p.a[i]].main(param)

parser = argparse.ArgumentParser(prog="Music Genre Recognizer",
	description="Trains a network to recognize electronic music genres",
	usage="An action is given in the form '-a <action> -p \"arg1 arg2 ...\"'."+\
		"A sequence of actions can be given, separated by '--action <action>'")
parser.add_argument('-a','--action',
					action='append',
					type=str,
					choices=actions,
					required=True,
					metavar='A',
					dest='a',
					help="The action to perform")
parser.add_argument('-p','--parameters',
					action='append',
					nargs='?',
					type=str,
					required=True,
					metavar='P',
					dest='p',
					help="The parameters passed straight to the requested file")

if __name__ == "__main__":
	main(sys.argv)