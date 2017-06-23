#! usr/bin/python
# filename:			evaluation.py
# author:			Joram Wessels
# date:				22-06-2017
# python versoin:	3.5
# dependencies:		MGR
# public functions:	prepare_data
# description:		Evaluates the MGR networks and parameters

import sys, logging, os, json
import numpy as np
from mgr_utils import MGRException
from mgr_utils import trackExceptions
from mgr_utils import log
from mgr_utils import log_exception
import cnn, k2c2, training
from dataset import Dataset

formatter = logging.Formatter("%(asctime)s.%(msecs)03d: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
handler = logging.FileHandler('../logs/results.log')        
handler.setFormatter(formatter)
results = logging.getLogger('results')
results.setLevel(logging.DEBUG)
results.addHandler(handler)

models = {'cnn':cnn,'k2c2':k2c2}
datasets = ['dataset-1.txt','dataset-2.txt']
abstractions = ['all','<1','1','2','leafs']
alphas = [0.002, 0.01, 0.05]
dropouts = [0.7, 0.75, 0.8]

batch_size = 40
k = 5
seed = 123
ev_abs='ev1'
model_save_dir = "." + os.path.sep + "eval_models" + os.path.sep
n_tests = 5

#all_results = np.zeros((len(models), len(datasets), len(abstractions), len(dropouts)),dtype=np.float32)

def main(argv):
	try:
		score = []
		score.append(test_model_on(cnn, 1))
		score.append(test_model_on(k2c2, 1))
		score.append(test_model_on(cnn, 2))
		score.append(test_model_on(k2c2, 2))
		results.info(30*'=')
		results.info(30*'=')
		results.info(30*'=')
		results.info("CNN - dataset-1: " + str(score[0]))
		results.info("k2c2 - dataset-1: " + str(score[1]))
		results.info("CNN - dataset-2: " + str(score[2]))
		results.info("k2c2 - dataset-2: " + str(score[3]))
		results.info("Dataset-1: " + str(np.mean([score[0][0], score[1][0]])))
		results.info("Dataset-2: " + str(np.mean([score[2][0], score[3][0]])))
		results.info("CNN: " + str(np.mean([score[0][0], score[2][0]])))
		results.info("k2c2: " + str(np.mean([score[1][0], score[3][0]])))
		results.info(30*'=')
		results.info(30*'=')
		results.info(30*'=' + '\n\n\n')
	except Exception as e:
		log_exception(e)
	store_all_results(argv[1])

def test_model_on(mod, n_dat):
	id = str(n_dat)
	dat = datasets[n_dat-1]
	info = "%s on dataset '%s'" %(mod.__name__, dat)
	log.info(51*'=')
	log.info(51*'=')
	log.info("Evaluating " + info)
	log.info(51*'=')
	log.info(51*'=')
	results.info("Evaluating " + info)
	(m, v) = test_abstractions(id, mod, dat, abstractions)
	log.info(51*'=' + '\n\n\n\n')
	log.info("Finished evaluating " + info)
	log.info("mean: %.4f, variance: %.4f" %(m,v))
	log.info(51*'=' + '\n\n')
	return (m,v)

@trackExceptions
def test_abstractions(id, mod, dat, values):
	acc = []
	for v in values:
		data = Dataset(dat, batch_size, k, abs=v, seed=seed)
		try:
			results.info("abstraction: %s" %v)
			idi = ('v1' if v == '<1' else v) + '_' + id
			(m,v) = test_alpha(idi, mod, dat, v, alphas, data)
			acc.append(m)
		except Exception as e:
			global err
			err += 1
			log_exception(e)
			results.info("Error caught")
	m = np.mean(acc)
	v = np.var(acc)
	max = np.max(acc)
	min = np.min(acc)
	result.info(30*'=')
	results.info("Model: %s   Dataset: %s" %(mod.__name__, dat))
	results.info("m = %.4f -- v = %.4f -- max = %.4f -- min = %.4f" %(m, v, max, min))
	result.info(30*'=' + '\n')
	log.info(51*'=' + '\n\n\n')
	log.info("m = %.4f -- v = %.4f" %(m, v))
	log.info('\n\n\n')
	log.info(51*'=')
	return (max, v)

def test_alpha(id, mod, dat, abs, values, data):
	acc = []
	for v in values:
		results.info("alpha: %.3f" %v)
		idi = str(v) + '_' + id
		(m,v) = test_dropout(idi, mod, dat, abs, v, dropouts, data)
		acc.append(m)
	m = np.mean(acc)
	v = np.var(acc)
	max = np.max(acc)
	min = np.min(acc)
	results.info("for abs=%s: m = %.4f -- v = %.4f -- max = %.4f -- min = %.4f\n\n" %(abs, m, v, max, min))
	log.info(51*'=' + '\n')
	log.info("m = %.4f -- v = %.4f" %(m, v))
	log.info(51*'=')
	return (max, v)

def test_dropout(id, mod, dat, abs, alp, values, data):
	acc = []
	for v in values:
		results.info("dropout: %.2f" %v)
		idi = str(v) + '_' + id
		(m,var) = train_n_times(idi, mod, dat, abs, alp, v, n_tests, data)
		acc.append(m)
		results.info("for do=%v: m = %.4f -- v = %.4f\n" %(v, m, var))
	m = np.mean(acc)
	v = np.var(acc)
	max = np.max(acc)
	min = np.min(acc)
	results.info("for a=%.3f: m = %.4f -- v = %.4f -- max = %.4f -- min = %.4f\n" %(alp, m, v, max, min))
	log.info(51*'=' + '\n')
	log.info("m = %.4f -- v = %.4f" %(m, v))
	log.info(51*'=')
	return (max, v)

@trackExceptions
def train_n_times(id, mod, dat, abs, alp, dro, n, data):
	acc = []
	for i in range(n):
		try:
			data.cross_validate()
			idi = str(i) + '_' + id
			(m,v) = training.train(mod, dat, batch_size=batch_size, k=k, id=idi,
									savedir=model_save_dir, tr_abs=abs,
									ev_abs=ev_abs, lr=alp, do=dro, seed=seed, data=data)
			acc.append((m,v))
			results.info(" v = %.4f" %v)
		except Exception as e:
			global err
			err += 1
			log_exception(e)
	m = np.mean(acc)
	v = np.var(acc)
	#all_results[list(models.keys()).index(mod.__name__),\
	#			datasets.index(dat),\
	#			abstractions.index(abs),\
	#			alphas.index(alp),\
	#			dropouts.index(dro)] = m
	log.info(51*'=' + '\n')
	log.info("m = %.4f -- v = %.4f" %(m, v))
	log.info('\n')
	log.info(51*'=')
	for (path, dirs, files) in os.walk(model_save_dir):
		for f in files:
			if (f.endswith(".data-00000-of-00001") or f.endswith(".index")):
				try:
					os.remove(f)
				except:
					pass
	return (m, v)

def store_all_results(filename):
	with open(filename, 'w') as out:
		json.dump(all_results, out)

def read_previous_results(filename):
	file = open(filename, 'r')
	return json.loads(next(file))

if __name__ == "__main__":
	main(sys.argv)
