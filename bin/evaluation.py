#! usr/bin/python
# filename:			evaluation.py
# author:			Joram Wessels
# date:				22-06-2017
# python versoin:	3.5
# dependencies:		MGR
# public functions:	prepare_data
# description:		Evaluates the MGR networks and parameters

import sys, logging
import numpy as np
from mgr_utils import MGRException
from mgr_utils import trackExceptions
from mgr_utils import log
from mgr_utils import log_exception
import cnn, k2c2, training
from dataset import Dataset

logging.basicConfig(filename='../logs/results.log', level=logging.DEBUG,
	format="%(asctime)s.%(msecs)03d: %(levelname)s: %(module)s."
	+ "%(funcName)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
results = logging.getLogger()

models = {'cnn':cnn,'k2c2':k2c2}
datasets = ['dataset-1.txt','dataset-2.txt']
abstractions = ['all','<1','2','1','leafs']
alphas = []
dropouts = [0.7, 0.75, 0.8]

batch_size = 40
k = 5
seed = 123
ev_abs='ev1'
model_save_dir = ".\\eval_models\\"
n_tests = 5

def main(argv):
	(m, v) = test_model_on(cnn, 1, 0.001, 0.75)

def test_model_on(mod, n_dat, a, d):
	id = str(n_dat)
	dat = datasets[n_dat-1]
	info = "%s on dataset '%s' using a=%.4f and dropout=%.2f" %(mod.__name__, dat, a, d)
	log.info(51*'=')
	log.info(51*'=')
	log.info("Evaluating " + info)
	log.info(51*'=')
	log.info(51*'=')
	results.info("Evaluating " + info)
	(m, v) = test_abstractions(id, mod, dat, a, d, abstractions[1:])
	results.info("\n\n Results: m = %.4f -- v = %.4f\n\n" %(m, v))
	log.info(51*'=' + '\n\n\n\n')
	log.info("Finished evaluating " + info)
	log.info("mean: %.4f, variance: %.4f" %(m,v))
	log.info(51*'=' + '\n\n')
	return (m,v)

@trackExceptions
def test_abstractions(id, mod, dat, a, d, values):
	acc = []
	for v in values:
		data = Dataset(dat, batch_size, k, abs=v, seed=seed)
		try:
			results.info("abstraction: %s\n" % v)
			idi = ('v1' if v == '<1' else v) + '_' + id
			(m,v) = train_n_times(idi, mod, dat, v, a, d, n_tests, data)
			acc.append(m)
			results.info("m = %.4f -- v = %.4f" %(m, v))
		except Exception as e:
			global err
			err += 1
			log_exception(e)
	m = np.mean(acc)
	v = np.var(acc)
	results.info("m = %.4f -- v = %.4f" %(m, v))
	log.info(51*'=' + '\n\n\n')
	log.info("m = %.4f -- v = %.4f" %(m, v))
	log.info('\n\n\n')
	log.info(51*'=')
	return (m, v)

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
			results.info("m = %.4f -- v = %.4f" %(m, v))
		except Exception as e:
			global err
			err += 1
			log_exception(e)
	m = np.mean(acc)
	v = np.var(acc)
	log.info(51*'=' + '\n')
	log.info("m = %.4f -- v = %.4f" %(m, v))
	log.info('\n')
	log.info(51*'=')
	return (m, v)

if __name__ == "__main__":
	main(sys.argv)