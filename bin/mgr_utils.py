#! usr/bin/python
# filename:			mgr_utils.py
# author:			Joram Wessels
# date:				29-05-2017
# python versoin:	3.5
# dependencies:		None
# public functions:	None
# description:		Provides logging, exceptions and decorators

import sys, traceback, logging

log_mode = 'monitor'
"""There are 3 log modes:

	debug:	  Presents exceptions as they are raised and prints info to stdout
	run:	  Logs info and exceptions without halting the execution
	monitor:  Logs the same way 'run' does, but also prints everyting to stdout
"""

err_total=0

formatter = logging.Formatter("%(asctime)s.%(msecs)03d: %(levelname)s: %(module)s."
		+ "%(funcName)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
std_out = logging.StreamHandler(sys.stdout)
std_out.setFormatter(formatter)
std_out.setLevel(logging.DEBUG)
err_out = logging.FileHandler('../logs/exceptions.log')
err_out.setFormatter(formatter)
err_out.setLevel(logging.ERROR)
info_out = logging.FileHandler('../logs/monitor.log')
info_out.setFormatter(formatter)
info_out.setLevel(logging.DEBUG)
log = logging.getLogger('log')

if (log_mode == 'debug'):
	log.addHandler(std_out)
elif (log_mode == 'run'):
	log.addHandler(info_out)
	log.addHandler(err_out)
elif (log_mode == 'monitor'):
	log.addHandler(info_out)
	log.addHandler(err_out)
	log.addHandler(std_out)
else:
	print('\n\n\t\t- Invalid log mode: ' + str(log_mode) + '\n\n')
	sys.exit(1)

if (log_mode == 'debug'):
	logging.basicConfig(stream=sys.stdout, level=logging.DEBUG,
		format="%(asctime)s.%(msecs)03d: %(levelname)s: %(module)s."
		+ "%(funcName)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
elif (log_mode == 'run'):
	logging.basicConfig(filename='../logs/run.log', level=logging.DEBUG,
		format="%(asctime)s.%(msecs)03d: %(levelname)s: %(module)s."
		+ "%(funcName)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
elif (log_mode == 'monitor'):
	logging.basicConfig(filename='../logs/monitor.log', level=logging.DEBUG,
		format="%(asctime)s.%(msecs)03d: %(levelname)s: %(module)s."
		+ "%(funcName)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
else:
	print('\n\n\t\t- Invalid log mode: ' + str(log_mode) + '\n\n')
	sys.exit(1)
log = logging.getLogger()
if (log_mode == 'monitor'): log.addHandler(logging.StreamHandler(sys.stdout))

class MGRException(Exception):
	"""The universal exception for any problems with the MGR package
	
	It can either wrap a native exception, or be raised as a new exception.
	The __str__ function allows it to be logged after being casted to a string.
	
	Args:
		ex:			An exception to wrap in a new MGRException
		msg:		An error message for a new MGRException
		mgr_utils:	A flag indicating whether ex was caught by trackExceptions
					If set to true, the first stack frame is removed
	Attributes:
		traceback:	The stack traceback list of the error
	
	"""
	def __init__(self, ex=None, msg=None, mgr_utils=False):
		if (not(ex) and msg):
			super(Exception, self).__init__(msg)
			self.traceback = traceback.extract_stack()[:-1]
		elif (ex):
			spec = (' -- specifics: ' + str(msg) if msg else '')
			if (type(ex) is MGRException):
				super(Exception, self).__init__(ex.args[0] + spec)
				self.traceback = ex.traceback
			else:
				tb = traceback.extract_tb(sys.exc_info()[2])
				if (mgr_utils): tb = tb[1:]
				message = str(type(ex).__name__) + ': ' + str(ex) + spec
				super(Exception, self).__init__(message)
				self.traceback = tb
		else:
			message = "MGRException without arguments raised"
			super(Exception, self).__init__(message)
			self.traceback = traceback.extract_stack()
	
	def __str__(self):
		return self.args[0] + '. In: ' + str(self.traceback)

class trackExceptions(object):
	"""A decorator function that handles exception logging around a function
	
	Any function using a loop is adviced to keep track of logged errors using
	its global 'err' variable. Exception handling for the enire function is
	handled by the decorator, but exceptions in the loop ought to be handled
	manually. Don't forget to start the function by accessing the global
	namespace using 'global err'.
	
	Args:
		f:	The function being decorated
	
	"""
	def __init__(self, f):
		self.f = f
	
	def __call__(self, *args, **kwargs):
		global_ns = self.f.__globals__
		global_ns['err'] = 0
		res = None
		try:
			res = self.f(*args, **kwargs)
		except Exception as e:
			if (log_mode == 'debug'): raise e
			global_ns['err'] += 1
			log_exception(e, mgr_utils=True)
		if (global_ns['err'] and log_mode == 'run'):
			print(str(global_ns['err']) + " exception(s) caught and logged " +\
					"while in " + self.f.__module__ + '.' + self.f.__name__)
		return res

def log_exception(e, msg=None, mgr_utils=False):
	"""Logs the exception and 
	
	Args:
		e:			The exception to log
		msg:		Specifics of the exception to show after the stacktrace
		mgr_utils:	A flag indicating whether ex was caught by trackExceptions
					If set to true, the error won't be added to the err count
	
	"""
	if (log_mode == 'debug'): raise e
	global err_total
	err_total += 1
	log.error(str(MGRException(ex=e, msg=msg, mgr_utils=mgr_utils)))
