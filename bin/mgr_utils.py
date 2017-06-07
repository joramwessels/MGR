#! usr/bin/python
# filename:			mgr_utils.py
# author:			Joram Wessels
# date:				29-05-2017
# python versoin:	2.7
# dependencies:		None
# public functions:	None
# description:		Provides logging, exceptions and decorators

import sys, traceback, logging

mode = 'debug'

# Creates a logger
if (mode == 'debug'):
	logging.basicConfig(stream=sys.stdout, level=logging.DEBUG,
		format="%(asctime)s.%(msecs)03d: %(levelname)s: %(module)s."
		+ "%(funcName)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
elif (mode == 'run'):
	logging.basicConfig(filename='../logs/main.log', level=logging.DEBUG,
		format="%(asctime)s.%(msecs)03d: %(levelname)s: %(module)s."
		+ "%(funcName)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger()

class MGRException(Exception):
	"""The universal exception for any problems with the MGR package
	
	It can either wrap a native exception, or be raised as a new exception.
	The __str__ function allows it to be logged after being casted to a string.
	
	Args:
		ex:		An exception to wrap in a new MGRException
		msg:	An error message for a new MGRException
	Attributes:
		
	
	"""
	def __init__(self, ex=None, msg=None, mgr_utils=False):
		if (msg):
			super(Exception, self).__init__(msg)
			self.traceback = traceback.extract_stack()[:-1]
		elif (ex):
			if (type(ex) is MGRException):
				super(Exception, self).__init__(ex.args[0])
				self.traceback = ex.traceback
			else:
				tb = traceback.extract_tb(sys.exc_info()[2])
				if (mgr_utils): tb = tb[1:]
				message = str(type(ex).__name__) + ': ' + str(ex)
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
			if (mode == 'debug'): raise e
			global_ns['err'] += 1
			log.error(str(MGRException(ex=e, mgr_utils=True)))
		if (global_ns['err']): print(str(global_ns['err']) + \
			" exception(s) caught and logged while in " + self.f.__name__	)
		return res

def log_exception(e):
	if (mode == 'debug'): raise e
	log.error(str(MGRException(ex=e)))
	if (mode == 'debug'): sys.exit(1)