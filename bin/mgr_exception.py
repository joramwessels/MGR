#! /usr/bin/python
# filename:			mgr_exception.py
# author:			Joram Wessels
# date:				28-05-2017
# python versoin:	2.7
# dependencies:		None
# description:		The exception for any problemm with the MGR package

import traceback

class MGRException(Exception):
	
	def __init__(self, ex=None, message=None):
		if (message): super(Exception, self).__init__(message)
		elif (ex):
			tb = traceback.extract_stack()
			message = str(type(ex).__name__) + ': ' + ex.message
			super(Exception, self).__init__(message)
			self.traceback = tb
	
	def __str__(self):
		return self.message + '\n' + str(self.traceback)
