#! /usr/bin/python
# filename:			mgr_exception.py
# author:			Joram Wessels
# date:				28-05-2017
# python versoin:	2.7
# dependencies:		None
# description:		The exception for any problemm with the MGR package

import Exception, traceback

class MGRException(Exception):
	
	def __init__(self, e):
		ex_type, ex, tb = sys.exc_info()
		message = str(ex_type) + ': ' + ex.message
		super(Exception, self).__init__(message)
		self.traceback = traceback.format_exc(tb)
	
	def __str__(self):
		return self.message + ' \n ' + self.traceback
