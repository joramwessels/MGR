#! usr/bin/python
# test exception file

from mgr_utils import log
from mgr_utils import MGRException
from mgr_utils import trackExceptions

def one():
	two()

@trackExceptions
def two():
	three()

def three():
	raise Exception('builtin afgevangen in utils')