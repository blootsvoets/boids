from time import time
import os

class SimpleTimer:
	def __init__(self, silent = False):
		self.silent = silent
		self.t = time()
	
	def reset(self):
		self.t = time()
	
	def elapsed(self):
		return time() - self.t
	
	def print_time(self, msg=""):
		if not self.silent:
			t1 = time()			
			dt = t1 - self.t
			print "[%5d | %.4f s | %8.1f fps] %s" % (os.getpid(), dt, 1.0/dt, msg)
			self.t = t1