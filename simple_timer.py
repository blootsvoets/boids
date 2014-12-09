from time import time
import multiprocessing

class SimpleTimer:
	def __init__(self, use_process_name = False, name = None, silent = False):
		self.silent = silent
		self.t = time()
		if use_process_name:
			self.name = multiprocessing.current_process().name
		else:
			self.name = name
	
	def reset(self):
		self.t = time()
	
	def elapsed(self):
		return time() - self.t
	
	def print_time(self, msg=""):
		if not self.silent:
			t1 = time()			
			dt = t1 - self.t
			if self.name is None:
				print "[%.4f s | %7.1f fps] %s" % (dt, 1.0/dt, msg)
			else:
				print "[%.4f s | %7.1f fps] <%s> %s" % (dt, 1.0/dt, self.name, msg)
			self.t = t1