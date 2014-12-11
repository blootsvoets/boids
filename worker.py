import multiprocessing as mp
from simple_timer import SimpleTimer

def do_work(*args):
	client = WorkerClient(*args)
	client.work()

class Worker(object):
	def __init__(self):
		pass
		
	def setWorkerClient(self, worker_client):
		self.worker = worker_client

	def init(self):
		pass
		
	def run(self):
		while self.worker.continue_run():
			inputs = self.worker.get_all_input()
			nowait_inputs = self.worker.get_all_nowait()

			if None in inputs.values() or None in nowait_inputs.values():
				break

			results = self.iteration(inputs, nowait_inputs)
			
			if self.worker.continue_run():
				self.worker.add_all_results(results)

	def iteration(self, inputs, nowait_inputs):
		return None
		
	def finalize(self):
		pass

def clear_queues(queues, empty_first = False):
	if empty_first:
		for name in queues.keys():
			while not queues[name].empty():
				val = queues[name].get()
				if val is None:
					del queues[name]

	for q in queues.itervalues():
		while q.get() is not None:
			continue

def close_queues(queues):
	for q in queues.itervalues():
		q.put(None)
		q.close()

class WorkerClient(object):
	def __init__(self, target, input_queues, nowait_queues, result_queues, is_running, *queues):
		self.target = target
		self.input_queues = input_queues
		self.nowait_queues = nowait_queues
		self.result_queues = result_queues
		self.is_running = is_running
		self.t = SimpleTimer(use_process_name=True)
	
	def work(self):
		self.t.reset()
		self.target.setWorkerClient(self)			
		self.target.init()
		try:
			self.target.run()
		except Exception as e:
			print e
		
		self.t.print_time("finished running")
		# empty input queue
		clear_queues(self.input_queues)
		clear_queues(self.nowait_queues)
		close_queues(self.result_queues)
		
		self.t.print_time("finalizing target")
		self.target.finalize()
		self.t.print_time("finalized target")
		
	def get_all_input(self):
		return dict((k, self.get_input(k)) for k in self.input_queues.iterkeys())
	
	def get_all_nowait(self):
		nowait_inputs = {}
		for key in self.nowait_queues.iterkeys():
			nowait_inputs[key] = []
			try:
				while True:
					value = self.get_input_nowait(key)
					if value is None:
						nowait_inputs[key] = None
						break
					nowait_inputs[key].append(value)
			except:
				continue
		
		return nowait_inputs
	
	def has_input_queue(self, queue):
		return queue in self.input_queues

	def has_input(self, queue):
		return not self.input_queues[queue].empty()
	
	def add_result(self, queue, value):
		self.result_queues[queue].put(value)
	
	def add_all_results(self, results):
		for queue, result in results.iteritems():
			self.add_result(queue, result)
	
	def get_input(self, queue):
		value = self.input_queues[queue].get()
		# don't finalize final queue
		if value is None:
			del self.input_queues[queue]
		return value
	
	def get_input_nowait(self, queue):
		value = self.nowait_queues[queue].get_nowait()
		# don't finalize final queue
		if value is None:
			del self.nowait_queues[queue]
		return value
	
	def continue_run(self):
		return self.is_running.value

class WorkerProcess(object):
	def __init__(self, pname, target, input_queue_args, result_queue_args):
		self.t = SimpleTimer(name='manager_' + pname)
		input_queues = {}
		nowait_queues = {}
		result_queues = {}
	
		for name, size in input_queue_args.iteritems():
			if size is None:
				input_queues[name] = mp.Queue()
			elif size is 0:
				nowait_queues[name] = mp.Queue()
			else:
				input_queues[name] = mp.Queue(maxsize=size)

		for name, size in result_queue_args.iteritems():
			if size is None:
				result_queues[name] = mp.Queue()
			else:
				result_queues[name] = mp.Queue(maxsize=size)
	
		is_running = mp.Value('b', True)
		
		args = (target, input_queues, nowait_queues, result_queues, is_running)
		print args
		
		self.process = mp.Process(name=pname,target=do_work,args=args)
		self.process.start()
		
		self.client = WorkerClient(*args)		

	def has_result_queue(self, queue):
		return queue in self.client.result_queues
	
	def get_result(self, queue):
		value = self.client.result_queues[queue].get()
		# don't finalize final queue
		if value is None:
			del self.client.result_queues[queue]
		return value
	
	def add_input(self, queue, value):
		self.client.input_queues[queue].put(value)
	
	def add_input_nowait(self, queue, value):
		self.client.nowait_queues[queue].put(value)
	
	def finalize(self):
		self.t.reset()
		self.stop_running()
		# send final call
		self.t.print_time("final iter value")
		close_queues(self.client.input_queues)
		self.t.print_time("final nowait value")
		close_queues(self.client.nowait_queues)
		# empty result queue
		clear_queues(self.client.result_queues, empty_first = True)

		self.process.join()
	
	def continue_run(self):
		return self.client.is_running.value
	
	def stop_running(self):
		self.client.is_running.value = False
