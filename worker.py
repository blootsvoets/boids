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
			inputs = {}
			nowait_inputs = {}

			for key in self.worker.input_queues.keys():
				inputs[key] = self.worker.get_input(key)
					
			for key in self.worker.nowait_queues.keys():
				nowait_inputs[key] = []
				try:
					value = self.worker.get_input_nowait(key)
					if value is None:
						nowait_inputs[key] = None
						break
					nowait_inputs[key].append(value)
				except:
					continue

			if None in inputs.values() or None in nowait_inputs.values():
				break

			results = self.iteration(inputs, nowait_inputs)
			
			if self.worker.continue_run():
				for queue, result in results.items():
					self.worker.add_result(queue, result)

	def iteration(self, inputs, nowait_inputs):
		return None
		
	def finalize(self):
		pass

class WorkerClient(object):
	def __init__(self, target, input_queues, nowait_queues, result_queues, is_running, *queues):
		self.target = target
		self.input_queues = self.add_queues(input_queues, queues)
		self.nowait_queues = self.add_queues(nowait_queues, queues)
		self.result_queues = self.add_queues(result_queues, queues)
		self.is_running = is_running
		self.t = SimpleTimer(use_process_name=True)
	
	def add_queues(self, idxs, queues):
		qs = {}
		for name, idx in idxs.iteritems():
			qs[name] = queues[idx]
		return qs
	
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
		for q in self.input_queues.itervalues():
			while q.get() is not None:
				continue

		# empty input queue
		for q in self.nowait_queues.itervalues():
			while q.get() is not None:
				continue
	
		# send final call
		for q in self.result_queues.itervalues():
			q.put(None)
			q.close()
		
		self.t.print_time("finalizing target")
		self.target.finalize()
		self.t.print_time("finalized target")
		
	def has_input_queue(self, queue):
		return queue in self.input_queues

	def has_input(self, queue):
		return not self.input_queues[queue].empty()
	
	def add_result(self, queue, value):
		self.result_queues[queue].put(value)
	
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

class WorkerServer(object):
	def __init__(self, pname, target, input_queue_args, result_queue_args):
		self.t = SimpleTimer(name='manager_' + pname)
		input_queues = {}
		nowait_queues = {}
		result_queues = {}
		queues = []
	
		for name, size in input_queue_args.iteritems():
			if size is None:
				input_queues[name] = len(queues)
				queues.append(mp.Queue())
			elif size is 0:
				nowait_queues[name] = len(queues)
				queues.append(mp.Queue())
			else:
				input_queues[name] = len(queues)
				queues.append(mp.Queue(maxsize=size))

		for name, size in result_queue_args.iteritems():
			result_queues[name] = len(queues)
			if size is None:
				queues.append(mp.Queue())
			else:
				queues.append(mp.Queue(maxsize=size))
	
		is_running = mp.Value('b', True)
		
		args = (target, input_queues, nowait_queues, result_queues, is_running) + tuple(queues)
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
		for q in self.client.input_queues.itervalues():
			q.put(None)
			q.close()
		# send final call
		self.t.print_time("final nowait value")
		for q in self.client.nowait_queues.itervalues():
			q.put(None)
			q.close()
		# empty result queue
		queues = dict(self.client.result_queues)

		for name in queues.keys():
			while not queues[name].empty():
				val = queues[name].get()
				if val is None:
					del queues[name]

		for q in queues.itervalues():
			while q.get() is not None:
				continue

		self.process.join()
	
	def continue_run(self):
		return self.client.is_running.value
	
	def stop_running(self):
		self.client.is_running.value = False
