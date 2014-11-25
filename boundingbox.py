import numpy as np

class BoundingBox(object):
	def __init__(self, min=None, max=None, points=None):
		if min is not None and max is not None:
			self._min = np.array(min,dtype=float)
			self._max = np.array(max,dtype=float)
		elif points is not None:
			self._min = points.min(axis=0)
			self._max = points.max(axis=0)
		else:
			raise ValueError("Need to give min and max or matrix")
			
		self._reset()
		
	def _reset(self):
		self._center = None
		self._size = None
			
	@property
	def max(self, new_max):
		self._reset()
		self.max = new_max

	@property
	def min(self, new_min):
		self._reset()
		self.min = new_min

	@property
	def min(self):
		return self._min
	@property
	def max(self):
		return self._max
		
	@property
	def center(self):
		if self._center is None:
			self._center = (self.min + self.max) / 2.0
		return self._center
	
	@property
	def size(self):
		if self._size is None:
			self._size = self.max - self.min
		return self._size
	
	@property
	def diagonal(self):
		return np.linalg.norm(self.size)

	def contains(self, pos):
		return all((pos > self.min) & (pos < self.max))
		