#!/usr/bin/env python

import numpy as np
from numpy.core.umath_tests import inner1d
from boundingbox import BoundingBox

class VectorCollection(object):
	def __init__(self, size, dimensions, max_velocity2, start_center = None):
		self.size = size
		self.dimensions = dimensions
		self.max_velocity2 = max_velocity2
		self._center = None
		self._bounding_box = None
		self._average_velocity = np.zeros(dimensions)
		if start_center is not None:
			self._init_position(start_center)
		
	def _init_position(self, start_center):
		self.velocity = np.zeros((self.size, self.dimensions))
		offset = np.array(start_center[:self.dimensions]) - 0.5
		self.position = np.random.random((self.size, self.dimensions)) + offset
	
	def copy(self):
		cpy = VectorCollection(self.size, self.dimensions, self.max_velocity2)
		cpy.position = self.position.copy()
		cpy.velocity = self.velocity.copy()
		return cpy
	
	def apply_max_velocity(self):
		velocity2 = inner1d(self.velocity, self.velocity)
		mask = (velocity2 > self.max_velocity2)
		self.velocity[mask] = (np.sqrt(self.max_velocity2 / velocity2[mask]) * self.velocity[mask].T).T
			
	def move(self, factor = None):
		if factor is None:
			self.position += self.velocity
		else:
			self.position += factor * self.velocity
		self._center = None
		self._bounding_box = None
	
	def update_velocity(self):
		self.calculate_velocity()
		self.fix_velocity()
	
	def fix_velocity(self):
		self._average_velocity = None
	
	def calculate_velocity(self):
		pass
	
	def approach_position(self, pos, factor):
		return (pos - self.position)*factor
	
	def escape_position(self, pos, threshold):
		v = np.zeros((self.size, self.dimensions))
	
		diff_matrix = self.position - pos
		dist2_vector = inner1d(diff_matrix, diff_matrix)#np.sum(diff_matrix*diff_matrix, axis=1)
	
		mask = dist2_vector < threshold
		v[mask] = (np.sqrt(self.max_velocity2 / dist2_vector[mask]) * diff_matrix[mask].T).T
		return v
	
	def c_int(self, neighborhood):
		c = 0.0
		
		norm_velocity = self.velocity / np.linalg.norm(self.velocity, axis=1)[:,None]

		for i in xrange(self.size):
			neighbors = self.neighbors(i, neighborhood)
			c += inner1d(norm_velocity[i], norm_velocity[neighbors]).sum()
		
		return c / (neighborhood * self.size)
	
	# get the first n neighbors of position i
	# @return unordered sequence of neighbors
	def neighbors(self, i, n):
		diff_matrix = self.position - self.position[i]
		dist2_vector = inner1d(diff_matrix, diff_matrix)
		neighbor = np.argpartition(dist2_vector, n + 1)[:n + 1]
		return neighbor[neighbor != i]
	
	@property
	def bounding_box(self):
		if self._bounding_box is None:
			self._bounding_box = BoundingBox(points=self.position)
		return self._bounding_box
	
	@property
	def bounding_box_diagonal(self):
		return self.bounding_box.diagonal
	
	@property
	def center(self):
		if self._center is None:
			self._center = self.position.sum(axis=0) / self.size
		return self._center
	
	@property
	def average_velocity(self):
		if self._average_velocity is None:
			self._average_velocity = self.velocity.sum(axis=0) / self.size
		return self._average_velocity
	
	@property
	def x(self):
		return self.position.T[0]
	@property
	def y(self):
		return self.position.T[1]
	@property
	def z(self):
		if self.dimensions == 2:
			raise ValueError("Cannot call 3rd dimension of 2d data")
		return self.position.T[2]

	@property
	def u(self):
		return self.velocity.T[0]
	@property
	def v(self):
		return self.velocity.T[1]
	@property
	def w(self):
		if self.dimensions == 2:
			raise ValueError("Cannot call 3rd dimension of 2d data")
		return self.velocity.T[2]

class BigBoids(VectorCollection):
	def __init__(self, num_big_boids, dimensions = 3, start_center = [-0.5,-0.5,0.5], max_velocity2 = 0.04, approach_factor=0.05):
		super(BigBoids, self).__init__(num_big_boids, dimensions, max_velocity2, start_center = start_center)
		self.approach_factor = approach_factor
	
	def set_boids(self, boids):
		self.boids = boids
		
	def calculate_velocity(self):
		self.velocity += self.approach_position(self.boids.center, self.approach_factor)
		self.apply_max_velocity()

def avoid_neighbors(positions, threshold2, factor, size, dimensions, q):
	v = np.zeros((size, dimensions))
	for i in xrange(size):
		diff_matrix = positions[i] - positions
		dist2_vector = inner1d(diff_matrix, diff_matrix)#np.sum(diff_matrix*diff_matrix, axis=1)

		mask = dist2_vector < threshold2
		mask[i] = False
		if mask.any():
			selected_diffs = (np.sqrt(threshold2 / dist2_vector[mask] - 1)*diff_matrix[mask].T).T
			v[i] = selected_diffs.sum(axis=0)*factor
	q.put(v)

class Boids(VectorCollection):
	def __init__(self, num_boids, big_boids, dimensions = 3, start_center = [1.5,1.5,0.5], rule1_factor = 0.01,
			rule2_threshold = 0.0005, rule2_factor = 1.0, rule3_factor = 0.16, bounds_factor = 0.01, escape_threshold = 0.1,
			max_velocity2 = 0.01, rule_direction = 0.002, in_random_direction = False, enforce_bounds = True, use_global_velocity_average = False):
		super(Boids, self).__init__(num_boids, dimensions, max_velocity2, start_center = start_center)
		self.big_boids = big_boids
		
		self.rule1_factor = rule1_factor
		self.rule2_threshold = rule2_threshold
		self.rule2_factor = rule2_factor
		self.rule3_factor = rule3_factor
		self.bounds_factor = bounds_factor
		self.escapes = np.array([])
		self.escape_threshold = escape_threshold
		self.rule_direction = rule_direction
		self.direction = np.array(dimensions*[0.5])
		self.direction_mask = np.array(self.size*[False])
		self.in_random_direction = in_random_direction
		self.enforce_bounds = enforce_bounds
		self.use_global_velocity_average = use_global_velocity_average

	def calculate_velocity(self):
		if self.use_global_velocity_average:
			self.velocity += self.converge_velocity(self.rule3_factor)
		else:
			for b in xrange(self.size):
				self.velocity[b] += self.converge_velocity_neighbors(b, 10, self.rule3_factor)

		self.velocity += self.approach_position(self.center, self.rule1_factor)
		if self.enforce_bounds:
			self.velocity += self.ruleBounds()
		if (self.in_random_direction):
			self.velocity += self.ruleDirection()
		
		for e in self.escapes:
			self.velocity += self.escape_position(e, self.escape_threshold)
		for bb in self.big_boids.position:
			self.velocity += self.escape_position(bb, self.escape_threshold)
		
		for b in xrange(self.size):
			self.velocity[b] += self.rule2(b)
			
		self.apply_max_velocity()

	def rule2(self, bj):
		diff_matrix = self.position[bj] - self.position
		dist2_vector = inner1d(diff_matrix, diff_matrix)#np.sum(diff_matrix*diff_matrix, axis=1)

		mask = dist2_vector < self.rule2_threshold
		mask[bj] = False
		if mask.any():
			selected_diffs = (np.sqrt(self.rule2_threshold / dist2_vector[mask] - 1)*diff_matrix[mask].T).T
			return self.rule2_factor*selected_diffs.sum(axis=0)
		else:
			return 0.0

	def converge_velocity(self, factor):
		return (self.average_velocity - self.velocity)*factor

	def converge_velocity_neighbors(self, bj, num_neighbors, factor):
		neighbors = self.neighbors(bj, num_neighbors)
		neighbor_velocity = self.velocity[neighbors].sum(axis=0) / len(neighbors)
		return (neighbor_velocity - self.velocity[bj])*factor

	def ruleBounds(self):
		v = np.zeros((self.dimensions, self.size))
		for i in xrange(self.dimensions):
			v[i][self.position.T[i] < -0.5] = self.bounds_factor
			v[i][self.position.T[i] > 1.5] = -self.bounds_factor
		
		return v.T
	
	def ruleDirection(self):
		v = np.zeros((self.size, self.dimensions))
		v[self.direction_mask]  = (self.direction - self.position[self.direction_mask])*self.rule_direction
		return v

	def add_escape(self, pos):
		self.escapes.append(pos)
	
	def remove_escape(self, pos):
		self.escapes.remove(pos)
	
	def set_random_direction(self):
		new_direction = np.random.random(self.dimensions)
		diff = new_direction - self.direction
		dist2 = np.dot(diff, diff)
		if dist2 < 1.0:
			new_direction += diff*(1.0/np.sqrt(dist2) - 1)
		print "old", self.direction, "new", new_direction
		self.direction = new_direction
		self.direction_mask = np.random.random(self.size) < 0.3
