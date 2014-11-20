#!/usr/bin/env python

from mpl_toolkits.mplot3d import axes3d
import matplotlib
import matplotlib.pyplot as plt

class Visualisation(object):
	def __init__(self, window_size, use_quivers=False):
		matplotlib.interactive(True)
		fig = plt.figure(figsize=window_size)
		self.makeAxes(fig)
		self.ax.set_axis_off()
		self.vis = None
		self.vis_bb = None
		self.use_quivers = use_quivers
	
	def draw(self, boids, big_boids):
		if self.vis is not None:
			self.vis.remove()
		if self.vis_bb is not None:
			self.vis_bb.remove()
	
		self.drawData(boids, big_boids)
		plt.draw()
	def makeAxes(self, fig):
		pass
	def drawData(self, boids, big_boids):
		pass

class Visualisation2D(Visualisation):
	def makeAxes(self, fig):
		self.ax = fig.add_subplot(111)
		self.ax.set_xlim( -1.5, 2.5 )
		self.ax.set_ylim( -1.5, 2.5 )
		
	def drawData(self, boids, big_boids):
		if self.use_quivers:
			self.vis = self.ax.quiver(boids.x, boids.y, boids.u, boids.v, scale=4.0)
			self.vis_bb = self.ax.quiver(big_boids.x, big_boids.y, big_boids.u, big_boids.v,scale=2.0,color='r')
		else:
			self.vis = self.ax.scatter(boids.x, boids.y, c='b', marker='o')
			self.vis_bb = self.ax.scatter(big_boids.x, big_boids.y, c='r', marker='^')

class Visualisation3D(Visualisation):
	def makeAxes(self, fig):
		self.ax = fig.add_subplot(111, projection='3d')
		self.ax.set_xlim3d( -0.5, 1.5 )
		self.ax.set_ylim3d( -0.5, 1.5 )
		self.ax.set_zlim3d( -0.5, 1.5 )

	def drawData(self, boids, big_boids):
		if self.use_quivers:
			self.vis = self.ax.quiver(boids.x, boids.y, boids.z, boids.u, boids.v, boids.w,length=0.02,arrow_length_ratio=1.0)
			self.vis_bb = self.ax.quiver(big_boids.x, big_boids.y, big_boids.z, big_boids.u, big_boids.v, big_boids.w, length=0.04, arrow_length_ratio=1.0, color='r')
		else:
			self.vis = self.ax.scatter(boids.x, boids.y, boids.z, c='b', marker='o')
			self.vis_bb = self.ax.scatter(big_boids.x, big_boids.y, big_boids.z, c='r', marker='^')
