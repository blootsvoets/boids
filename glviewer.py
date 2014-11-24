from OpenGL.GL import *
from OpenGL.GLU import *
from boundingbox import BoundingBox
from math import cos, sin, radians

class GLVisualisation3D(object):
	def __init__(self,
			screen_width = 1920,
			screen_height = 1080,
			vertical_fov = 50,
			bounding_box = BoundingBox([-2, -2, -2], [2, 2, 2]),
			show_velocity_vectors = False,
			camAzimuth = 40.0,
			camDistance = 6.0,
			camRotZ = 45.0):
		self.screen_width = screen_width
		self.screen_height = screen_height
		self.screen_aspect = float(self.screen_width) / self.screen_height
		self.show_velocity_vectors = show_velocity_vectors
		self.vertical_fov = vertical_fov
		self.world = bounding_box
		
		self.camDistance = camDistance
		self.camRotZ = camRotZ
		self.camAzimuth = camAzimuth
		
		glEnable(GL_DEPTH_TEST)				
		glEnable(GL_POINT_SMOOTH)				
		glEnable(GL_LINE_SMOOTH)				
		glEnableClientState(GL_VERTEX_ARRAY)
		
	
	def setup_camera(self, cor_x, cor_y, cor_z, azimuth, rotz, distance):
	
		# NOTE: Y is up in this model

		glMatrixMode(GL_PROJECTION)
		glLoadIdentity()
		gluPerspective(self.vertical_fov, self.screen_aspect, 0.1, 1000.0)

		glMatrixMode(GL_MODELVIEW)
		glLoadIdentity()

		# Verbose, but straightforward way, of positioning the camera.
		# Start with c being (distance, 0, 0)
		# First rotate around Z, then around Y.
		# Now we have c at the given distance from the origin, with specified rotation angles.

		# degrees -> radians
		beta = radians(azimuth)
		gamma = radians(rotz)

		cx = distance
		cy = cz = 0.0

		# Rotate around Z
		t = cx
		cx = cx * cos(beta) + cy * sin(beta)
		cy = t * sin(beta) + cy * cos(beta)

		# Rotate around Y
		t = cx
		cx = cx * cos(gamma) - cz * sin(gamma)
		cz = t * sin(gamma) + cz * cos(gamma)

		gluLookAt(cx+cor_x, cy+cor_y, cz+cor_z, cor_x, cor_y, cor_z, 0.0, 1.0, 0.0)

		# Move light along with position
		#GLfloat pos[] = { cx+cor_x, cy+cor_y, cz+cor_z, 0.0f };
		#glLightfv(GL_LIGHT0, GL_POSITION, pos);
	
	def draw_boids(self, boids, big_boids, show_velocity_vectors):
		# Velocity vectors
		if show_velocity_vectors:
			glColor3f(1, 0, 0)
			glBegin(GL_LINES)
			SCALE = 3.0
			for i, p in enumerate(boids.position):
				v = boids.velocity[i]
				glVertex3f(*p)		
				glVertex3f(p[0]+v[0]*SCALE, p[1]+v[1]*SCALE, p[2]+v[2]*SCALE)		
			glEnd()
	
		# Boids themselves
	
		glPointSize(3)
		glColor3f(1, 1, 1)
	
		glVertexPointer(3, GL_FLOAT, 0, boids.position)
		glDrawArrays(GL_POINTS, 0, len(boids.position))
	
		#glBegin(GL_POINTS)
		#for p in boids.position:
		#	glVertex3f(*p)		
		#glEnd()
	
		# Big boids 
	
		glPointSize(10)
		glColor3f(0, 1, 0)
	
		glVertexPointer(3, GL_FLOAT, 0, big_boids.position)
		glDrawArrays(GL_POINTS, 0, len(big_boids.position))
	
	
	def draw_grid(self):
		glColor3f(0.6, 0.6, 0.6)
		glBegin(GL_QUADS)
		glVertex3f(self.world.min[0], self.world.min[1], self.world.min[2])
		glVertex3f(self.world.max[0], self.world.min[1], self.world.min[2])
		glVertex3f(self.world.max[0], self.world.min[1], self.world.max[2])
		glVertex3f(self.world.min[0], self.world.min[1], self.world.max[2])
		glEnd()
	
		glLineWidth(3)
		glColor3f(0.5, 0.5, 0.5)

		N = 8
		S = 1.0 * (self.world.max[0] - self.world.min[0]) / (N-1)	
		glBegin(GL_LINES)
		for i in xrange(N):
			x = self.world.min[0] + i*S
			glVertex3f(x, self.world.min[1], self.world.min[2])
			glVertex3f(x, self.world.min[1], self.world.max[2])
		glEnd()

		S = 1.0 * (self.world.max[2] - self.world.min[2]) / (N-1)	
		glBegin(GL_LINES)
		for i in xrange(N):
			z = self.world.min[2] + i*S
			glVertex3f(self.world.min[0], self.world.min[1], z)
			glVertex3f(self.world.max[0], self.world.min[1], z)
		glEnd()
		
	def draw(self, boids, big_boids):
		glViewport(0, 0, self.screen_width, self.screen_height)
	
		glClearColor(0.0, 0.0, 0.0, 0.0)
		glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)

		self.setup_camera(0.0, 0.0, 0.0, self.camAzimuth, self.camRotZ, self.camDistance)

		# Grid
		self.draw_grid()

		# Boids 

		self.draw_boids(boids, big_boids, self.show_velocity_vectors)   

		#
		# Top view (X up, Z right, looking in negative Y direction)
		#

		S = self.screen_height / 4
		M = int(0.01 * self.screen_width)

		glViewport(self.screen_width - S - M, self.screen_height - S - M, S, S)

		glMatrixMode(GL_PROJECTION)			 
		glLoadIdentity() 

		# Make view slightly larger to allow boids to go outside world range and still be visible
		s = max(self.world.size[0], self.world.size[2]) * 1.1
		glOrtho(self.world.center[2]-0.5*s, self.world.center[2]+0.5*s, self.world.center[0]-0.5*s, self.world.center[0]+0.5*s, self.world.min[1]-10, self.world.max[1]+10) 

		glMatrixMode(GL_MODELVIEW)			 
		glLoadIdentity()			
		gluLookAt(
			self.world.center[0], self.world.max[1], self.world.center[2],
			self.world.center[0], self.world.min[1], self.world.center[2],
			1, 0, 0)

		glDisable(GL_DEPTH_TEST)
		self.draw_grid()
		glEnable(GL_DEPTH_TEST)
		self.draw_boids(boids, big_boids, False)   

		#
		# Side view (Y up, X right, looking in negative Z direction)
		#

		glViewport(self.screen_width - S - M, self.screen_height - 2*(S + M), S, S)

		glMatrixMode(GL_PROJECTION)			 
		glLoadIdentity()
		c = self.world.center
		# Make view slightly larger to allow boids to go outside world range and still be visible
		s = max(self.world.size[0], self.world.size[1]) * 1.1
		glOrtho(c[0]-0.5*s, c[0]+0.5*s, c[1]-0.5*s, c[1]+0.5*s, self.world.min[2]-10, self.world.max[2]+10) 

		glMatrixMode(GL_MODELVIEW)			 
		glLoadIdentity()
		gluLookAt(
			self.world.center[0], self.world.center[1], self.world.max[1],
			self.world.center[0], self.world.center[1], self.world.min[1],
			0, 1, 0)

		glDisable(GL_DEPTH_TEST)
		self.draw_grid()	
		glEnable(GL_DEPTH_TEST)
		self.draw_boids(boids, big_boids, False)
	