from OpenGL.GL import *
from OpenGL.GLU import *
from boundingbox import BoundingBox
from math import cos, sin, radians
from pygame.locals import *
import pygame

class GLPyGame3D(object):
	def __init__(self, screen_width=1920, screen_height=1080):
		pygame.display.init()  

		pygame.display.gl_set_attribute(pygame.locals.GL_MULTISAMPLEBUFFERS, 1)
		pygame.display.gl_set_attribute(pygame.locals.GL_MULTISAMPLESAMPLES, 4)

		pygame.display.set_mode((screen_width, screen_height),OPENGL|DOUBLEBUF)

		self.vis = GLVisualisation3D(screen_width = screen_width, screen_height = screen_height)
		self.mouse_button_down = None				  # we keep track of only one button at a time
		self.mouse_down_x = self.mouse_down_y = None
		self.animate = True
	
	def toggle_animate(self):
		self.animate = not self.animate
	
	def toggle_velocity_vectors(self):
		self.vis.show_velocity_vectors = not self.vis.show_velocity_vectors
		
	def draw(self, boids, big_boids):
		self.vis.draw(boids, big_boids)
		pygame.display.flip()
	
	def next_event(self):
		return pygame.event.poll()
	
	def print_info(self, text):
		self.vis.print_info(text)
	
	def process_mouse_event(self, event):
		MB_LEFT = 1
		MB_MIDDLE = 2
		MB_RIGHT = 3

		if event.type == MOUSEBUTTONDOWN and self.mouse_button_down is None:
			self.mouse_button_down = event.button

			self.mouse_down_x = event.pos[0]
			self.mouse_down_y = event.pos[1]

			self.camAzimuth = self.vis.camAzimuth
			self.camRotZ = self.vis.camRotZ
			self.camDistance = self.vis.camDistance
	
		elif event.type == MOUSEMOTION:
	
			if self.mouse_button_down == MB_LEFT:
				# Rotate
				self.vis.camRotZ = self.camRotZ + 0.2*(event.pos[0] - self.mouse_down_x)
				self.vis.camAzimuth = self.camAzimuth + 0.2*(event.pos[1] - self.mouse_down_y)
	
			elif self.mouse_button_down == MB_RIGHT:
				# Zoom
				self.vis.camDistance = self.camDistance + 5.0*(self.mouse_down_y - event.pos[1]) / self.vis.screen_height
		
		elif event.type == MOUSEBUTTONUP and self.mouse_button_down == event.button:
			self.mouse_button_down = None

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
		
		# Initialize OpenGL
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
		#
	# def print_text(self, text):
	# 	glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT )
	#
	#     blending = False
	#     if glIsEnabled(GL_BLEND) :
	#         blending = True
	#
	#     #glEnable(GL_BLEND)
	#     glColor3f(1,1,1)
	#
	# 	S = self.screen_height / 4
	# 	M = int(0.01 * self.screen_width)
	#
	# 	glWindowPos(self.screen_width - S - M, self.screen_height - 3*(S + M))
	#     for ch in text :
	#         glutBitmapCharacter( font , ctypes.c_int( ord(ch) ) )
	#
	#     if not blending :
	#         glDisable(GL_BLEND)
	
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
	
	# Draw a grid over X and Z
	def draw_grid(self):
		# Light gray
		glColor3f(0.6, 0.6, 0.6)
		glBegin(GL_QUADS)
		glVertex3f(self.world.min[0], self.world.min[1], self.world.min[2])
		glVertex3f(self.world.max[0], self.world.min[1], self.world.min[2])
		glVertex3f(self.world.max[0], self.world.min[1], self.world.max[2])
		glVertex3f(self.world.min[0], self.world.min[1], self.world.max[2])
		glEnd()
	
		# Darker gray
		glColor3f(0.5, 0.5, 0.5)
		glLineWidth(3)

		N = 8
		S = self.world.size[0] / (N-1)	
		glBegin(GL_LINES)
		for i in xrange(N):
			x = self.world.min[0] + i*S
			glVertex3f(x, self.world.min[1], self.world.min[2])
			glVertex3f(x, self.world.min[1], self.world.max[2])
		glEnd()

		S = self.world.size[2] / (N-1)	
		glBegin(GL_LINES)
		for i in xrange(N):
			z = self.world.min[2] + i*S
			glVertex3f(self.world.min[0], self.world.min[1], z)
			glVertex3f(self.world.max[0], self.world.min[1], z)
		glEnd()
		
	def draw(self, boids, big_boids):
		#
		# 3D view
		#
		
		glViewport(0, 0, self.screen_width, self.screen_height)
	
		glClearColor(0.0, 0.0, 0.0, 0.0)
		glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)

		self.setup_camera(0.0, 0.0, 0.0, self.camAzimuth, self.camRotZ, self.camDistance)

		self.draw_grid()
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
	