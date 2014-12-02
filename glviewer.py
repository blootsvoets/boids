from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from boundingbox import BoundingBox
from math import cos, sin, radians
from pygame.locals import *
import pygame
import numpy as np

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
		self.show_shadow_boids = False
		self.old_center = np.array([0.,0.,0.])
		self.bird_perspective = -1
		self.has_event = False
	
	def toggle_animate(self):
		self.animate = not self.animate
	
	def toggle_velocity_vectors(self):
		self.vis.show_velocity_vectors = not self.vis.show_velocity_vectors

	def toggle_shadow_boids(self):
		self.show_shadow_boids = not self.show_shadow_boids
	
	def draw(self, boids, big_boids, shadow_boids = None, shadow_big_boids = None):
		num_pos_bins = 50
		num_vel_bins = 50
		self.vis.draw(boids, big_boids, shadow_boids, shadow_big_boids, show_shadow_boids = self.show_shadow_boids, bird_perspective = self.bird_perspective)
		self.vis.print_text("Size: %0.1f" % (boids.bounding_box.diagonal), 0)
		self.vis.print_text("Components: %d" % (len(boids.connected_components)), 1)
		self.vis.print_text("Pos. entropy: %0.3f" % (boids.position_xyz_entropy(num_pos_bins)), 2)
		self.vis.print_text("Vel. entropy: %0.3f" % (boids.velocity_xyz_entropy(num_vel_bins)), 3)
		self.vis.print_text("PosVel. ent.: %0.3f" % (boids.position_velocity_entropy(num_vel_bins=num_vel_bins,num_pos_bins=num_pos_bins)), 4)
		
		# self.vis.print_text("Velocity: %0.2f" % (boids.velocity_stddev), 2)
		# self.vis.print_text("%0.3f; %0.3f; %0.3f" % (boids.c_int(5),boids.c_int(10),boids.c_int(20)), 6)
		# self.vis.print_text("%0.3f; %0.3f" % (boids.c_int(50),boids.c_int(100)), 7)
		if shadow_boids is not None:
			# self.vis.print_text("Unmodified", 4)
			self.vis.print_text("Size: %0.1f" % (shadow_boids.bounding_box.diagonal), 5)
			self.vis.print_text("Components: %d" % (len(shadow_boids.connected_components)), 6)
			self.vis.print_text("Pos. entropy: %0.3f" % (shadow_boids.position_xyz_entropy(num_pos_bins)), 7)
			self.vis.print_text("Vel. entropy: %0.3f" % (shadow_boids.velocity_xyz_entropy(num_vel_bins)), 8)
			self.vis.print_text("PosVel. ent.: %0.3f" % (shadow_boids.position_velocity_entropy(num_vel_bins=num_vel_bins,num_pos_bins=num_pos_bins)), 9)
			# self.vis.print_text("Orig. distance: %0.1f" % (shadow_boids.position_stddev), 4)
			# self.vis.print_text("Orig. velocity: %0.1f" % (shadow_boids.velocity_stddev), 5)
		pygame.display.flip()
		self.old_center = boids.center
				#
		# with open("/Users/joris/Desktop/velocity.csv", "a") as f:
		# 	f.write("%0.3f,%0.3f,%0.3f,%0.3f,%0.3f,%0.3f,%i\n" % (
		# 		boids.position_xyz_entropy(num_pos_bins),
		# 		boids.velocity_xyz_entropy(num_vel_bins),
		# 		boids.position_velocity_entropy(num_vel_bins=num_vel_bins,num_pos_bins=num_pos_bins),
		# 		shadow_boids.position_xyz_entropy(num_pos_bins),
		# 		shadow_boids.velocity_xyz_entropy(num_vel_bins),
		# 		shadow_boids.position_velocity_entropy(num_vel_bins=num_vel_bins,num_pos_bins=num_pos_bins),
		# 		self.has_event)
		# 	)
		self.has_event = False
	
	def next_event(self):
		return pygame.event.poll()
	
	def print_info(self, text):
		self.vis.print_info(text)
	
	def set_bird_perspective(self, new_perspective):
		self.bird_perspective = new_perspective
	
	def process_mouse_event(self, event):
		MB_LEFT = 1
		MB_MIDDLE = 2
		MB_RIGHT = 3
		ret = None

		if event.type == MOUSEBUTTONDOWN and self.mouse_button_down is None:
			self.mouse_button_down = event.button

			self.mouse_down_x = event.pos[0]
			self.mouse_down_y = event.pos[1]

			self.camAzimuth = self.vis.camAzimuth
			self.camRotZ = self.vis.camRotZ
			self.camDistance = self.vis.camDistance
			self.has_motion = False
	
		elif event.type == MOUSEMOTION:
			self.has_motion = True
	
			if self.mouse_button_down == MB_LEFT:
				# Rotate
				self.vis.camRotZ = self.camRotZ + 0.2*(event.pos[0] - self.mouse_down_x)
				self.vis.camAzimuth = self.camAzimuth + 0.2*(event.pos[1] - self.mouse_down_y)
	
			elif self.mouse_button_down == MB_RIGHT:
				# Zoom
				self.vis.camDistance = self.camDistance + 5.0*(self.mouse_down_y - event.pos[1]) / self.vis.screen_height
		
		elif event.type == MOUSEBUTTONUP and self.mouse_button_down == event.button:
			if self.mouse_button_down == MB_LEFT and not self.has_motion and self.bird_perspective == -1:
				ret = self.vis.get_points3D(self.mouse_down_x, self.mouse_down_y)
				self.has_event = True

			self.mouse_button_down = None
			
		return ret

class GLVisualisation3D(object):
	def __init__(self,
			screen_width = 1920,
			screen_height = 1080,
			vertical_fov = 50,
			bounding_box = BoundingBox([-3, -3, -3], [4, 4, 4]),
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
	
	def get_points3D(self, x, y):
		cameraGLProjection = (GLdouble * 16)()
		cameraGLView = (GLdouble * 16)()
		cameraGLViewport = (GLint * 4)()

		glViewport(0, 0, self.screen_width, self.screen_height)
		glGetIntegerv( GL_VIEWPORT, cameraGLViewport )
	
		# read projection matrix
		glMatrixMode(GL_PROJECTION)
		glLoadIdentity()
		gluPerspective(self.vertical_fov, self.screen_aspect, 0.1, 1000.0)
		glGetDoublev(GL_PROJECTION_MATRIX, cameraGLProjection)

		# read view matrix
		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();
		beta = radians(self.camAzimuth)
		gamma = radians(self.camRotZ)

		cx = self.camDistance
		cy = cz = 0.0

		# Rotate around Z
		t = cx
		cx = cx * cos(beta) + cy * sin(beta)
		cy = t * sin(beta) + cy * cos(beta)

		# Rotate around Y
		t = cx
		cx = cx * cos(gamma) - cz * sin(gamma)
		cz = t * sin(gamma) + cz * cos(gamma)
		
		gluLookAt(cx, cy, cz, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
		glGetDoublev(GL_MODELVIEW_MATRIX, cameraGLView)
		
		winX = float(x)
		winY = float(self.screen_height - y)
		
		near = np.array(gluUnProject(winX, winY, 0.0, model=cameraGLView, proj=cameraGLProjection, view=cameraGLViewport))
		far = np.array(gluUnProject(winX, winY, 1.0, model=cameraGLView, proj=cameraGLProjection, view=cameraGLViewport))
		
		# cameraProjection = np.matrix(np.array(cameraGLProjection).reshape((4,4)))
		# cameraView = np.matrix(np.array(cameraGLView).reshape((4,4)))
		#
		# viewProjInv = np.linalg.inv(cameraProjection * cameraView)
		# print viewProjInv
		# near = np.dot(viewProjInv, np.array([x, y, 0.0, 1.0]))[:3]
		# far = np.dot(viewProjInv, np.array([x, y, 1.0, 1.0]))[:3]
		print "Near", near, "; far", far
		
		return (near, far)
	
	def setup_camera(self, cor_x, cor_y, cor_z, azimuth, rotz, distance, bird_perspective):
	
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


	def setup_bird_camera(self, pos, vel):
	
		# NOTE: Y is up in this model

		glMatrixMode(GL_PROJECTION)
		glLoadIdentity()
		gluPerspective(self.vertical_fov, self.screen_aspect, 0.1, 1000.0)

		glMatrixMode(GL_MODELVIEW)
		glLoadIdentity()

		# look 10 steps ahead
		view = pos - 2.5*vel
		camera = pos - 3*vel

		gluLookAt(camera[0], camera[1], camera[2], view[0], view[1], view[2], 0.0, 1.0, 0.0)

	def print_text(self, text, line_num):
		# glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT )

		# blending = glIsEnabled(GL_BLEND)
		S = self.screen_height / 4
		M = int(0.01 * self.screen_width)

		glViewport(self.screen_width - S - M, self.screen_height - 3*(S + M), S, S)

		glMatrixMode(GL_PROJECTION)
		glLoadIdentity()
		c = self.world.center
		# Make view slightly larger to allow boids to go outside world range and still be visible
		glOrtho(0.0, 1.0, 0.0, 1.0, -1.0, 1.0) 

		glMatrixMode(GL_MODELVIEW)			 
		glLoadIdentity()
		
		# glEnable(GL_BLEND)
		glColor3f(1,1,1)

		glRasterPos(0.05, 0.95 - line_num * 0.1)
		for ch in text:
			glutBitmapCharacter( GLUT_BITMAP_9_BY_15 , ctypes.c_int( ord(ch) ) )
		
		# glMatrixMode(GL_MODELVIEW)

		# if not blending :
			# glDisable(GL_BLEND)

	def draw_boids(self, boids, big_boids, show_velocity_vectors, shadow_boids = None, shadow_big_boids = None, draw_shadow = False):
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
		if shadow_boids is None:
			glColor3f(1, 1, 1)
		else:
			glEnableClientState(GL_COLOR_ARRAY)
			# pos_diff = np.ones(len(boids.position)) - boids.diff_position(shadow_boids)
			vel_diff = np.ones(len(boids.position)) - 20*boids.diff_velocity(shadow_boids)
			coloring = np.array([np.ones(len(boids.position)),vel_diff,vel_diff]).T
			glColorPointer(3, GL_FLOAT, 0, coloring)
		
		glVertexPointer(3, GL_FLOAT, 0, boids.position)
		glDrawArrays(GL_POINTS, 0, len(boids.position))
	
		if draw_shadow:
			glDisableClientState(GL_COLOR_ARRAY)
			glColor3f(0.2, 0.2, 0.5)
			glVertexPointer(3, GL_FLOAT, 0, shadow_boids.position)
			glDrawArrays(GL_POINTS, 0, len(shadow_boids.position))
	
		elif shadow_boids is not None:
			glDisableClientState(GL_COLOR_ARRAY)
		#glBegin(GL_POINTS)
		#for p in boids.position:
		#	glVertex3f(*p)		
		#glEnd()
	
		if len(boids.escapes) > 0:
			glPointSize(5)
			glColor3f(0, 0, 1)
	
			glVertexPointer(3, GL_FLOAT, 0, boids.escapes)
			glDrawArrays(GL_POINTS, 0, len(boids.escapes))

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
		
	def draw(self, boids, big_boids, shadow_boids = None, shadow_big_boids = None, show_shadow_boids = False, bird_perspective = -1):
		#
		# 3D view
		#
		
		glViewport(0, 0, self.screen_width, self.screen_height)
	
		glClearColor(0.0, 0.0, 0.0, 0.0)
		glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
		
		if bird_perspective == -1:
			self.setup_camera(0.0, 0.0, 0.0, self.camAzimuth, self.camRotZ, self.camDistance, bird_perspective)
		else:
			self.setup_bird_camera(boids.position[bird_perspective], boids.velocity[bird_perspective])

		self.draw_grid()
		self.draw_boids(boids, big_boids, self.show_velocity_vectors, shadow_boids, shadow_big_boids, draw_shadow = show_shadow_boids)   

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
	