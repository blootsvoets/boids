import sys
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from boundingbox import BoundingBox
from math import cos, sin, radians
from pygame.locals import *
import pygame
import numpy as np
from PIL import Image

class HistoricValues(object):
	
	def __init__(self, max_length):
		self.max_length = max_length
		
		self.bbox_diagonal = []
		self.num_conn_components = []
		self.pos_entropy = []
		self.vel_entropy = []
		self.posvel_entropy = []
		# Only used for boids, not for shadow boids (will be same events)
		self.events = []
		
	def append(self, bbox_diagonal, num_conn_components, pos_entropy, vel_entropy, posvel_entropy, have_event):		
		
		if len(self.bbox_diagonal) == self.max_length:
			self.bbox_diagonal.pop(0)
			self.num_conn_components.pop(0)
			self.pos_entropy.pop(0)
			self.vel_entropy.pop(0)
			self.posvel_entropy.pop(0)
			self.events.pop(0)
			
		self.bbox_diagonal.append(bbox_diagonal)
		self.num_conn_components.append(num_conn_components)
		self.pos_entropy.append(pos_entropy)
		self.vel_entropy.append(vel_entropy)
		self.posvel_entropy.append(posvel_entropy)
		self.events.append(have_event)

class GLPyGame3D(object):
	def __init__(self, settings):	
		
		self.settings = settings

		pygame.display.init()  

		pygame.display.gl_set_attribute(pygame.locals.GL_MULTISAMPLEBUFFERS, 1)
		pygame.display.gl_set_attribute(pygame.locals.GL_MULTISAMPLESAMPLES, 4)

		screen_width = settings.screen_width
		screen_height = settings.screen_height
		
		flags = OPENGL | DOUBLEBUF
		if settings.fullscreen:
			flags |= FULLSCREEN
		
		pygame.display.set_mode((screen_width, screen_height), flags)			
		pygame.display.set_caption('Boids')

		self.vis = GLVisualisation3D(screen_width = screen_width, screen_height = screen_height, background_color = settings.mainview_boids.background_color,
			plot_width_factor=settings.plot_width_factor, plot_height_factor=settings.plot_height_factor, plot_history_length=settings.plot_history_length,
			smallview_size_factor = settings.smallview_size_factor, margin_factor = settings.margin_factor)
		self.mouse_button_down = None				  # we keep track of only one button at a time
		self.mouse_down_x = self.mouse_down_y = None
		self.animate = True
		self.show_axes = False
		self.show_shadow_boids = False
		#self.old_center = np.array([0.,0.,0.])
		self.bird_perspective = -1
		self.has_event = False
		self.had_vectors = False
		
		glutInit()
	
	def toggle_animate(self):
		self.animate = not self.animate
		
	def toggle_axes(self):
		self.show_axes = not self.show_axes
	
	def toggle_velocity_vectors(self):
		self.vis.show_velocity_vectors = not self.vis.show_velocity_vectors

	def toggle_shadow_boids(self):
		self.show_shadow_boids = not self.show_shadow_boids
	
	def draw(self, boids, big_boids, shadow_boids = None, shadow_big_boids = None):
		
		# Compute and store statistics

		num_pos_bins = 50
		num_vel_bins = 50

		bbox_diag = boids.bounding_box.diagonal
		num_comp = len(boids.connected_components)
		pos_entropy = boids.position_xyz_entropy(num_pos_bins)
		vel_entropy = boids.velocity_xyz_entropy(num_vel_bins)
		posvel_entropy = boids.position_velocity_entropy(num_vel_bins=num_vel_bins,num_pos_bins=num_pos_bins)
		
		if self.animate:
			self.vis.boids_historic_values.append(bbox_diag, num_comp, pos_entropy, vel_entropy, posvel_entropy, self.has_event)
		
		if shadow_boids is not None:
			
			bbox_diag = shadow_boids.bounding_box.diagonal
			num_comp = len(shadow_boids.connected_components)
			pos_entropy = shadow_boids.position_xyz_entropy(num_pos_bins)
			vel_entropy = shadow_boids.velocity_xyz_entropy(num_vel_bins)
			posvel_entropy = shadow_boids.position_velocity_entropy(num_vel_bins=num_vel_bins,num_pos_bins=num_pos_bins)		
			
			if self.animate:
				self.vis.shadow_boids_historic_values.append(bbox_diag, num_comp, pos_entropy, vel_entropy, posvel_entropy, None)
			
		# XXX needs update 
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
			
		# Draw
		
		self.vis.draw(boids, big_boids, shadow_boids, shadow_big_boids, show_shadow_boids = self.show_shadow_boids, bird_perspective = self.bird_perspective, show_axes = self.show_axes)
			
		# Done!
		
		pygame.display.flip()
		#self.old_center = boids.center
		
		self.has_event = False
	
	def next_event(self):
		return pygame.event.poll()
	
	def print_info(self, text):
		self.vis.print_info(text)
	
	def set_bird_perspective(self, new_perspective):
		if new_perspective != -1 and self.bird_perspective == -1:
			self.had_vectors = self.vis.show_velocity_vectors
			self.vis.show_velocity_vectors = True
		
		if new_perspective == -1 and self.bird_perspective != -1:
			self.vis.show_velocity_vectors = self.had_vectors

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
				self.vis.camAzimuth = min(self.vis.camAzimuth, 89.99)
	
			elif self.mouse_button_down == MB_RIGHT:
				# Zoom
				self.vis.camDistance = self.camDistance + 5.0*(self.mouse_down_y - event.pos[1]) / self.vis.screen_height
		
		elif event.type == MOUSEBUTTONUP and self.mouse_button_down == event.button:
			if self.mouse_button_down == MB_LEFT and not self.has_motion and self.bird_perspective == -1:
				ret = self.vis.get_points3D(self.mouse_down_x, self.mouse_down_y)
				self.has_event = True
				self.show_shadow_boids = True

			self.mouse_button_down = None
			
		return ret
		
class TextDrawer:
	
	def __init__(self, pos_x, pos_y, dx=0, dy=0):
		self.pos_x = pos_x
		self.pos_y = pos_y
		self.dx = dx
		self.dy = dy
	
	def draw(self, text):

		glRasterPos(self.pos_x, self.pos_y)
		for ch in text:
			glutBitmapCharacter( GLUT_BITMAP_TIMES_ROMAN_24 , ctypes.c_int( ord(ch) ) )
		self.pos_x += self.dx
		self.pos_y -= self.dy	

class TextDrawer2:		
	
	def __init__(self, glyph_file, gw, gh):
		img = Image.open(glyph_file)
		iw, ih = img.size
		pixels = img.tostring('raw')
		
		self.image_width, self.image_height = img.size
		
		self.texid = glGenTextures(1) 
		glBindTexture(GL_TEXTURE_2D, self.texid) 
		glPixelStorei(GL_UNPACK_ALIGNMENT,1)
		glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, iw, ih, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, pixels)
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)		
		
		self.glyph_width = gw
		self.glyph_height = gh
		self.glyph_width_texspace = 1.0 * gw / self.image_width
		self.glyph_height_texspace = 1.0 * gh / self.image_height
		
	def draw(self, text, x, y):
		
		glEnable(GL_TEXTURE_2D)
		glBindTexture(GL_TEXTURE_2D, self.texid)
		
		glDisable(GL_LIGHTING)
		glEnable(GL_BLEND)
		
		glColor3f(1, 1, 1)		
		glBegin(GL_QUADS)
		
		for ch in text:
			s = ord(ch) * self.glyph_width_texspace

			# Top-left
			glTexCoord2f(s, 0)
			glVertex2f(x, y)

			glTexCoord2f(s, self.glyph_height_texspace)
			glVertex2f(x, y-self.glyph_height)

			glTexCoord2f(s+self.glyph_width_texspace, self.glyph_height_texspace)
			glVertex2f(x+self.glyph_width, y-self.glyph_height)

			glTexCoord2f(s+self.glyph_width_texspace, 0)
			glVertex2f(x+self.glyph_width, y)

			x += self.glyph_width

		y -= self.glyph_height	
			
		glEnd()
		
		glDisable(GL_TEXTURE_2D)
		glBindTexture(GL_TEXTURE_2D, 0)
		glDisable(GL_BLEND)
		
		
class Plot:
	
	def __init__(self, caption, viewport, plot_size):
		self.caption = caption
		self.viewport = viewport
		self.plot_size = plot_size
		self.linewidth = 2
		
		self.td = TextDrawer2('fonts/glyphs-32-normal-19x38.png', 19, 38)
		
	def draw(self, hv_boids, hv_shadow_boids, events, show_shadow_boids):
		
		glViewport(*self.viewport)
		
		glMatrixMode(GL_PROJECTION)
		glLoadIdentity()
		glOrtho(0.0, self.plot_size[0], 0.0, self.plot_size[1], -1.0, 1.0) 

		glMatrixMode(GL_MODELVIEW)			 
		glLoadIdentity()		
		
		glLineWidth(self.linewidth)		
		
		# Axes
		
		glColor3f(0, 0, 0)
		
		glBegin(GL_LINE_STRIP)
		
		glVertex2f(0, self.plot_size[1])
		glVertex2f(0, 0)
		glVertex2f(self.plot_size[0], 0)
		
		glEnd()
		
		# Event markers
		
		glColor3f(1, 0, 0)
		
		for i, v in enumerate(events):
			if not v:
				continue
			glBegin(GL_LINES)
			glVertex2f(i, 0)
			glVertex2f(i, self.plot_size[1])
			glEnd()
		
		# Lines
		
		glColor3f(1, 1, 1)
		
		glBegin(GL_LINE_STRIP)		
		for i, v in enumerate(hv_boids):			
			glVertex2f(i, v)		
		glEnd()
		
		if show_shadow_boids:
			
			glColor3f(0, 0, 1)
		
			glBegin(GL_LINE_STRIP)			
			for i, v in enumerate(hv_shadow_boids):			
				glVertex2f(i, v)			
			glEnd()
			
		# Caption
			
		glDisable(GL_DEPTH_TEST)
		glColor3f(0, 0, 0)
		
		#td = TextDrawer(5, 0.9*self.plot_size[1])
		#td.draw(self.caption)
		self.td.draw(self.caption, 5, 0.9*self.plot_size[1])
			
		glEnable(GL_DEPTH_TEST)
		
class GLVisualisation3D(object):
	def __init__(self,
			screen_width = 1920,
			screen_height = 1080,
			background_color = (0, 0, 0, 0),
			vertical_fov = 50,
			bounding_box = BoundingBox([-3, -3, -3], [4, 4, 4]),
			show_velocity_vectors = False,
			camAzimuth = 40.0,
			camDistance = 6.0,
			camRotZ = 45.0,
			plot_width_factor = 1/3.0,
			plot_height_factor = 1/5.0,
			plot_history_length = 500,
			smallview_size_factor = 0.25,
			margin_factor = 0.01):
				
		self.screen_width = screen_width
		self.screen_height = screen_height
		self.background_color = background_color
		self.screen_aspect = float(self.screen_width) / self.screen_height
		self.show_velocity_vectors = show_velocity_vectors
		self.vertical_fov = vertical_fov
		self.world = bounding_box
		self.text_pos_x = 0.05
		self.text_pos_y = 0.95
		self.smallview_size_factor = smallview_size_factor
		
		self.camDistance = camDistance
		self.camRotZ = camRotZ
		self.camAzimuth = camAzimuth
		
		self.boids_historic_values = HistoricValues(plot_history_length)
		self.shadow_boids_historic_values = HistoricValues(plot_history_length)	
		
		self.margin = int(margin_factor * self.screen_width)

		# Set up plots
		# Size in pixels
		W = int(self.screen_width * plot_width_factor)
		H = int(self.screen_height * plot_height_factor)
		# Bbox diagonal
		vp = (self.margin, self.screen_height - self.margin - H, W, H)
		self.bbox_diagonal_plot = Plot('Bounding-box diagonal', vp, (self.boids_historic_values.max_length, 5.0))		
		# Position entropy
		vp = (self.margin, self.screen_height - 2*(self.margin + H), W, H)
		self.pos_entropy_plot = Plot('Entropy (position)', vp, (self.boids_historic_values.max_length, 5.0))
		# Number of components
		vp = (self.margin, self.screen_height - 2*(self.margin + H) - (self.margin + H/2), W, H/2)
		self.num_components_plot = Plot('Number of components', vp, (self.boids_historic_values.max_length, 5.0))
		
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
		try:
			glGetIntegerv( GL_VIEWPORT, cameraGLViewport )
		except ValueError:
			cameraGLViewport = glGetIntegerv(GL_VIEWPORT)
	
		# read projection matrix
		glMatrixMode(GL_PROJECTION)
		glLoadIdentity()
		gluPerspective(self.vertical_fov, self.screen_aspect, 0.1, 1000.0)
		try:
			glGetDoublev(GL_PROJECTION_MATRIX, cameraGLProjection)
		except ValueError:
			cameraGLProjection = glGetDoublev(GL_PROJECTION_MATRIX)

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
		try:
			glGetDoublev(GL_MODELVIEW_MATRIX, cameraGLView)
		except ValueError:
			cameraGLView = glGetDoublev(GL_MODELVIEW_MATRIX)
		
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
		# print "Near", near, "; far", far
		
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
		
	def draw_boids(self, boids, big_boids, show_velocity_vectors, shadow_boids = None, shadow_big_boids = None, draw_shadow = False, point_size=3):
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
	
		glPointSize(point_size)
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
	
		glPointSize(3*point_size)
		glColor3f(0, 1, 0)
	
		glVertexPointer(3, GL_FLOAT, 0, big_boids.position)
		glDrawArrays(GL_POINTS, 0, len(big_boids.position))
	
	# Draw a grid over X and Z
	def draw_grid(self, linewidth=3):
		
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
		glLineWidth(linewidth)

		N = 8
		S = self.world.size[0] / (N-1)	
		glBegin(GL_LINES)
		for i in xrange(N):
			x = self.world.min[0] + i*S
			glVertex3f(x, self.world.min[1]+0.001, self.world.min[2])
			glVertex3f(x, self.world.min[1]+0.001, self.world.max[2])
		glEnd()

		S = self.world.size[2] / (N-1)	
		glBegin(GL_LINES)
		for i in xrange(N):
			z = self.world.min[2] + i*S
			glVertex3f(self.world.min[0], self.world.min[1]+0.001, z)
			glVertex3f(self.world.max[0], self.world.min[1]+0.001, z)
		glEnd()
			
	def print_text(self, text):

		glRasterPos(self.text_pos_x, self.text_pos_y)
		for ch in text:
			glutBitmapCharacter( GLUT_BITMAP_9_BY_15 , ctypes.c_int( ord(ch) ) )
		self.text_pos_y -= self.text_line_height
		
	def draw_axes(self):
		
		glBegin(GL_LINES)
		
		glColor3f(1, 0, 0)
		glVertex3f(0, 0, 0)
		glVertex3f(1, 0, 0)

		glColor3f(0, 1, 0)
		glVertex3f(0, 0, 0)
		glVertex3f(0, 1, 0)
		
		glColor3f(0, 0, 1)
		glVertex3f(0, 0, 0)
		glVertex3f(0, 0, 1)
		
		glEnd()		
		
	def draw(self, boids, big_boids, shadow_boids = None, shadow_big_boids = None, show_shadow_boids = False, bird_perspective = -1, show_axes = False):
			
		#
		# Main view
		#
		
		glViewport(0, 0, self.screen_width, self.screen_height)			
		
		col = list(self.background_color)
		col.append(0)
		glClearColor(*col)
		glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)			
		
		# Boids in 3D
		
		if bird_perspective == -1:
			self.setup_camera(0.0, 0.0, 0.0, self.camAzimuth, self.camRotZ, self.camDistance, bird_perspective)
		else:
			self.setup_bird_camera(boids.position[bird_perspective], boids.velocity[bird_perspective])

		self.draw_grid()
		
		if show_axes:
			self.draw_axes()
			
		self.draw_boids(boids, big_boids, self.show_velocity_vectors, shadow_boids, shadow_big_boids, draw_shadow = show_shadow_boids)   			
		
		# Stats
		
		S = self.screen_height / 4		

		glViewport(self.screen_width - S - self.margin, self.screen_height - 3*(S + self.margin), S, S)

		glMatrixMode(GL_PROJECTION)
		glLoadIdentity()
		glOrtho(0.0, 1.0, 0.0, 1.0, -1.0, 1.0) 

		glMatrixMode(GL_MODELVIEW)			 
		glLoadIdentity()
		
		glColor3f(0, 0, 0)		
		self.text_pos_x = 0.05
		self.text_pos_y = 0.95
		self.text_line_height = 1.0 / 11
		
		hv = self.boids_historic_values
		self.print_text("Size: %0.1f" % hv.bbox_diagonal[-1])
		self.print_text("Components: %d" % hv.num_conn_components[-1])
		self.print_text("Pos. entropy: %0.3f" % hv.pos_entropy[-1])
		self.print_text("Vel. entropy: %0.3f" % hv.vel_entropy[-1])
		self.print_text("PosVel. ent.: %0.3f" % hv.posvel_entropy[-1])
		
		# self.print_text("Velocity: %0.2f" % (boids.velocity_stddev))
		# self.print_text("%0.3f; %0.3f; %0.3f" % (boids.c_int(5),boids.c_int(10),boids.c_int(20)))
		# self.print_text("%0.3f; %0.3f" % (boids.c_int(50),boids.c_int(100)))
		if shadow_boids is not None:
						
			# self.print_text("Unmodified")
			hv = self.shadow_boids_historic_values
			self.print_text('')
			self.print_text("Size: %0.1f" % hv.bbox_diagonal[-1])
			self.print_text("Components: %d" % hv.num_conn_components[-1])
			self.print_text("Pos. entropy: %0.3f" % hv.pos_entropy[-1])
			self.print_text("Vel. entropy: %0.3f" % hv.vel_entropy[-1])
			self.print_text("PosVel. ent.: %0.3f" % hv.posvel_entropy[-1])
			
			# self.print_text("Orig. distance: %0.1f" % (shadow_boids.position_stddev))
			# self.print_text("Orig. velocity: %0.1f" % (shadow_boids.velocity_stddev))	
				
		#					
		# Plots
		#
				
		self.bbox_diagonal_plot.draw(self.boids_historic_values.bbox_diagonal, self.shadow_boids_historic_values.bbox_diagonal, self.boids_historic_values.events, show_shadow_boids)
		self.pos_entropy_plot.draw(self.boids_historic_values.pos_entropy, self.shadow_boids_historic_values.pos_entropy, self.boids_historic_values.events, show_shadow_boids)
		self.num_components_plot.draw(self.boids_historic_values.num_conn_components, self.shadow_boids_historic_values.num_conn_components, self.boids_historic_values.events, show_shadow_boids)
		
		#
		# Top view (X right, Z DOWN, looking in negative Y direction)
		#

		S = int(self.smallview_size_factor * self.screen_height)

		glViewport(self.screen_width - S - self.margin, self.screen_height - S - self.margin, S, S)

		glMatrixMode(GL_PROJECTION)			 
		glLoadIdentity() 
		c = self.world.center		
		s = max(self.world.size[0], self.world.size[2])
		# Make view slightly larger to allow boids to go outside world range and still be visible
		s *= 1.1
		glOrtho(c[0]-0.5*s, c[0]+0.5*s, c[2]+0.5*s, c[2]-0.5*s, self.world.max[1]+10, self.world.min[1]-10) 
		
		glMatrixMode(GL_MODELVIEW)			 
		glLoadIdentity()					
		
		# Outline
		glLineWidth(2)
		glColor3f(1, 1, 1)
		glBegin(GL_LINE_LOOP)
		glVertex2f(c[0]-0.5*s, c[2]-0.5*s)
		glVertex2f(c[0]+0.5*s, c[2]-0.5*s)
		glVertex2f(c[0]+0.5*s, c[2]+0.5*s)
		glVertex2f(c[0]-0.5*s, c[2]+0.5*s)
		glEnd()		
		
		glRotatef(-90, 1, 0, 0)
		
		glDisable(GL_DEPTH_TEST)
		self.draw_grid(2)
		if show_axes:
			self.draw_axes()		
		glEnable(GL_DEPTH_TEST)
		
		self.draw_boids(boids, big_boids, False, shadow_boids, draw_shadow=show_shadow_boids, point_size=1)   

		#
		# Side view (Y up, X right, looking in negative Z direction)
		#

		glViewport(self.screen_width - S - self.margin, self.screen_height - 2*(S + self.margin), S, S)

		glMatrixMode(GL_PROJECTION)			 
		glLoadIdentity()
		c = self.world.center		
		s = max(self.world.size[0], self.world.size[1])
		# Make view slightly larger to allow boids to go outside world range and still be visible
		s *= 1.1
		glOrtho(c[0]-0.5*s, c[0]+0.5*s, c[1]-0.5*s, c[1]+0.5*s, self.world.min[2]-10, self.world.max[2]+10) 

		glMatrixMode(GL_MODELVIEW)			 
		glLoadIdentity()
		
		if show_axes:
			self.draw_axes()		
		
		# Outline
		glLineWidth(2)
		glColor3f(1, 1, 1)
		glBegin(GL_LINE_LOOP)
		glVertex2f(c[0]-0.5*s, c[1]-0.5*s)
		glVertex2f(c[0]+0.5*s, c[1]-0.5*s)
		glVertex2f(c[0]+0.5*s, c[1]+0.5*s)
		glVertex2f(c[0]-0.5*s, c[1]+0.5*s)
		glEnd()
		
		self.draw_boids(boids, big_boids, False, shadow_boids, draw_shadow=show_shadow_boids, point_size=1)
	
		