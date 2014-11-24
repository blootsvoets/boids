#!/usr/bin/env python
import sys, time, numpy
import pygame
from math import cos, sin, radians
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from boids import Boids, BigBoids

screen_width = 1920
screen_height = 1080
screen_aspect = 1.0 * screen_width / screen_height
VERTICAL_FOV = 50

DIMS = 3

world_min = numpy.array([-2, -2, -2])
world_max = numpy.array([2, 2, 2])
world_center = 0.5 * (world_min + world_max)
world_size = world_max - world_min

animate = True
show_velocity_vectors = False

def create_boids_2(nboids=1000, nbig=1):
	
	bb = BigBoids(
		num_big_boids = nbig,
		dimensions = 2,
		start_center = [1.6,1.6],
		max_velocity2 = 0.0009, # avoids ever-increasing velocity that causes it to escape the screen
		approach_factor = 0.001 # velocity at which it approaches the boids
	)
	
	b = Boids(
		num_boids = nboids,
		big_boids = bb,
		dimensions = 2,
		start_center = [0.1,0.1],
		rule1_factor = 0.0003, # factor for going to the common center
		rule2_threshold = 0.001, # threshold for birds being close
		rule2_factor = 0.08, # speed at which close birds should move outside the threshold
		rule3_factor = 0.06, # factor for going at the same velocity as average
		escape_threshold = 0.01, # threshold for a big bird being close
		max_velocity2 = 0.0008, # avoids ever-increasing velocity that causes boids to escape the screen
		rule_direction = 0.003 # factor for going to a random direction
		)
		
	bb.set_boids(b)
	return (b, bb)
		
	
def create_boids_3(nboids=1000, nbig=1):   
	
	bb = BigBoids(
		num_big_boids = nbig,
		dimensions = 3,
		start_center = [1.6,1.6, 0.5],
		max_velocity2 = 0.001, # avoids ever-increasing velocity that causes it to escape the screen
		approach_factor = 0.001 # velocity at which it approaches the boids
	)
	b = Boids(
		num_boids = nboids,
		big_boids = bb,
		dimensions = 3,
		start_center = [0.1,0.1,0.5],
		rule1_factor = 0.0003, # factor for going to the common center
		rule2_threshold = 0.005, # threshold for birds being close
		rule2_factor = 0.08, # speed at which close birds should move outside the threshold
		rule3_factor = 0.04, # factor for going at the same velocity as average
		escape_threshold = 0.03, # threshold for a big bird being close
		max_velocity2 = 0.001, # avoids ever-increasing velocity that causes boids to escape the screen
		rule_direction = 0.003 # factor for going to a random direction
		)

	bb.set_boids(b)
	return (b, bb)
	
def setup_camera(cor_x, cor_y, cor_z, azimuth, rotz, distance):
	
	# NOTE: Y is up in this model

	glMatrixMode(GL_PROJECTION)
	glLoadIdentity()
	gluPerspective(VERTICAL_FOV, screen_aspect, 0.1, 1000.0)

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
	
def draw_boids(show_velocity_vectors):
	
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
	
	
def draw_grid():
	
	glColor3f(0.6, 0.6, 0.6)
	glBegin(GL_QUADS)
	glVertex3f(world_min[0], world_min[1], world_min[2])
	glVertex3f(world_max[0], world_min[1], world_min[2])
	glVertex3f(world_max[0], world_min[1], world_max[2])
	glVertex3f(world_min[0], world_min[1], world_max[2])
	glEnd()
	
	glLineWidth(3)
	glColor3f(0.5, 0.5, 0.5)

	N = 8
	S = 1.0 * (world_max[0] - world_min[0]) / (N-1)	
	glBegin(GL_LINES)
	for i in xrange(N):
		x = world_min[0] + i*S
		glVertex3f(x, world_min[1], world_min[2])
		glVertex3f(x, world_min[1], world_max[2])
	glEnd()

	S = 1.0 * (world_max[2] - world_min[2]) / (N-1)	
	glBegin(GL_LINES)
	for i in xrange(N):
		z = world_min[2] + i*S
		glVertex3f(world_min[0], world_min[1], z)
		glVertex3f(world_max[0], world_min[1], z)
	glEnd()
	

def draw():
	
	assert DIMS == 3
	
	glViewport(0, 0, screen_width, screen_height)
	
	glClearColor(0.0, 0.0, 0.0, 0.0)
	glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
	
	setup_camera(0.0, 0.0, 0.0, camAzimuth, camRotZ, camDistance)
	
	# Grid
	draw_grid()

	"""
	glBegin(GL_LINES)
	glVertex3f(world_min[0], world_min[1], world_min[2])
	glVertex3f(world_max[0], world_min[1], world_min[2])
	glVertex3f(world_min[0], world_min[1], world_min[2])
	glVertex3f(world_min[0], world_max[1], world_min[2])
	glVertex3f(world_min[0], world_min[1], world_min[2])
	glVertex3f(world_min[0], world_min[1], world_max[2])
	
	glVertex3f(world_max[0], world_max[1], world_max[2])
	glVertex3f(world_min[0], world_max[1], world_max[2])
	glVertex3f(world_max[0], world_max[1], world_max[2])
	glVertex3f(world_max[0], world_min[1], world_max[2])
	glVertex3f(world_max[0], world_max[1], world_max[2])
	glVertex3f(world_max[0], world_max[1], world_min[2])
	glEnd()
	"""
	
	# Boids 
	
	draw_boids(show_velocity_vectors)   

	#
	# Top view (X up, Z right, looking in negative Y direction)
	#
	
	S = screen_height / 4
	M = int(0.01 * screen_width)
	
	glViewport(screen_width - S - M, screen_height - S - M, S, S)
	
	glMatrixMode(GL_PROJECTION)			 
	glLoadIdentity() 
	
	# Make view slightly larger to allow boids to go outside world range and still be visible
	s = max(world_size[0], world_size[2]) * 1.1
	glOrtho(world_center[2]-0.5*s, world_center[2]+0.5*s, world_center[0]-0.5*s, world_center[0]+0.5*s, world_min[1]-10, world_max[1]+10) 
	
	glMatrixMode(GL_MODELVIEW)			 
	glLoadIdentity()			
	gluLookAt(
		world_center[0], world_max[1], world_center[2],
		world_center[0], world_min[1], world_center[2],
		1, 0, 0)
	
	glDisable(GL_DEPTH_TEST)
	draw_grid()
	glEnable(GL_DEPTH_TEST)
	draw_boids(False)   
	
	#
	# Side view (Y up, X right, looking in negative Z direction)
	#

	glViewport(screen_width - S - M, screen_height - 2*(S + M), S, S)
	
	glMatrixMode(GL_PROJECTION)			 
	glLoadIdentity() 
	c = world_center
	# Make view slightly larger to allow boids to go outside world range and still be visible
	s = max(world_size[0], world_size[1]) * 1.1
	glOrtho(c[0]-0.5*s, c[0]+0.5*s, c[1]-0.5*s, c[1]+0.5*s, world_min[2]-10, world_max[2]+10) 
	
	glMatrixMode(GL_MODELVIEW)			 
	glLoadIdentity()			
	gluLookAt(
		world_center[0], world_center[1], world_max[1],
		world_center[0], world_center[1], world_min[1],
		0, 1, 0)
	
	glDisable(GL_DEPTH_TEST)
	draw_grid()	
	glEnable(GL_DEPTH_TEST)
	draw_boids(False)   
	

	pygame.display.flip()		   

def process_mouse_event(event):
	
	MB_LEFT = 1
	MB_MIDDLE = 2
	MB_RIGHT = 3
	
	global mouse_button_down, mouse_down_x, mouse_down_y
	global camAzimuth, camRotZ, camDistance
	global saved_camAzimuth, saved_camRotZ, saved_camDistance
	
	if event.type == MOUSEBUTTONDOWN and mouse_button_down == -1:
		
		mouse_button_down = event.button

		mouse_down_x = event.pos[0]
		mouse_down_y = event.pos[1]

		saved_camAzimuth = camAzimuth
		saved_camRotZ = camRotZ
		saved_camDistance = camDistance
		
	elif event.type == MOUSEMOTION:
		
		if mouse_button_down == MB_LEFT:
			# Rotate
			camRotZ = saved_camRotZ + 0.2*(event.pos[0] - mouse_down_x)
			camAzimuth = saved_camAzimuth + 0.2*(event.pos[1] - mouse_down_y)
		
		elif mouse_button_down == MB_RIGHT:
			# Zoom
			camDistance = saved_camDistance + 5.0*(mouse_down_y - event.pos[1]) / screen_height
			
	elif event.type == MOUSEBUTTONUP and mouse_button_down == event.button:
		mouse_button_down = -1
	
# Initialize OpenGL	

camDistance = 6.0
camRotZ = 45.0
camAzimuth = 40.0
saved_camRotZ = saved_camAzimuth = saved_camDistance = None

mouse_button_down = -1				  # we keep track of only one button at a time
mouse_down_x = mouse_down_y = None

pygame.display.init()  

pygame.display.gl_set_attribute(pygame.locals.GL_MULTISAMPLEBUFFERS, 1)
pygame.display.gl_set_attribute(pygame.locals.GL_MULTISAMPLESAMPLES, 4)

pygame.display.set_mode((screen_width,screen_height),OPENGL|DOUBLEBUF)

glEnable(GL_DEPTH_TEST)				
glEnable(GL_POINT_SMOOTH)				
glEnable(GL_LINE_SMOOTH)				
glEnableClientState(GL_VERTEX_ARRAY)

"""
# 2D projection
glMatrixMode(GL_PROJECTION)			 
glLoadIdentity()						
glOrtho(-2*screen_aspect, 2*screen_aspect, -2, 2, -2, 2)			

glMatrixMode(GL_MODELVIEW)			 
glLoadIdentity()						
"""

# Set up boids model

if DIMS == 2:
	boids, big_boids = create_boids_2()	
	assert False and "Not viewable yet"
else:
	boids, big_boids = create_boids_3(500, 1)	

# Increase to have more frames per velocity change. This slows down and smooths visualisation.
smoothness = 2

t0 = time.time()
i = 0

# Number of iterations after which to reset target for boids to move at.
# Needs to be run more often in 2D than in 3D.
new_target_iter = DIMS * 10

while True:
	
	event = pygame.event.poll()
	
	while event.type != NOEVENT:

		if event.type is QUIT:
			sys.exit(0)
			
		elif event.type is KEYDOWN:
			if event.key is K_ESCAPE:
				sys.exit(0)
			elif event.key is K_a:
				animate = not animate
			elif event.key is K_v:
				show_velocity_vectors = not show_velocity_vectors
				
		elif event.type in [MOUSEBUTTONDOWN, MOUSEBUTTONUP, MOUSEMOTION]:
			process_mouse_event(event)
			
		event = pygame.event.poll()
		
		draw()
		
	if animate:
	
		# apply rules that govern velocity
		t3 = time.time()
		boids.update_velocity()
		t4 = time.time()
		big_boids.update_velocity()
		t5 = time.time()
		#print 'update_velocity(): boids = %.1f ms, big_boids = %.1f ms' % ((t4-t3)*1000, (t5-t4)*1000)
		
		for _ in xrange(smoothness):
			# move with a fixed velocity
			boids.move(1.0/smoothness)
			big_boids.move(1.0/smoothness)
			
			draw()
							  
		i += 1
	
		if i % new_target_iter == 0:
			t1 = time.time()
			print "set new position after %0.3f s (%.1f fps)" % (t1 - t0, new_target_iter/(t1-t0))
			t0 = t1
			boids.set_random_direction()
		
