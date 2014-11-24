#!/usr/bin/env python
import sys, time, numpy
import pygame
from pygame.locals import *
from boids import Boids, BigBoids
from glviewer import GLVisualisation3D
import multiprocessing

DIMS = 3

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
		start_center = [-1.0,-1.0, 0.0],
		max_velocity2 = 0.001, # avoids ever-increasing velocity that causes it to escape the screen
		approach_factor = 0.001 # velocity at which it approaches the boids
	)
	b = Boids(
		num_boids = nboids,
		big_boids = bb,
		dimensions = 3,
		start_center = [-0.5,-0.5,0.5],
		rule1_factor = 0.003, # factor for going to the common center
		rule2_threshold = 0.005, # threshold for birds being close
		rule2_factor = 0.1, # speed at which close birds should move outside the threshold
		rule3_factor = 0.2, # factor for going at the same velocity as average
		escape_threshold = 0.03, # threshold for a big bird being close
		max_velocity2 = 0.001, # avoids ever-increasing velocity that causes boids to escape the screen
		rule_direction = 0.001, # factor for going to a random direction
		bounds_factor = 0.005,
		
		enforce_bounds = True,
		in_random_direction = True,
		use_global_velocity_average = False
		)

	bb.set_boids(b)
	return (b, bb)

screen = (1920,1080)

pygame.display.init()  

pygame.display.gl_set_attribute(pygame.locals.GL_MULTISAMPLEBUFFERS, 1)
pygame.display.gl_set_attribute(pygame.locals.GL_MULTISAMPLESAMPLES, 4)

pygame.display.set_mode(screen,OPENGL|DOUBLEBUF)

vis = GLVisualisation3D(screen_width = screen[0], screen_height = screen[1])

def draw(boids, big_boids):
	vis.draw(boids, big_boids)
	pygame.display.flip()		   

def process_mouse_event(event):
	MB_LEFT = 1
	MB_MIDDLE = 2
	MB_RIGHT = 3
	
	global mouse_button_down, mouse_down_x, mouse_down_y
	global saved_camAzimuth, saved_camRotZ, saved_camDistance
	
	if event.type == MOUSEBUTTONDOWN and mouse_button_down == -1:
		
		mouse_button_down = event.button

		mouse_down_x = event.pos[0]
		mouse_down_y = event.pos[1]

		saved_camAzimuth = vis.camAzimuth
		saved_camRotZ = vis.camRotZ
		saved_camDistance = vis.camDistance
		
	elif event.type == MOUSEMOTION:
		
		if mouse_button_down == MB_LEFT:
			# Rotate
			vis.camRotZ = saved_camRotZ + 0.2*(event.pos[0] - mouse_down_x)
			vis.camAzimuth = saved_camAzimuth + 0.2*(event.pos[1] - mouse_down_y)
		
		elif mouse_button_down == MB_RIGHT:
			# Zoom
			vis.camDistance = saved_camDistance + 5.0*(mouse_down_y - event.pos[1]) / vis.screen_height
			
	elif event.type == MOUSEBUTTONUP and mouse_button_down == event.button:
		mouse_button_down = -1
	
# Initialize OpenGL	

saved_camRotZ = saved_camAzimuth = saved_camDistance = None

mouse_button_down = -1				  # we keep track of only one button at a time
mouse_down_x = mouse_down_y = None

"""
# 2D projection
glMatrixMode(GL_PROJECTION)			 
glLoadIdentity()						
glOrtho(-2*screen_aspect, 2*screen_aspect, -2, 2, -2, 2)			

glMatrixMode(GL_MODELVIEW)			 
glLoadIdentity()						
"""

# Increase to have more frames per velocity change. This slows down and smooths visualisation.
smoothness = 2

# Number of iterations after which to reset target for boids to move at.
# Needs to be run more often in 2D than in 3D.
new_target_iter = DIMS * 15

def run_model(boid_q, big_boid_q, is_running):
	# Set up boids model
	if DIMS == 2:
		boids, big_boids = create_boids_2()	
		assert False and "Not viewable yet"
	else:
		boids, big_boids = create_boids_3(1000, 1)	

	i = 0
	while is_running.value:
		# apply rules that govern velocity
		t3 = time.time()
		boids.update_velocity()
		t4 = time.time()
		big_boids.update_velocity()
		t5 = time.time()
	
		boid_q.put(boids.copy())
		big_boid_q.put(big_boids.copy())

		boids.move(1.0)
		big_boids.move(1.0)
		
		i += 1

		if i % new_target_iter == 0:
			boids.set_random_direction()
			
	boid_q.close()
	big_boid_q.close()

# queue size gives bounds for how far the thread may be ahead
b_q = multiprocessing.Queue(maxsize=2)
bb_q = multiprocessing.Queue(maxsize=2)
is_running = multiprocessing.Value('b', True)

bds = multiprocessing.Process(target=run_model, args=(b_q, bb_q, is_running,))
bds.start()

boids = b_q.get()
big_boids = bb_q.get()

t0 = time.time()
i = 0
num_iter = 100
animate = True

while is_running.value:
	event = pygame.event.poll()
	
	while event.type != NOEVENT:

		if event.type is QUIT:
			is_running.value = False
			
		elif event.type is KEYDOWN:
			if event.key is K_ESCAPE:
				is_running.value = False
			elif event.key is K_a:
				animate = not animate
			elif event.key is K_v:
				vis.show_velocity_vectors = not vis.show_velocity_vectors
				draw(boids, big_boids)
				
		elif event.type in [MOUSEBUTTONDOWN, MOUSEBUTTONUP, MOUSEMOTION]:
			process_mouse_event(event)
			
		event = pygame.event.poll()
	
	if animate and is_running.value:
		boids = b_q.get()
		big_boids = bb_q.get()

		for _ in xrange(smoothness):
			# move with a fixed velocity
			boids.move(1.0/smoothness)
			big_boids.move(1.0/smoothness)
			draw(boids, big_boids)

		# draw(boids, big_boids)
		
		print "bounding box diagonal: %0.3f; C_int = %0.3f" % (boids.bounding_box_diagonal, boids.c_int(10))

		i += 1
		if i % num_iter == 0:
			t1 = time.time()
			print "%.1f fps" % (smoothness*num_iter/(t1-t0))
			t0 = t1		

while not b_q.empty():
	b_q.get()
	bb_q.get()

bds.join()
