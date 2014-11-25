#!/usr/bin/env python
import sys, time, numpy
from pygame.locals import *
from boids import Boids, BigBoids
from glviewer import GLPyGame3D
import multiprocessing
	
def create_boids_3D(nboids=1000, nbig=1):
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
		rule1_factor = 0.0002, # factor for going to the common center
		rule2_threshold = 0.008, # threshold for birds being close
		rule2_factor = 0.15, # speed at which close birds should move outside the threshold
		rule3_factor = 0.15, # factor for going at the same velocity as average
		escape_threshold = 0.03, # threshold for a big bird being close
		min_velocity2 = 0.0004, # avoids too much passivity
		max_velocity2 = 0.001, # avoids ever-increasing velocity that causes boids to escape the screen
		rule_direction = 0.001, # factor for going to a random direction
		bounds_factor = 0.001,
		num_neighbors = 7,
		
		enforce_bounds = True,
		in_random_direction = True,
		use_global_velocity_average = False
		)

	bb.set_boids(b)
	return (b, bb)


def run_model(boid_q, big_boid_q, is_running):
	# Number of iterations after which to reset target for boids to move at.
	# Needs to be run more often in 2D than in 3D.
	new_target_iter = 15

	# Set up boids model
	boids, big_boids = create_boids_3D(1000, 1)	

	# current_center = boids.center
	# boids.add_escape(current_center)
	i = 0
	while is_running.value:
		# apply rules that govern velocity
		boids.update_velocity()
		big_boids.update_velocity()
	
		boid_q.put(boids.copy())
		big_boid_q.put(big_boids.copy())

		boids.move(1.0)
		big_boids.move(1.0)
		
		i += 1

		if i % new_target_iter == 0:
			boids.set_random_direction()
			# boids.remove_escape(current_center)
			# current_center = boids.center
			# boids.add_escape(current_center)
			
	boid_q.close()
	big_boid_q.close()
	
def process_events(glgame, boids, big_boids):
	event = glgame.next_event()
	
	while event.type != NOEVENT:

		if event.type is QUIT:
			is_running.value = False
			
		elif event.type is KEYDOWN:
			if event.key is K_ESCAPE:
				is_running.value = False
			elif event.key is K_a:
				glgame.toggle_animate()
			elif event.key is K_v:
				glgame.toggle_velocity_vectors()
				glgame.draw(boids, big_boids)
				
		elif event.type in [MOUSEBUTTONDOWN, MOUSEBUTTONUP, MOUSEMOTION]:
			glgame.process_mouse_event(event)
			
		event = glgame.next_event()

if __name__ == '__main__':
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
	num_iter_per_print = 100

	# Increase to have more frames per velocity change. This slows down and smooths visualisation.
	smoothness = 2

	glgame = GLPyGame3D(1000,700)

	while is_running.value:
		process_events(glgame, boids, big_boids)
	
		if glgame.animate and is_running.value:
			boids = b_q.get()
			big_boids = bb_q.get()

			for _ in xrange(smoothness):
				# move with a fixed velocity
				boids.move(1.0/smoothness)
				big_boids.move(1.0/smoothness)
				glgame.draw(boids, big_boids)

			# draw(boids, big_boids)
		
			print "bounding box diagonal: %0.3f; C_int = %0.3f" % (boids.bounding_box.diagonal, boids.c_int(10))

			i += 1
			if i % num_iter_per_print == 0:
				t1 = time.time()
				print "%.1f fps" % (smoothness*num_iter_per_print/(t1-t0))
				t0 = t1		

	while not b_q.empty():
		b_q.get()
		bb_q.get()

	bds.join()
	