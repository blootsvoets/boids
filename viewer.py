#!/usr/bin/env python2
import sys, time, numpy
from pygame.locals import *
from boids import Boids, BigBoids
from glviewer import GLPyGame3D
import multiprocessing
import re

with_shadow_model = True
# Increase to have more frames per velocity change. This slows down and smooths visualisation.
smoothness = 1
num_boids = 600
dt = 0.001

def create_boids_3D(nboids=1000, nbig=1):
	bb = BigBoids(
		num_big_boids = nbig,
		dimensions = 3,
		start_center = [-1.0,-1.0, 0.0],
		max_velocity2 = 1.0, # avoids ever-increasing velocity that causes it to escape the screen
		approach_factor = 1.0, # velocity at which it approaches the boids
		dt = dt
	)
	b = Boids(
		num_boids = nboids,
		big_boids = bb,
		dimensions = 3,
		start_center = [0.5,0.5,0.5],
		rule1_factor = 1.9, # factor for going to the common center
		rule2_threshold = 0.01, # threshold for birds being close
		rule2_factor = 5.0, # speed at which close birds should move outside the threshold
		rule3_factor = 8.0, # factor for going at the same velocity as average
		escape_threshold = 0.014, # threshold for a big bird being close
		min_velocity2 = 0.2, # avoids too much passivity
		max_velocity2 = 1.0, # avoids ever-increasing velocity that causes boids to escape the screen
		rule_direction = 1.0, # factor for going to a random direction
		bounds_factor = 1.1,
		dt = dt,
		# rule1_factor = 0.0019, # factor for going to the common center
		# rule2_threshold = 0.01, # threshold for birds being close
		# rule2_factor = 0.15, # speed at which close birds should move outside the threshold
		# rule3_factor = 0.08, # factor for going at the same velocity as average
		# escape_threshold = 0.012, # threshold for a big bird being close
		# min_velocity2 = 0.0004, # avoids too much passivity
		# max_velocity2 = 0.001, # avoids ever-increasing velocity that causes boids to escape the screen
		# rule_direction = 0.001, # factor for going to a random direction
		# bounds_factor = 0.0011,
		num_neighbors = 60,
		# escape_factor = 0.3,
		
		enforce_bounds = True,
		in_random_direction = False,
		use_global_velocity_average = False
		)

	bb.set_boids(b)
	return (b, bb)

def run_boids(boids, big_boids, boid_q, big_boid_q, is_running, escape_q = None):
	# Number of iterations after which to reset target for boids to move at.
	# Needs to be run more often in 2D than in 3D.
	new_target_iter = 45

	# current_center = boids.center
	# boids.add_escape(current_center)
	i = 0
	while is_running.value:
		if escape_q is not None:
			while not escape_q.empty():
				i = 0
				near, far = escape_q.get()
				if near is None:
					break
				boids.add_escapes_between(near, far)
			
		# apply rules that govern velocity
		t0 = time.time()
		boids.update_velocity()
		big_boids.update_velocity()
		print "computed in %0.3f s" % (time.time() - t0)

		boid_q.put(boids.copy())
		big_boid_q.put(big_boids.copy())

		boids.move(1.0)
		big_boids.move(1.0)
		
		i += 1

		if i % new_target_iter == 0:
			boids.clear_escapes()
			# boids.set_random_direction()
			# boids.remove_escape(current_center)
			# current_center = boids.center
			# boids.add_escape(current_center)
	# clear queue
	if escape_q is not None:
		while escape_q.get()[0] is not None:
			pass
	
	boid_q.put(None)
	big_boid_q.put(None)
	boid_q.close()
	big_boid_q.close()

def run_model(boid_q, big_boid_q, is_running, escape_q):
	# Set up boids model
	boids, big_boids = create_boids_3D(num_boids, 0)
	boid_q.put(boids.copy())
	big_boid_q.put(big_boids.copy())
	run_boids(boids, big_boids, boid_q, big_boid_q, is_running, escape_q)

def run_shadow_model(init_boids, init_big_boids, boid_q, big_boid_q, is_running):
	boids, big_boids = create_boids_3D(num_boids, 0)
	boids.position = init_boids.position
	# big_boids.position = init_big_boids.position
	# big_boids.approach_factor = 0.0008
	boid_q.put(boids.copy())
	big_boid_q.put(big_boids.copy())
	run_boids(boids, big_boids, boid_q, big_boid_q, is_running)

numbers = re.compile('(keypad )?([0-9])')

def process_events(glgame, is_running, boids, big_boids, shadow_boids, shadow_big_boids, escape_q):
	event = glgame.next_event()
	
	while event.type != NOEVENT:

		if event.type is QUIT:
			is_running.value = False
			
		elif event.type is KEYDOWN:
			if event.key is K_ESCAPE:
				is_running.value = False
			elif event.key is K_a:
				glgame.toggle_animate()
			elif event.key is K_s and with_shadow_model:
				glgame.toggle_shadow_boids()
				glgame.draw(boids, big_boids, shadow_boids, shadow_big_boids)
			elif event.key is K_v:
				glgame.toggle_velocity_vectors()
				glgame.draw(boids, big_boids, shadow_boids, shadow_big_boids)
			else:
				match = numbers.match(event.unicode)
				if match:
					perspective = int(match.group(2))
					# 0 becomes -1 and unsets the bird perspective
					glgame.set_bird_perspective(perspective - 1)
				
		elif event.type in [MOUSEBUTTONDOWN, MOUSEBUTTONUP, MOUSEMOTION]:
			escape = glgame.process_mouse_event(event)
			if escape is not None:
				escape_q.put(escape)
			
		event = glgame.next_event()

if __name__ == '__main__':
	# queue size gives bounds for how far the thread may be ahead
	b_q = multiprocessing.Queue(maxsize=2)
	bb_q = multiprocessing.Queue(maxsize=2)
	escape_q = multiprocessing.Queue()
	
	is_running = multiprocessing.Value('b', True)

	bds = multiprocessing.Process(target=run_model, args=(b_q, bb_q, is_running,escape_q))
	bds.start()

	boids = b_q.get()
	big_boids = bb_q.get()

	if with_shadow_model:
		shadow_b_q = multiprocessing.Queue(maxsize=2)
		shadow_bb_q = multiprocessing.Queue(maxsize=2)

		shadow_bds = multiprocessing.Process(target=run_shadow_model, args=(boids, big_boids, shadow_b_q, shadow_bb_q, is_running,))
		shadow_bds.start()

		shadow_boids = shadow_b_q.get()
		shadow_big_boids = shadow_bb_q.get()
	else:
		shadow_boids = shadow_big_boids = None

	glgame = GLPyGame3D(1000,700)

	while is_running.value:
		points = process_events(glgame, is_running, boids, big_boids, shadow_boids, shadow_big_boids, escape_q)
	
		if glgame.animate and is_running.value:
			boids = b_q.get()
			big_boids = bb_q.get()
			if with_shadow_model:
				shadow_boids = shadow_b_q.get()
				shadow_big_boids = shadow_bb_q.get()

			for _ in xrange(smoothness):
				# move with a fixed velocity
				t0 = time.time()
				boids.move(1.0/smoothness)
				big_boids.move(1.0/smoothness)
				if with_shadow_model:
					shadow_boids.move(1.0/smoothness)
					shadow_big_boids.move(1.0/smoothness)
				glgame.draw(boids, big_boids, shadow_boids, shadow_big_boids)

				fps = smoothness/(time.time()-t0)
				print "%0.3f s; %.1f fps" % (1/fps, fps)
				

	escape_q.put((None,None))
	escape_q.close()

	while True:
		b_q.get()
		if bb_q.get() is None:
			break
	
	print "got values"

	if with_shadow_model:
		# while not shadow_b_q.empty():
		while True:
			shadow_b_q.get()
			if shadow_bb_q.get() is None:
				break
		print "got shadow values"
		
	bds.join()
	print "joined bds"
	
	if with_shadow_model:
		shadow_bds.join()
		print "joined shadow bds"
	
