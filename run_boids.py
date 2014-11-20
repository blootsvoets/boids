#!/usr/bin/env python

import time
import signal
from boids import Boids, BigBoids
from visualisation import Visualisation2D, Visualisation3D
import multiprocessing
import sys

# optional argument to run_boids.py
dims = 3

# set all parameters
def create_boids(dims):
	if dims == 3:
		bb = BigBoids(
			num_big_boids = 1,
			dimensions = 3,
			start_center = [1.6,1.6, 0.5],
			max_velocity2 = 0.001, # avoids ever-increasing velocity that causes it to escape the screen
			approach_factor = 0.001 # velocity at which it approaches the boids
		)
		b = Boids(
			num_boids = 1000,
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
	else:
		bb = BigBoids(
			num_big_boids = 1,
			dimensions = 2,
			start_center = [1.6,1.6],
			max_velocity2 = 0.0009, # avoids ever-increasing velocity that causes it to escape the screen
			approach_factor = 0.001 # velocity at which it approaches the boids
		)
		b = Boids(
			num_boids = 1000,
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

def run_visualisation(dims, boid_q, big_boid_q, is_running):
	window_size_inch = (14, 14)
	# Use 2D or 3D visualisation.
	if dims == 3:
		# 3D quivers (arrows) don't look very nice in matplotlib.
		vis = Visualisation3D(window_size_inch, use_quivers=False)
	elif dims == 2:
		vis = Visualisation2D(window_size_inch, use_quivers=True)
	else:
		raise ValueError("number of dimensions " + dims + " cannot be plotted")
	
	try:
		while is_running.value:
			boids = boid_q.get()
			if boids is None:
				return

			big_boids = big_boid_q.get()		
			vis.draw(boids, big_boids)
	except Exception as ex:
		print "exception:", ex
		is_running.value = False
		big_boid_q.get()
		boid_q.get()


def run_boids(dims, boid_q, big_boid_q, is_running):
	global __is_running
	__is_running = True
	
	boids, big_boids = create_boids(dims)
	
	# Increase to have more frames per velocity change. This slows down and smooths visualisation.
	smoothness = 3
	
	t0 = time.time()
	i = 0

	# Number of iterations after which to reset target for boids to move at.
	# Needs to be run more often in 2D than in 3D.
	new_target_iter = dims * 10
	while __is_running and is_running.value:
		# apply rules that govern velocity
		boids.update_velocity()
		big_boids.update_velocity()
		
		for _ in xrange(smoothness):
			# move with a fixed velocity
			boids.move(1.0/smoothness)
			big_boids.move(1.0/smoothness)
			# copy the boids datastructure to avoid simultanious modification
			boid_q.put(boids.copy())
			big_boid_q.put(big_boids.copy())
		
		i += 1
		if i % new_target_iter == 0:
			t1 = time.time()
			print "set new position after %0.3f s" % (t1 - t0)
			t0 = t1
			boids.set_random_direction()

# Quit on SIGINT
def signal_handler(signal, frame):
	global __is_running
	__is_running = False
	
signal.signal(signal.SIGINT, signal_handler)

if __name__ == '__main__':
	if len(sys.argv) == 2:
		dims = int(sys.argv[1])
		
	# Create queues for sending boids to visualisation, and start visualisation in a separate process (thread)
	# The algorithm blocks after it is ahead 60 steps on the visualisation
	b_q = multiprocessing.Queue(maxsize=60)
	bb_q = multiprocessing.Queue(maxsize=60)
	is_running = multiprocessing.Value('b', True)

	vis = multiprocessing.Process(target=run_visualisation, args=(dims, b_q, bb_q, is_running,))
	vis.start()
	
	run_boids(dims, b_q, bb_q, is_running)
	
	# Signal the visualisation to stop (and make sure the queue doesn't block)
	b_q.put(None)
	is_running.value = False
	b_q.close()
	bb_q.close()
	vis.join()
	
	print "Done!"
	sys.exit(0)
