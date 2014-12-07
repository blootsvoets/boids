#!/usr/bin/env python
import sys, numpy
from pygame.locals import *
from boids import Boids, BigBoids
from glviewer import GLPyGame3D
import multiprocessing
import multiprocessing.queues
import numpy as np
import re
from simple_timer import SimpleTimer

with_shadow_model = True

# Increase to have more frames per velocity change. This slows down and smooths visualisation.
smoothness = 1
num_boids = 600
dt = 0.001

def create_boids_3D(nboids=1000, nbig=1,use_process=False):
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
		use_global_velocity_average = False,
		use_process = use_process
		)

	bb.set_boids(b)
	return (b, bb)

def run_boids(boids, big_boids, boid_q, big_boid_q, is_running, escape_q = None):
	# Number of iterations after which to reset target for boids to move at.
	# Needs to be run more often in 2D than in 3D.
	new_target_iter = 45

	t = SimpleTimer()

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
		t.reset()
		boids.update_velocity()
		big_boids.update_velocity()
		t.print_time("velocity computed")

		boid_q.put(boids.copy())
		big_boid_q.put(big_boids.copy())

		boids.move(1.0)
		big_boids.move(1.0)

		i += 1

		if i % new_target_iter == 0:
			boids.clear_escapes()

	# clear queue
	if escape_q is not None:
		while escape_q.get()[0] is not None:
			pass

	boid_q.put(None)
	big_boid_q.put(None)
	boid_q.close()
	big_boid_q.close()
	boids.finalize()

def run_model(boid_q, big_boid_q, is_running, escape_q):
	# Set up boids model
	N_BIG_BOIDS = 0
	boids, big_boids = create_boids_3D(num_boids, N_BIG_BOIDS, use_process=True)
	boid_q.put(boids.copy())
	big_boid_q.put(big_boids.copy())
	run_boids(boids, big_boids, boid_q, big_boid_q, is_running, escape_q)

def run_shadow_model(init_boids, init_big_boids, boid_q, big_boid_q, is_running):
	boids, big_boids = create_boids_3D(num_boids, 0, use_process=True)
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
			elif event.key is K_b:
				glgame.show_boids_as_birds()
			elif event.key is K_p:
				glgame.show_boids_as_points()
			elif event.key is K_x:
				glgame.toggle_axes()
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


class BoidsSettings:

	def __init__(self):
		self.point_size = 3
		self.color = (1, 1, 1)
		self.shadow_color = (0.2, 0.2, 0.5)

class Settings:

	def __init__(self):
		self.screen_width = 1000
		self.screen_height = 700
		self.fullscreen = False

		self.background_color = (0.5, 0.5, 0.5)

		self.grid_size = 10
		self.grid_line_spacing = 1

		# Fraction of screen width, OR pixels
		self.margin = 0.01

		# Main 3D view
		self.mainview_boids = BoidsSettings()

		# Top and side view
		self.smallviews_boids = BoidsSettings()
		self.smallviews_boids.point_size = 1

		self.topview_size = 0.07
		self.topview_left = 0.9
		self.topview_top = 0.01

		self.sideview_size = 0.07
		self.sideview_left = 0.9
		self.sideview_top = 0.01

		self.stats_separation = 0.01

		# Plots
		self.plot_left = 0.01
		self.plot_top = 0.9
		self.plot_width = 1 / 3.0
		self.plot_height = 1 / 5.0
		self.plot_separation = 0.01
		self.plot_history_length = 500

		self.num_boids = 600
		self.dt = 0.001
		self.smoothness = 1
		self.boid_scale_factor = 0.05

		# List of file names
		self.logos = []
		self.logo_target_height = 100
		self.logo_left = 10
		self.logo_top = 100
		self.logo_separation = 40

		self.rules_left = 0.01
		self.rules_top = 0.01
		self.rules_width = 0.25

		self.equation_left = 0.01
		self.equation_top = 0.01
		self.equation_width = 0.25

if __name__ == '__main__':

	f = open('interactions.txt', 'a')
	f.write('SESSION\n')
	f.close()

	np.random.seed(123456)

	# Default settings
	settings = Settings()

	# Possibly overridden by user script
	if len(sys.argv) > 1:
		execfile(sys.argv[1], {'settings':settings})

		if len(sys.argv) >= 4:
			settings.screen_width = int(sys.argv[2])
			settings.screen_height = int(sys.argv[3])
			settings.fullscreen = False

	num_boids = settings.num_boids
	dt = settings.dt
	smoothness = settings.smoothness

	# queue size gives bounds for how far the thread may be ahead
	b_q = multiprocessing.Queue(maxsize=2)
	bb_q = multiprocessing.Queue(maxsize=2)
	escape_q = multiprocessing.queues.SimpleQueue()

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

	glgame = GLPyGame3D(settings)

	t = SimpleTimer()

	while is_running.value:
		points = process_events(glgame, is_running, boids, big_boids, shadow_boids, shadow_big_boids, escape_q)

		if glgame.animate and is_running.value:
			boids = b_q.get()
			big_boids = bb_q.get()
			if with_shadow_model:
				shadow_boids = shadow_b_q.get()
				shadow_big_boids = shadow_bb_q.get()

			t.reset()
			for _ in xrange(smoothness):
				# move with a fixed velocity

				boids.move(1.0/smoothness)
				big_boids.move(1.0/smoothness)
				if with_shadow_model:
					shadow_boids.move(1.0/smoothness)
					shadow_big_boids.move(1.0/smoothness)

				glgame.draw(boids, big_boids, shadow_boids, shadow_big_boids)

				fps = smoothness/t.elapsed()
				t.print_time("%.1f fps" % (fps))

		elif not glgame.animate:
			# Make sure 3D interaction stays possible when not animating
			# Mouse events will have been processed by process_events() above
			glgame.draw(boids, big_boids, shadow_boids, shadow_big_boids)

	escape_q.put((None,None))

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

