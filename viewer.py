#!/usr/bin/env python
import sys, numpy
from pygame.locals import *
from boids import Boids, BigBoids
from glviewer import GLPyGame3D
import worker
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
		num_neighbors = 60,
		# escape_factor = 0.3,

		enforce_bounds = True,
		in_random_direction = False,
		use_global_velocity_average = False,
		use_process = use_process
		)

	bb.set_boids(b)
	return (b, bb)

class BoidSimulation(worker.Worker):
	def init(self):
		self.t = SimpleTimer(use_process_name=True)
		# Number of iterations after which to reset target for boids to move at.
		# Needs to be run more often in 2D than in 3D.
		self.new_target_iter = 45
		self.i = 0
		self.init_boids()
		self.worker.add_result('boids', self.boids.copy())
		self.worker.add_result('big_boids', self.big_boids.copy())
		
	def init_boids(self):
		N_BIG_BOIDS = 0
		self.boids, self.big_boids = create_boids_3D(num_boids, N_BIG_BOIDS, use_process=True)
		
	def iteration(self, input, input_nowait):
		self.t.print_time("viewer.run_boids(): top of loop")
		
		if 'escape' in input_nowait:
			escapes = input_nowait['escape']
			for near, far in escapes:
				self.boids.add_escapes_between(near, far)
			if len(escapes) > 0:
				self.i = 0

		self.boids.move(1.0)
		self.big_boids.move(1.0)
		self.t.print_time("viewer.run_boids(): moved boids")
		
		self.boids.update_velocity()
		self.big_boids.update_velocity()
		self.t.print_time("viewer.run_boids(): velocity computed")

		self.i += 1

		if self.i % self.new_target_iter == 0:
			self.boids.clear_escapes()

		self.t.print_time("viewer.run_boids(): placed boids (copies) in queue")
		return {'boids': self.boids.copy(), 'big_boids': self.big_boids.copy()}
		
	def finalize(self):
		print "finalizing boids"
		self.boids.finalize()

class ShadowBoidSimulation(BoidSimulation):
	def __init__(self, init_boids, init_big_boids):
		self.init_position = init_boids.position
	
	def init_boids(self):
		super(ShadowBoidSimulation, self).init_boids()
		self.boids.position = self.init_position
		del self.init_position

numbers = re.compile('(keypad )?([0-9])')

def process_events(glgame, worker, boids, big_boids, shadow_boids, shadow_big_boids):
	event = glgame.next_event()

	while event.type != NOEVENT:

		if event.type is QUIT:
			worker.stop_running()

		elif event.type is KEYDOWN:
			if event.key is K_ESCAPE:
				worker.stop_running()
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
			elif event.key is K_v:
				glgame.toggle_velocity_vectors()
			else:
				match = numbers.match(event.unicode)
				if match:
					perspective = int(match.group(2))
					# 0 becomes -1 and unsets the bird perspective
					glgame.set_bird_perspective(perspective - 1)

		elif event.type in [MOUSEBUTTONDOWN, MOUSEBUTTONUP, MOUSEMOTION]:
			escape = glgame.process_mouse_event(event)
			if escape is not None:
				worker.add_input_nowait('escape', escape)

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

		self.stats_left = 0.1
		self.stats_top = 0.5
		self.stats_width = 0.25
		self.stats_height = 0.2
		self.stats_separation = 0.01
		self.stats_font = ('fonts/glyphs-14-normal-8x17.png', 8, 17)

		# Plots
		self.plot_left = 0.01
		self.plot_top = 0.9
		self.plot_width = 1 / 3.0
		self.plot_height = 1 / 5.0
		self.plot_separation = 0.01
		self.plot_history_length = 500
		self.plot_font = ('fonts/glyphs-24-normal-14x29.png', 14, 29)

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
	bds = worker.WorkerServer('boids', BoidSimulation(), {'escape': 0}, {'boids': 2, 'big_boids': 2})

	boids = bds.get_result('boids')
	big_boids = bds.get_result('big_boids')

	if with_shadow_model:
		shadow_bds = worker.WorkerServer('unaltered boids', ShadowBoidSimulation(boids, big_boids), {}, {'boids': 2, 'big_boids': 2})

		shadow_boids = shadow_bds.get_result('boids')
		shadow_big_boids = shadow_bds.get_result('big_boids')
	else:
		shadow_boids = shadow_big_boids = None
		
	t = SimpleTimer(name="main")
	t.print_time('Starting 3D interface')

	glgame = GLPyGame3D(settings)

	while bds.continue_run():
		
		t.print_time('calling process_events()')
		points = process_events(glgame, bds, boids, big_boids, shadow_boids, shadow_big_boids)

		if glgame.animate and bds.continue_run():
			
			t.print_time('getting boids from queue')
			
			boids = bds.get_result('boids')
			big_boids = bds.get_result('big_boids')
			
			if with_shadow_model:
				shadow_boids = shadow_bds.get_result('boids')
				shadow_big_boids = shadow_bds.get_result('big_boids')
				
			t.print_time('drawing boids')

			t.reset()
			for _ in xrange(smoothness):
				# move with a fixed velocity

				boids.move(1.0/smoothness)
				big_boids.move(1.0/smoothness)
				if with_shadow_model:
					shadow_boids.move(1.0/smoothness)
					shadow_big_boids.move(1.0/smoothness)

				glgame.draw(glgame.animate, boids, big_boids, shadow_boids, shadow_big_boids)

				fps = smoothness/t.elapsed()
				t.print_time("%.1f fps" % (fps))

		elif not glgame.animate:
			# Make sure 3D interaction stays possible when not animating
			# Mouse events will have been processed by process_events() above
			glgame.draw(glgame.animate, boids, big_boids, shadow_boids, shadow_big_boids)
	
	print "finalizing simulation"
	bds.finalize()
	print "got values"

	if with_shadow_model:
		shadow_bds.finalize()
		print "got shadow values"

	glgame.finalize()