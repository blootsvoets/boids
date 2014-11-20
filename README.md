# Boids

A simple model of a flock of birds, attacked by a predating bird.

## Usage

Install matplotlib. For quivers in the 3D representation, version of matplotlib must be > 0.4.

To run boids, execute run_boids.py. It takes one optional argument, being the number of dimensions used, either 2 or 3.

## Model

The boids movement at some point converges to single point. An example dynamics rule that is now implemented is to move a random 1/3rd of the boids to a random other location.

Classic rules that are enforced on the boids:

1. converge to the center of mass of the flock
2. if your neighbour is too close, move away from it
3. converge to the average velocity of the flock

This converges a bit too fast (or keeps moving offscreen) so I added a few dynamics.

Additional rules:

- every X iterations, a random 1/3rd of the boids move to a random direction in [0, 1]^3, 
- set a maximum velocity (otherwise, the boids keep increasing speed if a bounding box or fixed direction is implemented)
- escape from the big bird, if it is close
- escape from a point on the screen, if it is close (this could be a mouse click or finger tap)
- force the boids within a bounding box (not currently used)

The big bird has 2 rules:

- go to the center of the flock
- - set a maximum velocity (otherwise, velocity keeps increasing and the big bird is hardly on the screen)
-
- Each of the rules has one or two parameters, to make their effect stronger or weaker. This can also be tweaked towards the actual visualisation used.

## Implementation notes

The boids are computed in one process, the visualisation in another.
