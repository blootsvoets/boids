# Boids

A simple model of a flock of birds, attacked by a predating bird.

## Usage

Install NumPy, PIL, PyGame and PyOpenGL. To run boids, execute viewer.py with Python 2.

## Model

The boids movement at some point converges to single point. An example dynamics rule that is now implemented is to move a random 1/3rd of the boids to a random other location.

Classic rules that are enforced on the boids:

1. converge to the center of mass of the flock
2. if your neighbour is too close, move away from it
3. converge to the average velocity of the flock

For each rule, not the whole flock is considered, only the N closest neighbors. A few additional rules were added to keep the flock on screen and interesting:

- set a minimum and maximum velocity
- escape from the big bird, if it is close
- escape from a given point if it is close; this is activated by a mouse click.
- force the boids within a bounding box

The big bird has 2 rules:

- go to the center of the flock
- set a minimum and maximum velocity (otherwise, velocity keeps increasing and the big bird is hardly on the screen)

Each of the rules has one or two parameters, to make their effect stronger or weaker. This can also be tweaked towards the actual visualisation used.

## Implementation notes

The code uses 8 processes: 1 for visualisation; 1 for computing metrics; 3 for the simulation of the original flock; 3 for the simulation of the nudged flock.

The visualisation is done with OpenGL, with a custom font and bird mesh.

## Authors

Simulation and threading: Joris Borgdorff <j.borgdorff@esciencecenter.nl>
Visualisation and user interface: Paul Melis <paul.melis@surfsara.nl>