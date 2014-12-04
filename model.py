import numpy
from OpenGL.GL import *

class OBJModel:
	
	def __init__(self, fname):
		
		vertices = []
		vnormals = []
		
		self.num_triangles = 0
		self.vertex_array = []
		self.normal_array = []
		
		f = open(fname, 'rt')			
		
		for line in f.readlines():
			if line.startswith('v '):
				x, y, z = map(float, line.strip().split()[1:])
				vertices.append((x, y, z))
			elif line.startswith('vn '):
				nx, ny, nz = map(float, line.strip().split()[1:])
				vnormals.append((nx, ny, nz))
			elif line.startswith('f '):
				pp = line.strip().split()[1:]
				assert len(pp) == 3 and "Expected triangles only"
				assert pp[0].find('//') != -1 and "Normals missing on faces"				
				
				for p in pp:
					vi, ni = map(int, p.split('//'))
					vi -= 1
					ni -= 1
										
					self.vertex_array.append(vertices[vi])
					self.normal_array.append(vnormals[ni])
					
				self.num_triangles += 1
				
		f.close()
		
		self.vertex_array = numpy.array(self.vertex_array)
		self.normal_array = numpy.array(self.normal_array)
		
	def setup(self):
		glEnableClientState(GL_VERTEX_ARRAY)
		glEnableClientState(GL_NORMAL_ARRAY)		
		glVertexPointer(3, GL_FLOAT, 0, self.vertex_array)
		glNormalPointer(GL_FLOAT, 0, self.normal_array)		
				
	def draw(self):		
		glDrawArrays(GL_TRIANGLES, 0, 3*self.num_triangles)
		
	def done(self):
		glDisableClientState(GL_VERTEX_ARRAY)
		glDisableClientState(GL_NORMAL_ARRAY)
		

if __name__ == '__main__':
	
	import sys
	m = OBJModel(sys.argv[1])
	print m.vertex_array, m.normal_array