import numpy
from OpenGL.GL import *

SIZEOF_FLOAT = 4

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
		
		self.num_vertices = len(self.vertex_array)		
		
		self.interleaved_array = []
		for v, n in zip(self.vertex_array, self.normal_array):
			self.interleaved_array.append(v)
			self.interleaved_array.append(n)
		
		#self.vertex_array = numpy.array(self.vertex_array, dtype=numpy.float32)
		#self.normal_array = numpy.array(self.normal_array, dtype=numpy.float32)
		self.interleaved_array = numpy.array(self.interleaved_array, dtype=numpy.float32)
						
		self.interleaved_buffer = glGenBuffers(1)		
		glBindBuffer(GL_ARRAY_BUFFER, self.interleaved_buffer)		
		glBufferData(GL_ARRAY_BUFFER, self.num_vertices*6*SIZEOF_FLOAT, self.interleaved_array, GL_STATIC_DRAW)						
				
		glBindBuffer(GL_ARRAY_BUFFER, 0)		
		
	def setup(self):
		glEnableClientState(GL_VERTEX_ARRAY)
		glEnableClientState(GL_NORMAL_ARRAY)
		
		glBindBuffer(GL_ARRAY_BUFFER, self.interleaved_buffer)		
		glVertexPointer(3, GL_FLOAT, SIZEOF_FLOAT*6, ctypes.c_void_p(0))
		glNormalPointer(GL_FLOAT, SIZEOF_FLOAT*6, ctypes.c_void_p(3*SIZEOF_FLOAT))
				
	def draw(self):		
		glDrawArrays(GL_TRIANGLES, 0, self.num_triangles*3)
		
	def done(self):
		glDisableClientState(GL_VERTEX_ARRAY)
		glDisableClientState(GL_NORMAL_ARRAY)
		
		glBindBuffer(GL_ARRAY_BUFFER, 0)
		

if __name__ == '__main__':
	
	import sys
	m = OBJModel(sys.argv[1])
	print m.vertex_array, m.normal_array