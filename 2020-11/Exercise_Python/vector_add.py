## This code performs vector addition and tests perfromance of CPU vs. GPU.
import numpy as np
from timeit import default_timer as timer

#load an enhanced-performance python compiler that works with GPU
from numba import vectorize 
 
# The size of the array. This should be a substantially high value in order 
# to test the difference between CPU and GPU:
array_size = 20000000
 
# The CPU version of addition product. Gets arrays a and b and returns thier addition.
def vector_add_cpu(a, b):
	c = np.zeros(array_size, dtype=np.float32)
	for i in range(array_size):
		c[i] = a[i] + b[i]
	return c
 
# This is the GPU version. Note the @vectorize decorator. This tells
# numba to turn this into a GPU vectorized function.
@vectorize(["float32(float32, float32)"], target='cuda')
def vector_add_gpu(a, b):
	return a + b;
 
def main():
 	#define the original vectors as arrays of ones of size: array_size:
	vec1 = np.ones(array_size, dtype=np.float32)
	vec2 = np.ones(array_size, dtype=np.float32)
 
  	# Time the CPU function:
	start = timer()
	vector_add_cpu(vec1, vec2)
	cpu_run_time = timer() - start
 
	# Time the GPU function:
	start = timer()
	vector_add_gpu(vec1, vec2)
	gpu_run_time = timer() - start
 
	# Report times
	print("CPU vector add function took %f seconds." % cpu_run_time)
	print("GPU vector add function took %f seconds." % gpu_run_time)
 
	return 0
 
if __name__ == "__main__":
	main()

