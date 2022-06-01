/*
Demonstration of CUDA programming in C using SAXPY as an example. SAXPY
stands for “Single-precision A*X Plus Y”.

Adapted from:
https://developer.nvidia.com/blog/easy-introduction-cuda-c-and-c/?ncid=afm-chs-44270&ranMID=44270&ranEAID=a1LgFw09t88&ranSiteID=a1LgFw09t88-IKbgke2Ie3siiYxpJZGAog

*/

#include <stdio.h>
#include <math.h>
#include <malloc.h>

/*

Device Code:
Kernels are defined using __global__ declaration specifier. 

*/

__global__
void saxpy(int n, double a, double *x, double *y)
{

  int i = blockIdx.x*blockDim.x + threadIdx.x;

/*
  n, a & i variables are stored by each thread in a register, and
  pointers x and y point to the device memory address space.
  Kernel is executed by multiple threads in parallel. Hence, each
  thread needs to be distinguished and identified. CUDA defines the
  variables "blockDim", "blockIdx", and "threadIdx". The predefined
  variable "blockDim" contains the dimensions of each thread block as
  specified in the second execution configuration parameter for the
  kernel launch. The variables "threadIdx" and "blockIdx" contain the
  index of the thread within its thread block and the thread block
  within the grid, respectively. 
*/

/*
  To ensure that there are no out-of-bounds memory accesses. It is
  required for cases where the number of elements in an array is not
  evenly divisible by the thread block size, and as a result the
  number of threads launched by the kernel is larger than the array
  size.
*/

  if (i < n) {
    y[i] = a*x[i] + y[i];
  }

}


// Host Code
//
int main(void) 
{

  int j, size = 1<<20;
  double *host_x, *host_y, *device_x, *device_y;
  double maxError = 0.0;


  // USING MALLOC() FOR HOST VECTORS TO AVOID HEAP OVERFLOW
  // ALLOCATE HOST MEMORY
  host_x = (double*) malloc(sizeof(double) * size);
  host_y = (double*) malloc(sizeof(double) * size);

  // ALLOCATE DEVICE MEMORY 
  // using cudaMalloc function from the CUDA runtime API
  cudaMalloc(&device_x, size*sizeof(double)); 
  cudaMalloc(&device_y, size*sizeof(double));


  // INITIALIZE HOST DATA
  for (j = 0; j < size; j++) {
    host_x[j] = 1.0; 
    host_y[j] = 2.0;
  }

  printf("\n Start of Main\n");

  // TRANSFER DATA FROM HOST TO DEVICE
  // using cudaMemcpy, which is similar to standard C memcpy function
  // but takes a 4th argument to specify the direction of the copy

  cudaMemcpy(device_x, host_x, size*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(device_y, host_y, size*sizeof(double), cudaMemcpyHostToDevice);


  // LAUNCH/EXECUTE KERNEL
  // Perform SAXPY on 1M elements
  // almost same as calling a C function
  saxpy<<<(size+255)/256, 256>>>(size, 2.0, device_x, device_y);

/*

  Information between triple chevrons is the "execution
  configuration". It dictates how many GPU threads execute the kernel
  in parallel. A kernel is launched with a "grid of thread
  blocks". First argument in the execution configuration specifies the
  number of thread blocks in the grid, and the second stands for the
  number of threads in a thread block.

  In this example, we have chosen to launch the kernel with 256
  threads in a thread block, and [(N+255)/256] number of thread
  blocks, where N is the number of elements in an array (which is 1M
  here).

*/


  // TRANSFER RESULTS FROM GPU TO HOST
  // using cudaMemcpy with cudaMemcpyDeviceToHost
  cudaMemcpy(host_y, device_y, size*sizeof(double), cudaMemcpyDeviceToHost);

  // Test to see if the kernel performed as expected
  for (j = 0; j < size; j++) {
    
    maxError = max(maxError, abs(host_y[j] - 4.0));
  }

  printf("Max error for this calculation is: %f\n", maxError);


  // FREE DEVICE AND HOST MEMORY
  cudaFree(device_x);
  cudaFree(device_y);
  free(host_x);
  free(host_y);

  printf("\n End of Main\n");
  return 0;

}
