/* Demonstration of a simple kernel to add two integers  */
  
#include <stdio.h>
#define N (10*10)  // overall size of the data set (100)
#define THREADS_PER_BLOCK 20  // Threads per block = 20, so total blocks=100/20=5

/* Function to generate an array of random numbers (range 0-100) "arr" of size "m" */
void random_ints(int *arr, int m)
{
   int i;
   for (i = 0; i < m; ++i)
    arr[i] = rand() % 100;
}

/* Function executed on device GPU */
__global__
void add(int *a, int *b, int *c)
{
  // add() uses both blocks and threads:
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  c[index] = a[index] + b[index];
  printf("\tParameters on thread %d and block %d are: %d , %d \n Result: %d \n", threadIdx.x, blockIdx.x, a[index], b[index], c[index]);
  // By using blockIdx.x to index into the array, each block handles a different element of the array
}

/* Main function, executed on host CPU */
int main() {

  int *a, *b, *c; // host copies of a, b, c
  int *d_a, *d_b, *d_c; // device copies of a, b, c
  int size = N * sizeof(int);

  // Alloc space for device copies of a, b, c
  cudaMalloc((void **)&d_a, size);
  cudaMalloc((void **)&d_b, size);
  cudaMalloc((void **)&d_c, size);

  // Alloc space for host copies of a, b, c and setup input values (arrays):  
  a = (int *)malloc(size); random_ints(a, N);
  b = (int *)malloc(size); random_ints(b, N);
  c = (int *)malloc(size);
 
  // Copy inputs to device:
  cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

  // Launch parallel with N/THREADS_PER_BLOCK blocks and THREADS_PER_BLOCK threads per block:
  add<<<N/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(d_a, d_b, d_c);

  // Copy result back to host:
  cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

  // Cleanup:
  free(a); free(b); free(c);
  cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

  return 0;
}
