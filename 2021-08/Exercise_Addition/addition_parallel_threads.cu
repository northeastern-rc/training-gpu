/* Demonstration of a simple kernel to add two integers  */
  
#include <stdio.h>
#define N 10

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
  // Note - we use pointers for the variables
  // add() runs on the device, so a, b and c must point to device memory
  c[threadIdx.x] = a[threadIdx.x] + b[threadIdx.x];
  // We use threadIdx.x instead of blockIdx.x
  printf("\tParameters on thread %d and block %d are: %d , %d \n Result: %d \n", threadIdx.x, blockIdx.x, a[threadIdx.x], b[threadIdx.x], c[threadIdx.x]);
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

  // Launch add() kernel on GPU with N threads: 
  add<<<1,N>>>(d_a, d_b, d_c);

  // Copy result back to host:
  cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

  // Cleanup:
  free(a); free(b); free(c);
  cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

  return 0;
}
