/* Demonstration of a simple kernel to add two integers  */
  
#include <stdio.h>
#define NUM_BLOCKS 1
#define BLOCK_WIDTH 1

/* Function executed on device GPU */
__global__
void add(int *a, int *b, int *c)
{
  // Note - we use pointers for the variables
  // add() runs on the device, so a, b and c must point to device memory
  *c = *a + *b;
}

/* Main function, executed on host CPU */
int main() {

  int a, b, c; // host copies of a, b, c
  int *d_a, *d_b, *d_c; // device copies of a, b, c
  int size = sizeof(int);

  // Allocate space for device copies of a, b, c
  cudaMalloc((void **)&d_a, size);
  cudaMalloc((void **)&d_b, size);
  cudaMalloc((void **)&d_c, size);

  // Setup input values:
  a = 3;
  b = 8;

  printf( "The parameters are: a = %d, b = %d \n", a, b);

  // Copy inputs to device:
  cudaMemcpy(d_a, &a, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, &b, size, cudaMemcpyHostToDevice);

  // Launch add() kernel on GPU:
  add<<<1,1>>>(d_a, d_b, d_c);

  // Copy result back to host:
  cudaMemcpy(&c, d_c, size, cudaMemcpyDeviceToHost);

  printf( "The result is: a + b  = %d \n", c);

  // Cleanup:
  cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

  return 0;
}
