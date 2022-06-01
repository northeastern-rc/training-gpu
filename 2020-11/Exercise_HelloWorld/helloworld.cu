/* Demonstration of a "Hello World" Cuda example in C */

#include <stdio.h>
#define NUM_BLOCKS 4
#define BLOCK_WIDTH 8


/* Function executed on device GPU */
__global__ 
void hello() 
{
  printf("\tHello from GPU: thread %d and block %d\n", threadIdx.x, blockIdx.x);

}

/* Main function, executed on host CPU */
int main() {

  /* print message from CPU */
  printf( "Hello Cuda From CPU!\n" );

  /* execute function on device */
  hello<<<NUM_BLOCKS, BLOCK_WIDTH>>>();

  /* wait until all threads finish their job */
  cudaDeviceSynchronize();

  /* print message from CPU */
  printf( "Welcome back to CPU!\n" );

  return (0);
}
