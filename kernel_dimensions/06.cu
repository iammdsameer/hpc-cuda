#include <stdio.h>
#include <errno.h>
#include <cuda_runtime_api.h>

/****************************************************************************
 * An experiment with cuda kernel invocation parameters. This 
 * is to prove that the technique used for a 200x200 map for the coursework
 * is correct.
 *
 * Compile with:
 *   nvcc -o 06 06.cu
 * 
 * Dr Kevan Buckley, University of Wolverhampton, January 2018
 *****************************************************************************/

__global__ void kernel(){
  int i = (blockIdx.x * blockDim.x) + threadIdx.x;
  if(i>39950){
    printf("gd(%4d,%4d,%4d) bd(%4d,%4d,%4d) bi(%4d,%4d,%4d) ti(%4d,%4d,%4d) %d\n",
      gridDim.x, gridDim.y, gridDim.z, 
      blockDim.x, blockDim.y, blockDim.z,
      blockIdx.x, blockIdx.y, blockIdx.z,
      threadIdx.x, threadIdx.y, threadIdx.z, i); 

  }
}

void advice(){
  printf("\ngd = gridDim\n");
  printf("bd = blockDim\n");  
  printf("bi = blockIdx\n");  
  printf("ti = threadIdx\n\n");
}

int main() {
  cudaError_t error;

  advice();

  kernel <<<200, 200>>>();
  cudaDeviceSynchronize();

  error = cudaGetLastError();
  
  if(error){
    fprintf(stderr, "Kernel launch returned %d %s\n", 
      error, cudaGetErrorString(error));
    return 1;
  } else {
    fprintf(stderr, "Kernel launch successful.\n");
  }
}

