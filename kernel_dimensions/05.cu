#include <stdio.h>
#include <errno.h>
#include <cuda_runtime_api.h>

/****************************************************************************
 * An experiment with cuda kernel invocation parameters. 2x3x4 threads on  
 * 5 blocks should yield 120 kernel invocations.
 *
 * Compile with:
 *   nvcc -o 05 05.cu
 * 
 * Dr Kevan Buckley, University of Wolverhampton, January 2018
 *****************************************************************************/

__global__ void kernel(){
  int i = (blockIdx.x * blockDim.z * blockDim.y * blockDim.x) +
          (threadIdx.z * blockDim.y * blockDim.x) + 
          (threadIdx.y * blockDim.x) + 
           threadIdx.x;

  printf("gd(%4d,%4d,%4d) bd(%4d,%4d,%4d) bi(%4d,%4d,%4d) ti(%4d,%4d,%4d) %d\n",
    gridDim.x, gridDim.y, gridDim.z, 
    blockDim.x, blockDim.y, blockDim.z,
    blockIdx.x, blockIdx.y, blockIdx.z,
    threadIdx.x, threadIdx.y, threadIdx.z, i); 
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

  dim3 bd(2, 3, 4);
  kernel <<<5, bd>>>();
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

