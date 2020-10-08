#include <stdio.h>
#include <errno.h>
#include <cuda_runtime_api.h>

/****************************************************************************
 * An experiment with cuda kernel invocation parameters. One thread on one 
 * block should yield one kernel invocation.
 *
 * Compile with:
 *   nvcc -o 01 01.cu
 * 
 * If you get a warning like:
 *   "nvcc warning : The 'compute_20', 'sm_20', and 'sm_21' architectures are deprecated"
 * you can use an alias to alter the way nvcc is invoked to suppress the 
 * warning. To do this type the following at the command prompt or to make a
 * more permanent change put it in your .bashrc startup script. 
 * 
 *   alias nvcc='nvcc -Wno-deprecated-gpu-targets'
 * 
 * By doing this whenever you enter the nvcc command it will include the 
 * switch to suppress the warning.
 * 
 * Dr Kevan Buckley, University of Wolverhampton, January 2018
 *****************************************************************************/


__global__ void kernel(){
  int i = threadIdx.x;

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

  kernel <<<1, 1>>>();
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

