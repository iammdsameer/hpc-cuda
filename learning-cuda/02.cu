#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <math.h>
#include <cuda_runtime_api.h>

/******************************************************************************
  The main hinderance to programming Tesla architecture GPUs is that they are
  "running blind" - you cannot use printf to output results so debugging is
  very difficult. With the introduction of the Fermi architecture printf was
  enabled. This program needs to use printf whilst exploring thread ids so
  cannot be run on GPUs with compute capability less than 2.0. A compiler
  directive to enforce this is shown below.

  cudaThreadSynchronize() ensures that all GPU threads have completed execution
  before the host code continues.
  
  The code demonstrates thread indexing. Threads are grouped into blocks.
  Blocks are grouped into a grid. Both the grid and blocks are 3 dimensional
  so a specific thread needs to be indexed using the x, y, z index of
  the block in the grid and the x, y, z index of the block in the thread.
  
  The first two examples use default 1 dimensional grid and block, so only the 
  x components of the indices is important. The other 2 examples demonstrate
  using three dimensional grid and blocks respectively.  
  
  The long printf call in the kernel cannot be broken into two calls or the
  output becomes interleaved.
  
  Be careful not to run a program that will call printf too many times as this
  can fill buffers and cause a system crash.
  
  To compile:  
    nvcc -o 02 02.cu
   
  Dr Kevan Buckley, University of Wolverhampton, 2018
******************************************************************************/

__global__ void kernel(){
  printf(
    "blockIdx.x=%-5d blockIdx.y=%-5d blockIdx.z=%-5d threadIdx.x=%-5d threadIdx.y=%-5d threadIdx.z=%-5d\n", 
    blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z);  
}

int main() {
  
  printf("Running with kernel <<<3,2>>>()\n");
  kernel <<<3,2>>>();
  cudaThreadSynchronize();

  printf("\nRunning with kernel <<<2,4>>>()\n");
  kernel <<<2,4>>>();
  cudaThreadSynchronize();

  dim3 dim(2, 3, 4);
  printf("\nRunning with kernel <<<dim,2>>>()\n");
  kernel <<<dim,2>>>();
  cudaThreadSynchronize();

  printf("\nRunning with kernel <<<2, dim>>>()\n");
  kernel <<<2, dim>>>();
  cudaThreadSynchronize();

  return 0;
}

