#include <stdio.h>
#include <errno.h>
#include <cuda_runtime_api.h>

/****************************************************************************
 * An experiment with cuda kernel invocation parameters. This 
 * is to prove that the technique used for indexing large maps for the 
 * coursework is correct. Note thattoo many calls to printf from a kernel 
 * function will crash your computer (because the buffers between GPU and CPU 
 * will get full) so if you are going to invoke with large parameters use an 
 * if statement on i to only print the messages from the last few threads. 
 * This is explained by the commented out code which we would use if we were 
 * expecting 4 million threads.
 * 
 * To begin exploring thread indexing start with all grid and block dimensions
 * set to 1 and gradually build up block dimensions then grid dimensions. You
 * should try to estimate how many threads will be invoked before each 
 * experiment. Suggested experiment parameters are as follows:
 * 
 * 1.  bd(1, 1, 1) gd(1, 1, 1)
 * 2.  bd(2, 1, 1) gd(1, 1, 1)
 * 3.  bd(2, 3, 1) gd(1, 1, 1)
 * 4.  bd(2, 3, 4) gd(1, 1, 1)
 * 5.  bd(2, 3, 4) gd(2, 1, 1)
 * 6.  bd(2, 3, 4) gd(2, 3, 1)
 * 7.  bd(2, 3, 4) gd(2, 3, 4)
 * 8.  bd(5, 3, 4) gd(2, 3, 4) // include condition i > 999
 *
 * Compile with:
 *   nvcc -o 07 07.cu
 * 
 * Dr Kevan Buckley, University of Wolverhampton, January 2018
 *****************************************************************************/

__global__ void kernel(){
  int i = 
    threadIdx.x +
    (threadIdx.y * blockDim.x) +
    (threadIdx.z * blockDim.x * blockDim.y) + 
    (blockIdx.x * blockDim.x * blockDim.y * blockDim.z) +
    (blockIdx.y * blockDim.x * blockDim.y * blockDim.z * gridDim.x) +
    (blockIdx.z * blockDim.x * blockDim.y * blockDim.z * gridDim.x * gridDim.y);

//if(i>3999990){
  printf("gd(%4d,%4d,%4d) bd(%4d,%4d,%4d) bi(%4d,%4d,%4d) ti(%4d,%4d,%4d) %d\n",
    gridDim.x, gridDim.y, gridDim.z, 
    blockDim.x, blockDim.y, blockDim.z,
    blockIdx.x, blockIdx.y, blockIdx.z,
    threadIdx.x, threadIdx.y, threadIdx.z, i); 
//}
}

void advice(){
  printf("\ngd = gridDim\n");
  printf("bd = blockDim\n");  
  printf("bi = blockIdx\n");  
  printf("ti = threadIdx\n\n");
}

int main() {
  cudaError_t error;

//  advice();

  dim3 bd(1, 1, 1);
  dim3 gd(1, 1, 1);

  kernel <<<gd, bd>>>();
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

