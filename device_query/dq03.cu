#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

/*****************************************************************************
 * This program introduces using cudaGetDeviceProperties to find about the 
 * available GPUs. This version of the program concentrates on the 
 * multithreading capabilities: * 

      int maxThreadsPerBlock;
      int maxThreadsDim[3];
      int maxGridSize[3];

 * 
 * Compile with:
 *   nvcc -o dq03 dq03.cu
 * 
 * Dr Kevan Buckley, University of Wolverhampton, 2018 
 ****************************************************************************/


int main() {
  
  cudaDeviceProp device_properties;
  cudaGetDeviceProperties(&device_properties, 0);

  printf("Maximum number of threads per block = %d\n", 
          device_properties.maxThreadsPerBlock);
  
  printf("Maximum size of each dimension of a block = [%d][%d][%d]\n", 
          device_properties.maxThreadsDim[0], 
          device_properties.maxThreadsDim[1],
          device_properties.maxThreadsDim[2]);
      
  printf("Maximum size of each dimension of a grid = [%d][%d][%d]\n", 
          device_properties.maxGridSize[0], 
          device_properties.maxGridSize[1],
          device_properties.maxGridSize[2]);  
              
  return 0;
}
