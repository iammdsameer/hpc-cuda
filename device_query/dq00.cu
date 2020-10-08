#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

/*****************************************************************************
 * This program checks if it is being run on a computer with a CUDA compatible
 * GPU, i.e. a modern nVidia GPU. If this program reports there are no
 * devices then no other programs in this set are going to work on the machine
 * being run on.
 * 
 * Compile with:
 *   nvcc -o dq00 dq00.cu
 * 
 * Dr Kevan Buckley, University of Wolverhampton, 2018 
 ****************************************************************************/
int main() {
  int device_count;
  cudaError_t error_id = cudaGetDeviceCount(&device_count);

  if (error_id != 0) {
    fprintf(stderr, "cudaGetDeviceCount returned %d\n-> %s\n", 
           (int)error_id, cudaGetErrorString(error_id));
    exit(1);
  }

  if (device_count == 0) {
    fprintf(stderr, "There are no available device(s) that support CUDA\n");
  } else {
    printf("Detected %d CUDA capable device(s)\n", device_count);
  }  
  
  return 0;
}
