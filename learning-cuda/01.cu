#include <stdio.h>
#include <cuda_runtime_api.h>

/******************************************************************************
  Very simple CUDA program that shows the principles of copying data to and 
  from a GPU and dynamic memory allocation on a GPU. The standard pattern for
  a lot of GPU work is:
  
    1) Prepare the data on the host part of the program. By host we mean the 
       the CPU. In this case, the h_n integer is set to 19. The h_ prefix
       indicates a host variable, i.e. one that we will use with the CPU side
       of the program.
    2) Allocate memory on the device. By device we mean GPU. In this case a 
       single integer, identified by d_n, is allocated using cudaMalloc. The 
       d_ prefix indicates a device variable, i.e. one that we will use with 
       the GPU side of the program.
    3) Transfer data from the host to device. In this case cudaMemcpy is used 
       to copy the contents of h_n to d_n.
    4) The kernel function is invoked. In this case the kernel function is 
       called kernel and is defined as __global__ which means a function
       that will execute on the device but is invoked from the host. The
       <<<1,1>>> part indicates that we want to execute the kernel with one 
       thread block consisting of one thread. The kernel function here will
       only be invoked once in total.
    5) The kernel function is executed, which in this case sets the contents
       of the memory pointed to by d_n to 97.
    6) Data is copied from the device to the host. In this case the contents
       of memory pointed to by d_n are copied into the h_n variable.
    7) Dynamically allocated memory is freed using cudaFree.
    8) Results are output. In this case the value of h_n is printed, and if 
       all goes well should print 97.

  CUDA functions return an integer code. If this code is not equal to zero
  something has gone wrong. cudaGetErrorString returns a description of an 
  error given its code This program is rather paranoid and checks 
  the return codes of all call CUDA function calls and terminates the program
  if zero was not returned.

  To compile:  
    nvcc -o 01 01.cu
   
  Dr Kevan Buckley, University of Wolverhampton, 2018
******************************************************************************/

__global__ void kernel(int *n){
  *n = 97; // this is an arbitary number, just to see some results.
}

int main() {
  cudaError_t error;
  int *d_n;
  int h_n = 19;
  
  error = cudaMalloc(&d_n, sizeof(int));
  if(error){
    fprintf(stderr, "cudaMalloc on d_n returned %d %s\n", error,
      cudaGetErrorString(error));
    exit(1);
  }

  error = cudaMemcpy(d_n, &h_n, sizeof(int), cudaMemcpyHostToDevice);
  if(error){
    fprintf(stderr, "cudaMemcpy to d_n returned %d %s\n", error,
      cudaGetErrorString(error));
  }
  
  kernel <<<1,1>>>(d_n);
  cudaThreadSynchronize();

  error = cudaMemcpy(&h_n, d_n, sizeof(int), cudaMemcpyDeviceToHost);  
  if(error){
    fprintf(stderr, "cudaMemcpy to h_n returned %d %s\n", error,
      cudaGetErrorString(error));
  }

  error = cudaFree(d_n);
  if(error){
    fprintf(stderr, "cudaFree on d_n returned %d %s\n", error,
      cudaGetErrorString(error));
    exit(1);
  }
  
  printf("result: h_n = %d\n", h_n);
  return 0;
}

