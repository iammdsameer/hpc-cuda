#include <stdio.h>
#include <cuda_runtime_api.h>

/******************************************************************************
  This program adds two arrays of integers together and stores the results in
  another array. As the arrays are quite small then a single block can be used
  for all threads. Memory needs to be allocated on the device before copying
  the arrays to it. On completion of the execution only the result array needs
  to be copied back to the host. See next example for exactly the same thing
  but with the error checking removed. That version will enable you to see the
  sequence of operations more easily.
  
  Compile with:
    nvcc -o 03 03.cu
  
  Dr Kevan Buckley, University of Wolverhampton, 2018
******************************************************************************/

int h_a[] = {253, 215, 223, 116, 90, 184, 119, 180, 150, 175, 175, 18, 70, 18,
           103, 183, 247, 99, 175, 71, 230, 22, 75, 146, 87, 27, 157, 22, 176, 
           109, 190, 182, 65, 146, 252, 49, 153, 181, 247, 11, 1, 13, 171, 159,
           170, 205, 222, 46, 64, 134, 56, 191, 149, 64, 0, 174, 204, 118, 22,
           51, 14, 7, 20, 25, 3, 226, 15, 216, 99, 113, 10, 151, 41, 189, 204,
           198, 120, 92, 64, 97, 231, 185, 198, 118, 225, 197, 60, 252, 189, 
           186, 161, 81, 18, 243, 25, 233, 38, 212, 49, 173, 155, 113, 233, 
           56, 252, 134, 40, 16, 80, 192, 79, 50, 67, 158, 241, 231, 19, 165, 
           212, 76, 192, 161, 136, 224, 43, 39, 156, 27};
int h_b[] = {135, 113, 155, 52, 145, 172, 55, 112, 121, 248, 84, 216, 186, 111,
           107, 135, 149, 111, 184, 188, 60, 8, 238, 30, 35, 132, 210, 229, 
           153, 126, 8, 27, 21, 134, 250, 166, 240, 226, 121, 132, 221, 175, 
           247, 185, 68, 98, 178, 43, 65, 165, 1, 187, 16, 172, 251, 9, 191, 
           101, 193, 241, 167, 16, 108, 231, 117, 234, 59, 194, 164, 168, 
           242, 73, 202, 238, 211, 42, 92, 202, 202, 223, 5, 186, 220, 171, 
           165, 111, 45, 212, 79, 64, 235, 47, 245, 207, 20, 164, 189, 163, 
           160, 129, 27, 22, 16, 88, 58, 10, 149, 254, 52, 57, 167, 138, 71, 
           132, 183, 228, 178, 60, 190, 32, 23, 175, 193, 160, 250, 216, 145, 
           147};           
int h_c[128];   

int *d_a, *d_b, *d_c;        
           
__global__ void kernel(int *a, int *b, int *c){
  int i = threadIdx.x;
  c[i] = a[i] + b[i];
}

int main() {
  cudaError_t error;

  error = cudaMalloc(&d_a, sizeof(int) * 128);
  if(error){
    fprintf(stderr, "cudaMalloc on d_a returned %d %s\n", error,
      cudaGetErrorString(error));
    exit(1);
  }

  error = cudaMalloc(&d_b, sizeof(int) * 128);
  if(error){
    fprintf(stderr, "cudaMalloc on d_b returned %d %s\n", error,
      cudaGetErrorString(error));
    exit(1);
  }

  error = cudaMalloc(&d_c, sizeof(int) * 128);
  if(error){
    fprintf(stderr, "cudaMalloc on d_c returned %d %s\n", error,
      cudaGetErrorString(error));
    exit(1);
  }

  error = cudaMemcpy(d_a, &h_a, sizeof(int) * 128, cudaMemcpyHostToDevice);
  if(error){
    fprintf(stderr, "cudaMemcpy to d_b returned %d %s\n", error,
      cudaGetErrorString(error));
    exit(1);
  }  

  error = cudaMemcpy(d_b, &h_b, sizeof(int) * 128, cudaMemcpyHostToDevice);
  if(error){
    fprintf(stderr, "cudaMemcpy to d_b returned %d %s\n", error,
      cudaGetErrorString(error));
    exit(1);
  }

  kernel <<<1,128>>>(d_a, d_b, d_c);
  cudaThreadSynchronize();

  error = cudaMemcpy(h_c, d_c, sizeof(int) * 128, cudaMemcpyDeviceToHost);  
  if(error){
    fprintf(stderr, "cudaMemcpy to h_c returned %d %s\n", error,
      cudaGetErrorString(error));
    exit(1);
  }

  error = cudaFree(d_a);
  if(error){
    fprintf(stderr, "cudaFree on d_a returned %d %s\n", error,
      cudaGetErrorString(error));
    exit(1);
  }

  error = cudaFree(d_b);
  if(error){
    fprintf(stderr, "cudaFree on d_b returned %d %s\n", error,
      cudaGetErrorString(error));
    exit(1);
  }

  error = cudaFree(d_c);
  if(error){
    fprintf(stderr, "cudaFree on d_c returned %d %s\n", error,
      cudaGetErrorString(error));;
    exit(1);
  }

  int i;
  for(i=0;i<128;i++){
    printf("%-3d + %-3d = %-4d\n", h_a[i], h_b[i], h_c[i]);
  }  
  return 0;
}

