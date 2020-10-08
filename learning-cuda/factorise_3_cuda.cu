/****************************************************************************
  Similar to the factorise programs studied with POSIX threads, only this 
  version runs on a GPU with CUDA.

  Compile with:
    nvcc -o factorise_3_cuda factorise_3_cuda.cu -lrt
  
  Dr Kevan Buckley, University of Wolverhampton, 2018
*****************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>
#include <errno.h>
#include <sys/stat.h>
#include <string.h>
#include <time.h>
#include <pthread.h>
#include <math.h>

#define goal 98931313

__global__ void factorise(){
  int a = threadIdx.x;
  int b = blockIdx.x;
  int c = blockIdx.y;
  
  if(a*b*c == goal){
     printf("solution is %d, %d, %d\n", a, b, c);
  }
}

int time_difference(struct timespec *start, struct timespec *finish, 
                              long long int *difference) {
  long long int ds =  finish->tv_sec - start->tv_sec; 
  long long int dn =  finish->tv_nsec - start->tv_nsec; 

  if(dn < 0 ) {
    ds--;
    dn += 1000000000; 
  } 
  *difference = ds * 1000000000 + dn;
  return !(*difference > 0);
}

int main() {
  cudaError_t error;
  struct timespec start, finish;   
  long long int time_elapsed;

  clock_gettime(CLOCK_MONOTONIC, &start);

  dim3 gd(1000, 1000, 1); 
  dim3 bd(1000, 1, 1);
  factorise<<<gd, bd>>>();

  cudaDeviceSynchronize();

  error = cudaGetLastError();
  
  if(error){
    fprintf(stderr, "Kernel launch returned %d %s\n", 
      error, cudaGetErrorString(error));
    return 1;
  } else {
    fprintf(stderr, "Kernel launch successful.\n");
  }
  clock_gettime(CLOCK_MONOTONIC, &finish);
  time_difference(&start, &finish, &time_elapsed);
  printf("Time elapsed was %lldns or %0.9lfs\n", 
    time_elapsed, (time_elapsed/1.0e9)); 

  return 0;
}
