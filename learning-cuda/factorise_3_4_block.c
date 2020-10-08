/****************************************************************************
  Similar to factorise_3_0 but solves the problem with 4 threads using a 
  block method for search space partitioning. It is included here to 
  accompany a CUDA version of the program.

  Compile with:

    cc -o factorise_3_4_block factorise_3_4_block.c -lrt -pthread
  
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

typedef struct arguments {
  int start;
  int block_size;
} arguments_t;

void factorise(int n) {
  pthread_t t1, t2, t3, t4;

  arguments_t t1_arguments;
  t1_arguments.start = 0;
  t1_arguments.block_size = 250;

  arguments_t t2_arguments;
  t2_arguments.start = 250;
  t2_arguments.block_size = 250;

  arguments_t t3_arguments;
  t3_arguments.start = 500;
  t3_arguments.block_size = 250;

  arguments_t t4_arguments;
  t4_arguments.start = 750;
  t4_arguments.block_size = 250;

  void *find_factors();
  
  pthread_create(&t1, NULL, find_factors, &t1_arguments);
  pthread_create(&t2, NULL, find_factors, &t2_arguments);
  pthread_create(&t3, NULL, find_factors, &t3_arguments);
  pthread_create(&t4, NULL, find_factors, &t4_arguments);

  pthread_join(t1, NULL);
  pthread_join(t2, NULL);
  pthread_join(t3, NULL);
  pthread_join(t4, NULL);
}


void *find_factors(arguments_t *args){
  int a, b, c;
  for(a=args->start;a<args->start+args->block_size;a++){
    for(b=0;b<1000;b++){
      for(c=0;c<1000;c++){
        if(a*b*c == goal){
           printf("solution is %d, %d, %d\n", a, b, c);
        }
      }
    }
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
  struct timespec start, finish;   
  long long int time_elapsed;

  clock_gettime(CLOCK_MONOTONIC, &start);
 
  factorise(goal);

  clock_gettime(CLOCK_MONOTONIC, &finish);
  time_difference(&start, &finish, &time_elapsed);
  printf("Time elapsed was %lldns or %0.9lfs\n", 
    time_elapsed, (time_elapsed/1.0e9)); 

  return 0;
}
