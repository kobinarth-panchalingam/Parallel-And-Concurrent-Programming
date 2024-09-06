#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

int thread_count;

double** allocate_matrix( int size);

int main (int argc, char* argv[]) {
    long thread;
    pthread_t* thread_handles;

    // Get number of threads from command line
    thread_count = strtol(argv[1], NULL, 10);

    thread_handles = malloc(thread_count * sizeof(pthread_t));

    int m, n;

    for (thread = 0; thread < thread_count; thread++) {
        pthread_create(&thread_handles[thread], NULL, Hello, (void*) thread);
    }

    printf("Hello from main thread\n");

    for (thread=0; thread < thread_count; thread++) {
        pthread_join(thread_handles[thread], NULL);
    }

    free(thread_handles);
    return 0;
}

void* Pth_mat_vect(void* rank) {
    long my_rank = (long) rank;
    int i, j;
    int local_m = m / sthread_count;
    int my_first_row = my_rank * local_m;
    int my_last_row =  (my_rank + 1) * local_m - 1
}


double ** allocate_matrix( int size )
{
  /* Allocate 'size' * 'size' doubles contiguously. */
  double* vals = (double*) malloc( size * size * sizeof(double) );

  /* Allocate array of double* with size 'size' */
  double** ptrs = (double**) malloc( size * sizeof(double*) );

  int i;
  for (i = 0; i < size; ++i) {
    ptrs[ i ] = &vals[ i * size ];
  }

  return ptrs;
}