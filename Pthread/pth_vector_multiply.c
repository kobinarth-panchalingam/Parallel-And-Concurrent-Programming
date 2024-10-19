#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

int thread_count;
int n;
double **MatA, **MatB, **MatC;

void *mat_vect(void *rank);
double **allocate_matrix(int size);
void init_matrix(double** matrix, int size);

int main (int argc, char* argv[]) {
    long thread;
    pthread_t *thread_handles;

    // Get number of threads from command line
    thread_count = strtol(argv[1], NULL, 10);
    // Get matrix size from command line
    n = strtol(argv[2], NULL, 10);
    // Allocate memory for matrices
    MatA = allocate_matrix(n);
    MatB = allocate_matrix(n);
    MatC = allocate_matrix(n);
    // Initialize matrices
    init_matrix(MatA, n);
    init_matrix(MatB, n);

    thread_handles = malloc(thread_count * sizeof(pthread_t));

    for (thread = 0; thread < thread_count; thread++) {
        pthread_create(&thread_handles[thread], NULL, mat_vect, (void*) thread);
    }

    printf("Hello from main thread\n");

    for (thread=0; thread < thread_count; thread++) {
        pthread_join(thread_handles[thread], NULL);
    }

    free(thread_handles);
    return 0;
}

void* mat_vect(void* rank) {
    long my_rank = (long) rank;
    int i, j;
    int local_m = m / thread_count;
    int my_first_row = my_rank * local_m;
    int my_last_row = (my_rank + 1) * local_m - 1;
}


double **allocate_matrix( int size )
{
  /* Allocate 'size' * 'size' doubles contiguously. */
  double *vals = (double*) malloc( size * size * sizeof(double) );

  /* Allocate array of double* with size 'size' */
  double **ptrs = (double**) malloc( size * sizeof(double*) );

  int i;
  for (i = 0; i < size; ++i) {
    ptrs[ i ] = &vals[ i * size ];
  }

  return ptrs;
}

void init_matrix(double** matrix, int size) {
    int i, j;
    for (i = 0; i < size; i++) {
        for (j = 0; j < size; j++) {
            matrix[i][j] = 1.0;
        }
    }
}