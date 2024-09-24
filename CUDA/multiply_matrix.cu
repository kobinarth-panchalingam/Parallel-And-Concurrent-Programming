#include <stdio.h>
#include <cuda_runtime.h>
#include <chrono>

// Kernel function to add the elements of two arrays
__global__ void multiply_matrix(int *a, int *b, int *c, int n) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y;
    int sum = 0;
    if (i < n && j < n) {
    for (int k=0; k<n; k++) {
        sum += a[ k + n*i ] * b[ k*n + j];
    }
    c[n*i + j] = sum;
    }
}

int main () {
    int n = 4;
    int MAX = 4;
    int size = n * n * sizeof(int);

    // Allocate host memory for vectors
    int *h_a = (int *) malloc( size);
    int *h_b = (int *) malloc( size);
    int *h_c = (int *) malloc( size);

    // Assign random values for vector a and b
    for (int i=0; i< n * n; i++) {
        int value = rand() % MAX;
        h_a[i] = value;
        h_b[i] = value;
    }

    // Allocate memory in device memory
    int *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);

    // Copy data from host to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // Define block size and grid size
    dim3 block_dim(16);
    dim3 grid_dim((n + block_dim.x - 1) / block_dim.x, n);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Start CPU timing
    auto cpu_start = std::chrono::high_resolution_clock::now();

    // Start GPU timing
    cudaEventRecord(start);

    // Launch the kernel 
    multiply_matrix<<<grid_dim, block_dim>>>(d_a, d_b, d_c, n);

    // Stop GPU timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Stop CPU timing
    auto cpu_stop = std::chrono::high_resolution_clock::now();

    // Calculate GPU elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Calculate CPU elapsed time
    std::chrono::duration<double> cpu_duration = cpu_stop - cpu_start;

    // Copy the result back to host
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    // Print the a and b matrices
    printf("Matrix A:\n");
    for (int i=0; i<n; i++) {
        for (int j=0; j<n; j++) {
            printf("%d ", h_a[i*n + j]);
        }
        printf("\n");
    }

    printf("Matrix B:\n");
    for (int i=0; i<n; i++) {
        for (int j=0; j<n; j++) {
            printf("%d ", h_b[i*n + j]);
        }
        printf("\n");
    }

    // print the result
    printf("Matrix C:\n");
    for (int i=0; i<n; i++) {
        for (int j=0; j<n; j++) {
            printf("%d ", h_c[i*n + j]);
        }
        printf("\n");
    }

    // Print the timing results
    printf("Time taken by GPU: %f ms\n", milliseconds);
    printf("Time taken by CPU: %f s\n", cpu_duration.count());

    // Free the memory
    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // Destroy CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}

// nvcc multiply_matrix.cu -o multiply_matrix.exe