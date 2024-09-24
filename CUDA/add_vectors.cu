#include <stdio.h>
#include <cuda_runtime.h>
#include <chrono>

// Kernel function to add the elements of two arrays
__global__ void add_vectors(int *a, int *b, int *c, int n) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    if (index < n) {
        c[index] = a[index] + b[index];
    }
}

int main () {
    int n = 1000;
    int MAX = 65536;
    int size = n * sizeof(int);

    // Allocate host memory for vectors
    int *h_a = (int *) malloc( size);
    int *h_b = (int *) malloc( size);
    int *h_c = (int *) malloc( size);

    // Assign random values for vector a and b
    for (int i=0; i<n; i++) {
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

    int block_size = 256;
    int num_blocks = (n + block_size - 1) / block_size;

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Start CPU timing
    auto cpu_start = std::chrono::high_resolution_clock::now();

    // Start GPU timing
    cudaEventRecord(start);

    // Launch the kernel 
    add_vectors<<<num_blocks, block_size>>>(d_a, d_b, d_c, n);

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

    // Print the result
    for (int i=0; i<n; i++) {
        printf("%d + %d = %d\n", h_a[i], h_b[i], h_c[i]);
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