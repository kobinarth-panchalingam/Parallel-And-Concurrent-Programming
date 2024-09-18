#include <stdio.h>
#include <cuda_runtime.h>

// Kernel function to add the elements of two arrays
__global__ void add_vectors(int *a, int *b, int *c, int n) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    if (index < n) {
        c[index] = a[index] + b[index];
    }
}

int main () {
    int n = 1000000;
    int MAX = 655636;

    // Allocate host memory for vectors
    int *h_a = (int *) malloc( n * sizeof(int));
    int *h_b = (int *) malloc( n * sizeof(int));
    int *h_c = (int *) malloc( n * sizeof(int));

    // Assign random values for vector a and b
    for (int i=0; i<n; i++) {
        int value = rand() % MAX;
        h_a[i] = value;
        h_b[i] = value;
    }

    // Allocate memory in device memory
    int *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, n * sizeof(int));
    cudaMalloc((void**)&d_b, n * sizeof(int));
    cudaMalloc((void**)&d_c, n * sizeof(int));

    // Copy data from host to device
    cudaMemcpy(d_a, h_a, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, n * sizeof(int), cudaMemcpyHostToDevice);

    int block_size = 256;
    int num_blocks = (n + block_size - 1) / block_size;

    // Launch the kernel 
    add_vectors<<<num_blocks, block_size>>>(d_a, d_b, d_c, n);

    // Copy the result back to host
    cudaMemcpy(h_c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);

    //  Print the result
    for (int i=0; i<n; i++) {
        printf("%d + %d = %d\n", h_a[i], h_b[i], h_c[i]);
    }

    // Free the memory
    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return 0;
}