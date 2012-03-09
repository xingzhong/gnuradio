#include <stdio.h>
#include <cutil_inline.h>
#include <shrQATest.h>

// Variables
float* h_A;
float* h_B;
float* d_A;
float* d_B;


// Device code
__global__ void VecAdd(const float* A,  float* B, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
        B[i] = A[i];
}

// Host code
int main(int argc, char** argv)
{
    shrQAStart(argc, argv);

    printf("Vector Equal\n");
    int N = 50000;
    size_t size = N * sizeof(float);

    // Allocate input vectors h_A and h_B in host memory
    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
   
    // Initialize input vectors
    for (int i = 0; i < N; ++i)
        h_A[i] = rand() / (float)RAND_MAX;

    // Allocate vectors in device memory
    cutilSafeCall( cudaMalloc((void**)&d_A, size) );
    cutilSafeCall( cudaMalloc((void**)&d_B, size) );
 

    // Copy vectors from host memory to device memory
    cutilSafeCall( cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice) );
    
    // Invoke kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    VecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, N);
   
    // Copy result from device memory to host memory
    // h_C contains the result in host memory
    cutilSafeCall( cudaMemcpy(h_B, d_B, size, cudaMemcpyDeviceToHost) );
    printf("The h_A:\n");
    for( int i=0; i<100; i++)
    {printf("%f ",h_A[i]);}
    printf("\nThe h_B:\n");
    for( int i=0; i<100; i++)
    {printf("%f ",h_B[i]);}
    cudaFree(d_A);
    cudaFree(d_B);
    free(h_A);
    free(h_B);
}
   


