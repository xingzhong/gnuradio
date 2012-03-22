#include <ssp_myradio_ff.h>

// Device Code
__global__ void cuda_kernel(const float *in, float *out, int M){
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i<M){
		out[i] = in[i]*in[i];

	}
}


// Host Code
void ssp_kernel(const float *in, float *out, int M){
	float *d_A;
	float *d_B;
	size_t size = M * sizeof(out);
	cudaMalloc((void**)&d_A, size);
	cudaMalloc((void**)&d_B, size);
	cudaMemcpy(d_A, in, size, cudaMemcpyHostToDevice);
	
	int threadsPerBlock = 256;
	int blocksPerGrid = (M + threadsPerBlock - 1)/threadsPerBlock;
	cuda_kernel <<< blocksPerGrid, threadsPerBlock >>> (d_A, d_B, M);
	cudaMemcpy(out, d_B, size, cudaMemcpyDeviceToHost);
	cudaFree(d_A);
	cudaFree(d_B);

}

