\#include <${funcName}.h>

// Device Code
__global__ void cuda_kernel(const ${t1} *${input}, ${t2} *${output}, int ${sizeout}){
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i<${sizeout}){
		${kernel}
	}
}


// Host Code
void ssp_kernel(const ${t1} *${input}, ${t2} *${output}, int ${sizeout}){
	${t1} *d_A;
	${t2} *d_B;
	size_t size = ${sizeout} * sizeof(${output});
	cudaMalloc((void**)&d_A, size);
	cudaMalloc((void**)&d_B, size);
	cudaMemcpy(d_A, ${input}, size, cudaMemcpyHostToDevice);
	
	int threadsPerBlock = 256;
	int blocksPerGrid = (${sizeout} + threadsPerBlock - 1)/threadsPerBlock;
	cuda_kernel <<< blocksPerGrid, threadsPerBlock >>> (d_A, d_B, ${sizeout});
	cudaMemcpy(${output}, d_B, size, cudaMemcpyDeviceToHost);
	cudaFree(d_A);
	cudaFree(d_B);

}

