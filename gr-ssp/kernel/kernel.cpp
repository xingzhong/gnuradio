#include <ssp_myradio_ff.h>

void ssp_kernel(const float *input, float *output, int N){
	int i;
	for (i=0; i<N; i++)
		output[i] = input[i]*input[i];
}


