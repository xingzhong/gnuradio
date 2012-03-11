#include <ssp_myradio_ff.h>

void ssp_kernel(const float *in, float *out, int M){
	int i,j;
for (i=0; i<M; i++){
	out[i]=0;
	for (j=i; j<M; j++)
		out[i] += in[j];
}

}


