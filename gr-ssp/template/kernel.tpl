\#include <${funcName}.h>

void ssp_kernel(const ${t1} *input, ${t2} *output, int N){
	int i;
	for (i=0; i<N; i++)
		output[i] = input[i]*input[i];
}


