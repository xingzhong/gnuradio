
void ssp_kernel(float *x, float *y, int N){
	int i;
	for (i=0; i<N; i++)
		y[i] = x[i];

}

/*
int main(){
	float x[] = {1.1,2.2,3.3,4.4};
	float y[4];
	ssp_kernel(x, y, 4);
	printf("%f %f %f %f\n", y[0], y[1], y[2], y[3]);
	return 1;
}
*/
