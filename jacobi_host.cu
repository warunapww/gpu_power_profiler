#include <assert.h>
#include <stdio.h>
#include "jacobi_kernel.hu"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <assert.h>
#define TIME
#define N 4096
#define T 8192
float A [2][N][N];

#include <unistd.h>
#include <sys/time.h>

#include "high_resolution_power.h"

#ifdef TIME
#define IF_TIME(foo) foo;
#else
#define IF_TIME(foo)
#endif

float *dev_A;

void *call_gpu_functions(void *nothing);
void *gpu_data_reset(void *nothing);


void init_array()
{
    int i, j, k;

    for (i=0; i<N; i++) {
    for (j=0; j<N; j++)
	for (k = 0; k < 2; k++) {
            A [k][i][j] = ((float) i*(j+2) + 2) / N;
	}
    }
}


void print_array()
{
    int i, j;
    for (i=0; i<N; i++) {
    	for (j=0; j<N; j++)
        	fprintf(stdout, "%0.15lf ", A [0][i][j]);
        fprintf(stdout, "\n");
    }
    fprintf(stdout, "\n");
}

double rtclock()
{
    struct timezone Tzp;
    struct timeval Tp;
    int stat;
    stat = gettimeofday (&Tp, &Tzp);
    if (stat != 0) printf("Error return from gettimeofday: %d",stat);
    return(Tp.tv_sec + Tp.tv_usec*1.0e-6);
}



int main(int argc, char **argv)
{
    long t, i, j;
    double t_start, t_end;

    init_array();
     #ifdef __CUDACC__
	// Initialize cuda before starting the timing.
        float *dev_X;
        cudaMalloc((void **) &dev_X, 1);
     #endif

    IF_TIME(t_start = rtclock());
	
#ifdef P4A
#pragma scop
    for (t=0; t < T/2; t++) {
      for (i=1; i < N-1; i++)
	#pragma ivdep
      	for (j=1; j < N-1; j++)
A[1][i][j] = (0.2f) * (A[0][i][j] + A[0][i][j-1] + A[0][i][j+1] + A[0][i+1][j] + A[0][i-1][j] );
      for (i=1; i < N-1; i++)
	#pragma ivdep
      	for (j=1; j < N-1; j++)
A[0][i][j] = (0.2f) * (A[1][i][j] + A[1][i][j-1] + A[1][i][j+1] + A[1][i+1][j] + A[1][i-1][j] );

    }
#pragma endscop

#else
{
#define cudaCheckReturn(ret) \
  do { \
    cudaError_t cudaCheckReturn_e = (ret); \
    if (cudaCheckReturn_e != cudaSuccess) { \
      fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(cudaCheckReturn_e)); \
      fflush(stderr); \
    } \
    assert(cudaCheckReturn_e == cudaSuccess); \
  } while(0)
#define cudaCheckKernel() \
  do { \
    cudaCheckReturn(cudaGetLastError()); \
  } while(0)

  
  cudaCheckReturn(cudaMalloc((void **) &dev_A, (2) * (4096) * (4096) * sizeof(float)));
  
  cudaCheckReturn(cudaMemcpy(dev_A, A, (2) * (4096) * (4096) * sizeof(float), cudaMemcpyHostToDevice));
  
  #define floord(n,d) (((n)<0) ? -((-(n)+(d)-1)/(d)) : (n)/(d))
	
	//if you need to measure time explicitely, uncomment next two lines
	//long long exec_time_ns = get_exec_time_in_nanoseconds(call_gpu_functions, gpu_data_reset);
	//printf("Execution time: %fs\n", exec_time_ns/1E9);

	high_resolution_power_profile(call_gpu_functions, gpu_data_reset);

  cudaCheckReturn(cudaMemcpy(A, dev_A, (2) * (4096) * (4096) * sizeof(float), cudaMemcpyDeviceToHost));
  
  cudaCheckReturn(cudaFree(dev_A));
}
#endif
	if (argc == 42)
        print_array();

    IF_TIME(t_end = rtclock());

#ifndef NGFLOPS
    IF_TIME(fprintf(stderr, "%0.3lfs, %f GFLOPS\n", t_end - t_start, 1.0e-9*9*T*(N-2)*(N-2)/(t_end-t_start)));
#else
    IF_TIME(fprintf(stderr, "%0.3lf\n", t_end - t_start, 1.0e-9*9*T*(N-2)*(N-2)/(t_end-t_start)));
#endif

    if (fopen(".test", "r")) {
        print_array();
    }
    return 0;
}

void *call_gpu_functions(void *nothing) {
	for (int h0 = 0; h0 <= 2048; h0 += 1) {
	{
		dim3 k0_dimBlock(128, 1);
		dim3 k0_dimGrid(342);
		kernel0_1 <<<k0_dimGrid, k0_dimBlock>>> (dev_A, h0);
		cudaCheckKernel();
	}
	
	if (h0 <= 2047)
	{
		dim3 k1_dimBlock(128, 1);
		dim3 k1_dimGrid(342);
		kernel1_1 <<<k1_dimGrid, k1_dimBlock>>> (dev_A, h0);
		cudaCheckKernel();
	}
		
  }
	return NULL;	
}


void *gpu_data_reset(void *nothing) {
	// for jacobi, you don't have to do anything here
	return NULL;
}
