/*
 * Nvidia Tesla C2075 GPU
 * Jens Lang, 2013
 * This programme creates high-resolution power profiles for GPU routines executed on Nvidia GPUs.
 * It needs the CUDA, NVML and PAPI libraries. It should be compiled with gcc using the switch
 * -std=c++11. For further information, please refer to:
 * Lang, Jens; Rünger, Gudula: High-Resolution Power Profiling of GPU Functions Using Low-Resolution
 * Measurement. In: Wolf, F.; Mohr, B.; an Mey, D. (Hrsg.): Euro-Par 2013 Parallel Processing
 * (LNCS, Bd. 8097): S. 801–812. Springer  –  ISBN 978-3-642-40046-9, 2013. DOI: 10.1007/978-3-642-40047-6_80
 */
#include <papi.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "nvml.h"

#include "gpu_heater.h"

//#define M 2300
int M = 19456;
//int M = 1024;
#define IDX2C(i,j,M) (i*M+j)

cublasHandle_t handle;

typedef float matrix_t;
matrix_t* devPtrA;
matrix_t* devPtrB;
matrix_t* devPtrC;

void heatup_gpu() 
{
	// cuBLAS call
	matrix_t alpha = 2.3f;
	matrix_t beta = 5.7f;

	for (int xxx=0; xxx<1; xxx++) {
		cublasSgemm(
			handle, CUBLAS_OP_N, CUBLAS_OP_N,
			M, M, M,
			&alpha,
			devPtrA, M,
			devPtrB, M,
			&beta,
			devPtrC, M);

		cudaThreadSynchronize();	// Wait for the GPU launched work to complete
	}
//	status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha,
//	d_A, N, d_B, N, &beta, d_C, N);
}

void init_heater()
{

	/* Initialise random number generator (for sleep) */
/*	std::uniform_int_distribution<long long> random_sleep_time{20*1000*1000, 40*1000*1000};
	std::default_random_engine random_engine(std::chrono::system_clock::now().time_since_epoch().count());
*/
	cudaError_t cudaStat;
	cublasStatus_t stat;

	// allocate host memory for matrices and copy them to the device
	int i, j;
	matrix_t* a;
	matrix_t* b;
	//matrix_t* c;
	a = (matrix_t *)malloc (M * M * sizeof (*a));
	b = (matrix_t *)malloc (M * M * sizeof (*b));
	//c = (matrix_t *)malloc (M * M * sizeof (*c));
	if (!a) {
			printf ("host memory allocation failed\n");
			return;
	}
	for (j = 0; j < M; j++) {
			for (i = 0; i < M; i++) {
					a[IDX2C(i,j,M)] = (matrix_t)(i * M + j + 1);
					b[IDX2C(i,j,M)] = (matrix_t)(i * M + j - 1);
			}
	}
	cudaStat = cudaMalloc ((void**)&devPtrA, M*M*sizeof(*a));
	if (cudaStat != cudaSuccess) {
			printf ("device memory allocation failed\n");
			return;
	}
	cudaStat = cudaMalloc ((void**)&devPtrB, M*M*sizeof(*b));
	if (cudaStat != cudaSuccess) {
			printf ("device memory allocation failed\n");
			return;
	}
	cudaStat = cudaMalloc ((void**)&devPtrC, M*M*sizeof(*b));
	if (cudaStat != cudaSuccess) {
			printf ("device memory allocation failed\n");
			return;
	}
	stat = cublasCreate(&handle);
	if (stat != CUBLAS_STATUS_SUCCESS) {
			printf ("CUBLAS initialization failed\n");
			return;
	}
	stat = cublasSetMatrix (M, M, sizeof(*a), a, M, devPtrA, M);
	if (stat != CUBLAS_STATUS_SUCCESS) {
			printf ("data download failed\n");
			cudaFree (devPtrA);
			cublasDestroy(handle);
			return;
	}
	stat = cublasSetMatrix (M, M, sizeof(*b), b, M, devPtrB, M);
	if (stat != CUBLAS_STATUS_SUCCESS) {
			printf ("data download failed\n");
			cudaFree (devPtrB);
			cublasDestroy(handle);
			return;
	}

  printf("#Size of matrices: %d\n", M);
}

void shutdown_heater() 
{
	// shutdown cuBLAS
	cudaFree (devPtrA);
	cublasDestroy(handle);
}
