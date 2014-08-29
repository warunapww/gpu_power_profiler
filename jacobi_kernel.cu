#include "jacobi_kernel.hu"
__global__ void kernel0_1(float *A, int h0)
{
    int b0 = blockIdx.x;
    int t0 = threadIdx.y, t1 = threadIdx.x;

    #define floord(n,d) (((n)<0) ? -((-(n)+(d)-1)/(d)) : (n)/(d))
    #define min(x,y)    ((x) < (y) ? (x) : (y))
    for (int g7 = 0; g7 <= min(32, -((h0 + 32) / 32) + 96); g7 += 1) {
      if (h0 >= 1 && g7 <= 31 && b0 >= 1) {
        if (t1 + 128 * g7 >= 1 && t1 + 128 * g7 <= 4094) {
          A[(1 * 4096 + (12 * b0 - 4)) * 4096 + (t1 + 128 * g7)] = (0.2f * ((((A[(0 * 4096 + (12 * b0 - 4)) * 4096 + (t1 + 128 * g7)] + A[(0 * 4096 + (12 * b0 - 4)) * 4096 + (t1 + 128 * g7 - 1)]) + A[(0 * 4096 + (12 * b0 - 4)) * 4096 + (t1 + 128 * g7 + 1)]) + A[(0 * 4096 + (12 * b0 - 3)) * 4096 + (t1 + 128 * g7)]) + A[(0 * 4096 + (12 * b0 - 5)) * 4096 + (t1 + 128 * g7)]));
          A[(1 * 4096 + (12 * b0 - 3)) * 4096 + (t1 + 128 * g7)] = (0.2f * ((((A[(0 * 4096 + (12 * b0 - 3)) * 4096 + (t1 + 128 * g7)] + A[(0 * 4096 + (12 * b0 - 3)) * 4096 + (t1 + 128 * g7 - 1)]) + A[(0 * 4096 + (12 * b0 - 3)) * 4096 + (t1 + 128 * g7 + 1)]) + A[(0 * 4096 + (12 * b0 - 2)) * 4096 + (t1 + 128 * g7)]) + A[(0 * 4096 + (12 * b0 - 4)) * 4096 + (t1 + 128 * g7)]));
          A[(1 * 4096 + (12 * b0 - 2)) * 4096 + (t1 + 128 * g7)] = (0.2f * ((((A[(0 * 4096 + (12 * b0 - 2)) * 4096 + (t1 + 128 * g7)] + A[(0 * 4096 + (12 * b0 - 2)) * 4096 + (t1 + 128 * g7 - 1)]) + A[(0 * 4096 + (12 * b0 - 2)) * 4096 + (t1 + 128 * g7 + 1)]) + A[(0 * 4096 + (12 * b0 - 1)) * 4096 + (t1 + 128 * g7)]) + A[(0 * 4096 + (12 * b0 - 3)) * 4096 + (t1 + 128 * g7)]));
          A[(1 * 4096 + (12 * b0 - 1)) * 4096 + (t1 + 128 * g7)] = (0.2f * ((((A[(0 * 4096 + (12 * b0 - 1)) * 4096 + (t1 + 128 * g7)] + A[(0 * 4096 + (12 * b0 - 1)) * 4096 + (t1 + 128 * g7 - 1)]) + A[(0 * 4096 + (12 * b0 - 1)) * 4096 + (t1 + 128 * g7 + 1)]) + A[(0 * 4096 + 12 * b0) * 4096 + (t1 + 128 * g7)]) + A[(0 * 4096 + (12 * b0 - 2)) * 4096 + (t1 + 128 * g7)]));
          A[(1 * 4096 + 12 * b0) * 4096 + (t1 + 128 * g7)] = (0.2f * ((((A[(0 * 4096 + 12 * b0) * 4096 + (t1 + 128 * g7)] + A[(0 * 4096 + 12 * b0) * 4096 + (t1 + 128 * g7 - 1)]) + A[(0 * 4096 + 12 * b0) * 4096 + (t1 + 128 * g7 + 1)]) + A[(0 * 4096 + (12 * b0 + 1)) * 4096 + (t1 + 128 * g7)]) + A[(0 * 4096 + (12 * b0 - 1)) * 4096 + (t1 + 128 * g7)]));
        }
        __syncthreads();
      }
      if (h0 >= 1 && g7 <= 31) {
        if (t1 + 128 * g7 >= 2 && b0 >= 1) {
          A[(0 * 4096 + (12 * b0 - 5)) * 4096 + (t1 + 128 * g7 - 1)] = (0.2f * ((((A[(1 * 4096 + (12 * b0 - 5)) * 4096 + (t1 + 128 * g7 - 1)] + A[(1 * 4096 + (12 * b0 - 5)) * 4096 + (t1 + 128 * g7 - 2)]) + A[(1 * 4096 + (12 * b0 - 5)) * 4096 + (t1 + 128 * g7)]) + A[(1 * 4096 + (12 * b0 - 4)) * 4096 + (t1 + 128 * g7 - 1)]) + A[(1 * 4096 + (12 * b0 - 6)) * 4096 + (t1 + 128 * g7 - 1)]));
          A[(0 * 4096 + (12 * b0 - 4)) * 4096 + (t1 + 128 * g7 - 1)] = (0.2f * ((((A[(1 * 4096 + (12 * b0 - 4)) * 4096 + (t1 + 128 * g7 - 1)] + A[(1 * 4096 + (12 * b0 - 4)) * 4096 + (t1 + 128 * g7 - 2)]) + A[(1 * 4096 + (12 * b0 - 4)) * 4096 + (t1 + 128 * g7)]) + A[(1 * 4096 + (12 * b0 - 3)) * 4096 + (t1 + 128 * g7 - 1)]) + A[(1 * 4096 + (12 * b0 - 5)) * 4096 + (t1 + 128 * g7 - 1)]));
          A[(0 * 4096 + (12 * b0 - 3)) * 4096 + (t1 + 128 * g7 - 1)] = (0.2f * ((((A[(1 * 4096 + (12 * b0 - 3)) * 4096 + (t1 + 128 * g7 - 1)] + A[(1 * 4096 + (12 * b0 - 3)) * 4096 + (t1 + 128 * g7 - 2)]) + A[(1 * 4096 + (12 * b0 - 3)) * 4096 + (t1 + 128 * g7)]) + A[(1 * 4096 + (12 * b0 - 2)) * 4096 + (t1 + 128 * g7 - 1)]) + A[(1 * 4096 + (12 * b0 - 4)) * 4096 + (t1 + 128 * g7 - 1)]));
          A[(0 * 4096 + (12 * b0 - 2)) * 4096 + (t1 + 128 * g7 - 1)] = (0.2f * ((((A[(1 * 4096 + (12 * b0 - 2)) * 4096 + (t1 + 128 * g7 - 1)] + A[(1 * 4096 + (12 * b0 - 2)) * 4096 + (t1 + 128 * g7 - 2)]) + A[(1 * 4096 + (12 * b0 - 2)) * 4096 + (t1 + 128 * g7)]) + A[(1 * 4096 + (12 * b0 - 1)) * 4096 + (t1 + 128 * g7 - 1)]) + A[(1 * 4096 + (12 * b0 - 3)) * 4096 + (t1 + 128 * g7 - 1)]));
          A[(0 * 4096 + (12 * b0 - 1)) * 4096 + (t1 + 128 * g7 - 1)] = (0.2f * ((((A[(1 * 4096 + (12 * b0 - 1)) * 4096 + (t1 + 128 * g7 - 1)] + A[(1 * 4096 + (12 * b0 - 1)) * 4096 + (t1 + 128 * g7 - 2)]) + A[(1 * 4096 + (12 * b0 - 1)) * 4096 + (t1 + 128 * g7)]) + A[(1 * 4096 + 12 * b0) * 4096 + (t1 + 128 * g7 - 1)]) + A[(1 * 4096 + (12 * b0 - 2)) * 4096 + (t1 + 128 * g7 - 1)]));
          A[(0 * 4096 + 12 * b0) * 4096 + (t1 + 128 * g7 - 1)] = (0.2f * ((((A[(1 * 4096 + 12 * b0) * 4096 + (t1 + 128 * g7 - 1)] + A[(1 * 4096 + 12 * b0) * 4096 + (t1 + 128 * g7 - 2)]) + A[(1 * 4096 + 12 * b0) * 4096 + (t1 + 128 * g7)]) + A[(1 * 4096 + (12 * b0 + 1)) * 4096 + (t1 + 128 * g7 - 1)]) + A[(1 * 4096 + (12 * b0 - 1)) * 4096 + (t1 + 128 * g7 - 1)]));
        }
        if (t1 + 128 * g7 >= 2)
          A[(0 * 4096 + (12 * b0 + 1)) * 4096 + (t1 + 128 * g7 - 1)] = (0.2f * ((((A[(1 * 4096 + (12 * b0 + 1)) * 4096 + (t1 + 128 * g7 - 1)] + A[(1 * 4096 + (12 * b0 + 1)) * 4096 + (t1 + 128 * g7 - 2)]) + A[(1 * 4096 + (12 * b0 + 1)) * 4096 + (t1 + 128 * g7)]) + A[(1 * 4096 + (12 * b0 + 2)) * 4096 + (t1 + 128 * g7 - 1)]) + A[(1 * 4096 + 12 * b0) * 4096 + (t1 + 128 * g7 - 1)]));
        __syncthreads();
      }
      if (h0 <= 2047) {
        if (t1 + 128 * g7 >= 3 && b0 >= 1 && t1 + 128 * g7 <= 4096) {
          A[(1 * 4096 + (12 * b0 - 5)) * 4096 + (t1 + 128 * g7 - 2)] = (0.2f * ((((A[(0 * 4096 + (12 * b0 - 5)) * 4096 + (t1 + 128 * g7 - 2)] + A[(0 * 4096 + (12 * b0 - 5)) * 4096 + (t1 + 128 * g7 - 3)]) + A[(0 * 4096 + (12 * b0 - 5)) * 4096 + (t1 + 128 * g7 - 1)]) + A[(0 * 4096 + (12 * b0 - 4)) * 4096 + (t1 + 128 * g7 - 2)]) + A[(0 * 4096 + (12 * b0 - 6)) * 4096 + (t1 + 128 * g7 - 2)]));
          A[(1 * 4096 + (12 * b0 - 4)) * 4096 + (t1 + 128 * g7 - 2)] = (0.2f * ((((A[(0 * 4096 + (12 * b0 - 4)) * 4096 + (t1 + 128 * g7 - 2)] + A[(0 * 4096 + (12 * b0 - 4)) * 4096 + (t1 + 128 * g7 - 3)]) + A[(0 * 4096 + (12 * b0 - 4)) * 4096 + (t1 + 128 * g7 - 1)]) + A[(0 * 4096 + (12 * b0 - 3)) * 4096 + (t1 + 128 * g7 - 2)]) + A[(0 * 4096 + (12 * b0 - 5)) * 4096 + (t1 + 128 * g7 - 2)]));
          A[(1 * 4096 + (12 * b0 - 3)) * 4096 + (t1 + 128 * g7 - 2)] = (0.2f * ((((A[(0 * 4096 + (12 * b0 - 3)) * 4096 + (t1 + 128 * g7 - 2)] + A[(0 * 4096 + (12 * b0 - 3)) * 4096 + (t1 + 128 * g7 - 3)]) + A[(0 * 4096 + (12 * b0 - 3)) * 4096 + (t1 + 128 * g7 - 1)]) + A[(0 * 4096 + (12 * b0 - 2)) * 4096 + (t1 + 128 * g7 - 2)]) + A[(0 * 4096 + (12 * b0 - 4)) * 4096 + (t1 + 128 * g7 - 2)]));
          A[(1 * 4096 + (12 * b0 - 2)) * 4096 + (t1 + 128 * g7 - 2)] = (0.2f * ((((A[(0 * 4096 + (12 * b0 - 2)) * 4096 + (t1 + 128 * g7 - 2)] + A[(0 * 4096 + (12 * b0 - 2)) * 4096 + (t1 + 128 * g7 - 3)]) + A[(0 * 4096 + (12 * b0 - 2)) * 4096 + (t1 + 128 * g7 - 1)]) + A[(0 * 4096 + (12 * b0 - 1)) * 4096 + (t1 + 128 * g7 - 2)]) + A[(0 * 4096 + (12 * b0 - 3)) * 4096 + (t1 + 128 * g7 - 2)]));
          A[(1 * 4096 + (12 * b0 - 1)) * 4096 + (t1 + 128 * g7 - 2)] = (0.2f * ((((A[(0 * 4096 + (12 * b0 - 1)) * 4096 + (t1 + 128 * g7 - 2)] + A[(0 * 4096 + (12 * b0 - 1)) * 4096 + (t1 + 128 * g7 - 3)]) + A[(0 * 4096 + (12 * b0 - 1)) * 4096 + (t1 + 128 * g7 - 1)]) + A[(0 * 4096 + 12 * b0) * 4096 + (t1 + 128 * g7 - 2)]) + A[(0 * 4096 + (12 * b0 - 2)) * 4096 + (t1 + 128 * g7 - 2)]));
          A[(1 * 4096 + 12 * b0) * 4096 + (t1 + 128 * g7 - 2)] = (0.2f * ((((A[(0 * 4096 + 12 * b0) * 4096 + (t1 + 128 * g7 - 2)] + A[(0 * 4096 + 12 * b0) * 4096 + (t1 + 128 * g7 - 3)]) + A[(0 * 4096 + 12 * b0) * 4096 + (t1 + 128 * g7 - 1)]) + A[(0 * 4096 + (12 * b0 + 1)) * 4096 + (t1 + 128 * g7 - 2)]) + A[(0 * 4096 + (12 * b0 - 1)) * 4096 + (t1 + 128 * g7 - 2)]));
        }
        if (t1 + 128 * g7 >= 3 && t1 + 128 * g7 <= 4096)
          A[(1 * 4096 + (12 * b0 + 1)) * 4096 + (t1 + 128 * g7 - 2)] = (0.2f * ((((A[(0 * 4096 + (12 * b0 + 1)) * 4096 + (t1 + 128 * g7 - 2)] + A[(0 * 4096 + (12 * b0 + 1)) * 4096 + (t1 + 128 * g7 - 3)]) + A[(0 * 4096 + (12 * b0 + 1)) * 4096 + (t1 + 128 * g7 - 1)]) + A[(0 * 4096 + (12 * b0 + 2)) * 4096 + (t1 + 128 * g7 - 2)]) + A[(0 * 4096 + 12 * b0) * 4096 + (t1 + 128 * g7 - 2)]));
        __syncthreads();
        if (b0 >= 1) {
          if (t1 + 128 * g7 >= 4 && t1 + 128 * g7 <= 4097) {
            A[(0 * 4096 + (12 * b0 - 4)) * 4096 + (t1 + 128 * g7 - 3)] = (0.2f * ((((A[(1 * 4096 + (12 * b0 - 4)) * 4096 + (t1 + 128 * g7 - 3)] + A[(1 * 4096 + (12 * b0 - 4)) * 4096 + (t1 + 128 * g7 - 4)]) + A[(1 * 4096 + (12 * b0 - 4)) * 4096 + (t1 + 128 * g7 - 2)]) + A[(1 * 4096 + (12 * b0 - 3)) * 4096 + (t1 + 128 * g7 - 3)]) + A[(1 * 4096 + (12 * b0 - 5)) * 4096 + (t1 + 128 * g7 - 3)]));
            A[(0 * 4096 + (12 * b0 - 3)) * 4096 + (t1 + 128 * g7 - 3)] = (0.2f * ((((A[(1 * 4096 + (12 * b0 - 3)) * 4096 + (t1 + 128 * g7 - 3)] + A[(1 * 4096 + (12 * b0 - 3)) * 4096 + (t1 + 128 * g7 - 4)]) + A[(1 * 4096 + (12 * b0 - 3)) * 4096 + (t1 + 128 * g7 - 2)]) + A[(1 * 4096 + (12 * b0 - 2)) * 4096 + (t1 + 128 * g7 - 3)]) + A[(1 * 4096 + (12 * b0 - 4)) * 4096 + (t1 + 128 * g7 - 3)]));
            A[(0 * 4096 + (12 * b0 - 2)) * 4096 + (t1 + 128 * g7 - 3)] = (0.2f * ((((A[(1 * 4096 + (12 * b0 - 2)) * 4096 + (t1 + 128 * g7 - 3)] + A[(1 * 4096 + (12 * b0 - 2)) * 4096 + (t1 + 128 * g7 - 4)]) + A[(1 * 4096 + (12 * b0 - 2)) * 4096 + (t1 + 128 * g7 - 2)]) + A[(1 * 4096 + (12 * b0 - 1)) * 4096 + (t1 + 128 * g7 - 3)]) + A[(1 * 4096 + (12 * b0 - 3)) * 4096 + (t1 + 128 * g7 - 3)]));
            A[(0 * 4096 + (12 * b0 - 1)) * 4096 + (t1 + 128 * g7 - 3)] = (0.2f * ((((A[(1 * 4096 + (12 * b0 - 1)) * 4096 + (t1 + 128 * g7 - 3)] + A[(1 * 4096 + (12 * b0 - 1)) * 4096 + (t1 + 128 * g7 - 4)]) + A[(1 * 4096 + (12 * b0 - 1)) * 4096 + (t1 + 128 * g7 - 2)]) + A[(1 * 4096 + 12 * b0) * 4096 + (t1 + 128 * g7 - 3)]) + A[(1 * 4096 + (12 * b0 - 2)) * 4096 + (t1 + 128 * g7 - 3)]));
            A[(0 * 4096 + 12 * b0) * 4096 + (t1 + 128 * g7 - 3)] = (0.2f * ((((A[(1 * 4096 + 12 * b0) * 4096 + (t1 + 128 * g7 - 3)] + A[(1 * 4096 + 12 * b0) * 4096 + (t1 + 128 * g7 - 4)]) + A[(1 * 4096 + 12 * b0) * 4096 + (t1 + 128 * g7 - 2)]) + A[(1 * 4096 + (12 * b0 + 1)) * 4096 + (t1 + 128 * g7 - 3)]) + A[(1 * 4096 + (12 * b0 - 1)) * 4096 + (t1 + 128 * g7 - 3)]));
          }
          __syncthreads();
        }
      }
    }
}
__global__ void kernel1_1(float *A, int h0)
{
    int b0 = blockIdx.x;
    int t0 = threadIdx.y, t1 = threadIdx.x;

    #define floord(n,d) (((n)<0) ? -((-(n)+(d)-1)/(d)) : (n)/(d))
    for (int g7 = 0; g7 <= 32; g7 += 1) {
      if (g7 <= 31) {
        if (t1 + 128 * g7 >= 1 && t1 + 128 * g7 <= 4094) {
          A[(1 * 4096 + (12 * b0 + 2)) * 4096 + (t1 + 128 * g7)] = (0.2f * ((((A[(0 * 4096 + (12 * b0 + 2)) * 4096 + (t1 + 128 * g7)] + A[(0 * 4096 + (12 * b0 + 2)) * 4096 + (t1 + 128 * g7 - 1)]) + A[(0 * 4096 + (12 * b0 + 2)) * 4096 + (t1 + 128 * g7 + 1)]) + A[(0 * 4096 + (12 * b0 + 3)) * 4096 + (t1 + 128 * g7)]) + A[(0 * 4096 + (12 * b0 + 1)) * 4096 + (t1 + 128 * g7)]));
          if (b0 <= 340) {
            A[(1 * 4096 + (12 * b0 + 3)) * 4096 + (t1 + 128 * g7)] = (0.2f * ((((A[(0 * 4096 + (12 * b0 + 3)) * 4096 + (t1 + 128 * g7)] + A[(0 * 4096 + (12 * b0 + 3)) * 4096 + (t1 + 128 * g7 - 1)]) + A[(0 * 4096 + (12 * b0 + 3)) * 4096 + (t1 + 128 * g7 + 1)]) + A[(0 * 4096 + (12 * b0 + 4)) * 4096 + (t1 + 128 * g7)]) + A[(0 * 4096 + (12 * b0 + 2)) * 4096 + (t1 + 128 * g7)]));
            A[(1 * 4096 + (12 * b0 + 4)) * 4096 + (t1 + 128 * g7)] = (0.2f * ((((A[(0 * 4096 + (12 * b0 + 4)) * 4096 + (t1 + 128 * g7)] + A[(0 * 4096 + (12 * b0 + 4)) * 4096 + (t1 + 128 * g7 - 1)]) + A[(0 * 4096 + (12 * b0 + 4)) * 4096 + (t1 + 128 * g7 + 1)]) + A[(0 * 4096 + (12 * b0 + 5)) * 4096 + (t1 + 128 * g7)]) + A[(0 * 4096 + (12 * b0 + 3)) * 4096 + (t1 + 128 * g7)]));
            A[(1 * 4096 + (12 * b0 + 5)) * 4096 + (t1 + 128 * g7)] = (0.2f * ((((A[(0 * 4096 + (12 * b0 + 5)) * 4096 + (t1 + 128 * g7)] + A[(0 * 4096 + (12 * b0 + 5)) * 4096 + (t1 + 128 * g7 - 1)]) + A[(0 * 4096 + (12 * b0 + 5)) * 4096 + (t1 + 128 * g7 + 1)]) + A[(0 * 4096 + (12 * b0 + 6)) * 4096 + (t1 + 128 * g7)]) + A[(0 * 4096 + (12 * b0 + 4)) * 4096 + (t1 + 128 * g7)]));
            A[(1 * 4096 + (12 * b0 + 6)) * 4096 + (t1 + 128 * g7)] = (0.2f * ((((A[(0 * 4096 + (12 * b0 + 6)) * 4096 + (t1 + 128 * g7)] + A[(0 * 4096 + (12 * b0 + 6)) * 4096 + (t1 + 128 * g7 - 1)]) + A[(0 * 4096 + (12 * b0 + 6)) * 4096 + (t1 + 128 * g7 + 1)]) + A[(0 * 4096 + (12 * b0 + 7)) * 4096 + (t1 + 128 * g7)]) + A[(0 * 4096 + (12 * b0 + 5)) * 4096 + (t1 + 128 * g7)]));
          }
        }
        __syncthreads();
        if (t1 + 128 * g7 >= 2) {
          A[(0 * 4096 + (12 * b0 + 1)) * 4096 + (t1 + 128 * g7 - 1)] = (0.2f * ((((A[(1 * 4096 + (12 * b0 + 1)) * 4096 + (t1 + 128 * g7 - 1)] + A[(1 * 4096 + (12 * b0 + 1)) * 4096 + (t1 + 128 * g7 - 2)]) + A[(1 * 4096 + (12 * b0 + 1)) * 4096 + (t1 + 128 * g7)]) + A[(1 * 4096 + (12 * b0 + 2)) * 4096 + (t1 + 128 * g7 - 1)]) + A[(1 * 4096 + 12 * b0) * 4096 + (t1 + 128 * g7 - 1)]));
          A[(0 * 4096 + (12 * b0 + 2)) * 4096 + (t1 + 128 * g7 - 1)] = (0.2f * ((((A[(1 * 4096 + (12 * b0 + 2)) * 4096 + (t1 + 128 * g7 - 1)] + A[(1 * 4096 + (12 * b0 + 2)) * 4096 + (t1 + 128 * g7 - 2)]) + A[(1 * 4096 + (12 * b0 + 2)) * 4096 + (t1 + 128 * g7)]) + A[(1 * 4096 + (12 * b0 + 3)) * 4096 + (t1 + 128 * g7 - 1)]) + A[(1 * 4096 + (12 * b0 + 1)) * 4096 + (t1 + 128 * g7 - 1)]));
          if (b0 <= 340) {
            A[(0 * 4096 + (12 * b0 + 3)) * 4096 + (t1 + 128 * g7 - 1)] = (0.2f * ((((A[(1 * 4096 + (12 * b0 + 3)) * 4096 + (t1 + 128 * g7 - 1)] + A[(1 * 4096 + (12 * b0 + 3)) * 4096 + (t1 + 128 * g7 - 2)]) + A[(1 * 4096 + (12 * b0 + 3)) * 4096 + (t1 + 128 * g7)]) + A[(1 * 4096 + (12 * b0 + 4)) * 4096 + (t1 + 128 * g7 - 1)]) + A[(1 * 4096 + (12 * b0 + 2)) * 4096 + (t1 + 128 * g7 - 1)]));
            A[(0 * 4096 + (12 * b0 + 4)) * 4096 + (t1 + 128 * g7 - 1)] = (0.2f * ((((A[(1 * 4096 + (12 * b0 + 4)) * 4096 + (t1 + 128 * g7 - 1)] + A[(1 * 4096 + (12 * b0 + 4)) * 4096 + (t1 + 128 * g7 - 2)]) + A[(1 * 4096 + (12 * b0 + 4)) * 4096 + (t1 + 128 * g7)]) + A[(1 * 4096 + (12 * b0 + 5)) * 4096 + (t1 + 128 * g7 - 1)]) + A[(1 * 4096 + (12 * b0 + 3)) * 4096 + (t1 + 128 * g7 - 1)]));
            A[(0 * 4096 + (12 * b0 + 5)) * 4096 + (t1 + 128 * g7 - 1)] = (0.2f * ((((A[(1 * 4096 + (12 * b0 + 5)) * 4096 + (t1 + 128 * g7 - 1)] + A[(1 * 4096 + (12 * b0 + 5)) * 4096 + (t1 + 128 * g7 - 2)]) + A[(1 * 4096 + (12 * b0 + 5)) * 4096 + (t1 + 128 * g7)]) + A[(1 * 4096 + (12 * b0 + 6)) * 4096 + (t1 + 128 * g7 - 1)]) + A[(1 * 4096 + (12 * b0 + 4)) * 4096 + (t1 + 128 * g7 - 1)]));
            A[(0 * 4096 + (12 * b0 + 6)) * 4096 + (t1 + 128 * g7 - 1)] = (0.2f * ((((A[(1 * 4096 + (12 * b0 + 6)) * 4096 + (t1 + 128 * g7 - 1)] + A[(1 * 4096 + (12 * b0 + 6)) * 4096 + (t1 + 128 * g7 - 2)]) + A[(1 * 4096 + (12 * b0 + 6)) * 4096 + (t1 + 128 * g7)]) + A[(1 * 4096 + (12 * b0 + 7)) * 4096 + (t1 + 128 * g7 - 1)]) + A[(1 * 4096 + (12 * b0 + 5)) * 4096 + (t1 + 128 * g7 - 1)]));
            A[(0 * 4096 + (12 * b0 + 7)) * 4096 + (t1 + 128 * g7 - 1)] = (0.2f * ((((A[(1 * 4096 + (12 * b0 + 7)) * 4096 + (t1 + 128 * g7 - 1)] + A[(1 * 4096 + (12 * b0 + 7)) * 4096 + (t1 + 128 * g7 - 2)]) + A[(1 * 4096 + (12 * b0 + 7)) * 4096 + (t1 + 128 * g7)]) + A[(1 * 4096 + (12 * b0 + 8)) * 4096 + (t1 + 128 * g7 - 1)]) + A[(1 * 4096 + (12 * b0 + 6)) * 4096 + (t1 + 128 * g7 - 1)]));
          }
        }
        __syncthreads();
      }
      if (t1 + 128 * g7 >= 3 && t1 + 128 * g7 <= 4096) {
        A[(1 * 4096 + (12 * b0 + 1)) * 4096 + (t1 + 128 * g7 - 2)] = (0.2f * ((((A[(0 * 4096 + (12 * b0 + 1)) * 4096 + (t1 + 128 * g7 - 2)] + A[(0 * 4096 + (12 * b0 + 1)) * 4096 + (t1 + 128 * g7 - 3)]) + A[(0 * 4096 + (12 * b0 + 1)) * 4096 + (t1 + 128 * g7 - 1)]) + A[(0 * 4096 + (12 * b0 + 2)) * 4096 + (t1 + 128 * g7 - 2)]) + A[(0 * 4096 + 12 * b0) * 4096 + (t1 + 128 * g7 - 2)]));
        A[(1 * 4096 + (12 * b0 + 2)) * 4096 + (t1 + 128 * g7 - 2)] = (0.2f * ((((A[(0 * 4096 + (12 * b0 + 2)) * 4096 + (t1 + 128 * g7 - 2)] + A[(0 * 4096 + (12 * b0 + 2)) * 4096 + (t1 + 128 * g7 - 3)]) + A[(0 * 4096 + (12 * b0 + 2)) * 4096 + (t1 + 128 * g7 - 1)]) + A[(0 * 4096 + (12 * b0 + 3)) * 4096 + (t1 + 128 * g7 - 2)]) + A[(0 * 4096 + (12 * b0 + 1)) * 4096 + (t1 + 128 * g7 - 2)]));
        if (b0 <= 340) {
          A[(1 * 4096 + (12 * b0 + 3)) * 4096 + (t1 + 128 * g7 - 2)] = (0.2f * ((((A[(0 * 4096 + (12 * b0 + 3)) * 4096 + (t1 + 128 * g7 - 2)] + A[(0 * 4096 + (12 * b0 + 3)) * 4096 + (t1 + 128 * g7 - 3)]) + A[(0 * 4096 + (12 * b0 + 3)) * 4096 + (t1 + 128 * g7 - 1)]) + A[(0 * 4096 + (12 * b0 + 4)) * 4096 + (t1 + 128 * g7 - 2)]) + A[(0 * 4096 + (12 * b0 + 2)) * 4096 + (t1 + 128 * g7 - 2)]));
          A[(1 * 4096 + (12 * b0 + 4)) * 4096 + (t1 + 128 * g7 - 2)] = (0.2f * ((((A[(0 * 4096 + (12 * b0 + 4)) * 4096 + (t1 + 128 * g7 - 2)] + A[(0 * 4096 + (12 * b0 + 4)) * 4096 + (t1 + 128 * g7 - 3)]) + A[(0 * 4096 + (12 * b0 + 4)) * 4096 + (t1 + 128 * g7 - 1)]) + A[(0 * 4096 + (12 * b0 + 5)) * 4096 + (t1 + 128 * g7 - 2)]) + A[(0 * 4096 + (12 * b0 + 3)) * 4096 + (t1 + 128 * g7 - 2)]));
          A[(1 * 4096 + (12 * b0 + 5)) * 4096 + (t1 + 128 * g7 - 2)] = (0.2f * ((((A[(0 * 4096 + (12 * b0 + 5)) * 4096 + (t1 + 128 * g7 - 2)] + A[(0 * 4096 + (12 * b0 + 5)) * 4096 + (t1 + 128 * g7 - 3)]) + A[(0 * 4096 + (12 * b0 + 5)) * 4096 + (t1 + 128 * g7 - 1)]) + A[(0 * 4096 + (12 * b0 + 6)) * 4096 + (t1 + 128 * g7 - 2)]) + A[(0 * 4096 + (12 * b0 + 4)) * 4096 + (t1 + 128 * g7 - 2)]));
          A[(1 * 4096 + (12 * b0 + 6)) * 4096 + (t1 + 128 * g7 - 2)] = (0.2f * ((((A[(0 * 4096 + (12 * b0 + 6)) * 4096 + (t1 + 128 * g7 - 2)] + A[(0 * 4096 + (12 * b0 + 6)) * 4096 + (t1 + 128 * g7 - 3)]) + A[(0 * 4096 + (12 * b0 + 6)) * 4096 + (t1 + 128 * g7 - 1)]) + A[(0 * 4096 + (12 * b0 + 7)) * 4096 + (t1 + 128 * g7 - 2)]) + A[(0 * 4096 + (12 * b0 + 5)) * 4096 + (t1 + 128 * g7 - 2)]));
          A[(1 * 4096 + (12 * b0 + 7)) * 4096 + (t1 + 128 * g7 - 2)] = (0.2f * ((((A[(0 * 4096 + (12 * b0 + 7)) * 4096 + (t1 + 128 * g7 - 2)] + A[(0 * 4096 + (12 * b0 + 7)) * 4096 + (t1 + 128 * g7 - 3)]) + A[(0 * 4096 + (12 * b0 + 7)) * 4096 + (t1 + 128 * g7 - 1)]) + A[(0 * 4096 + (12 * b0 + 8)) * 4096 + (t1 + 128 * g7 - 2)]) + A[(0 * 4096 + (12 * b0 + 6)) * 4096 + (t1 + 128 * g7 - 2)]));
        }
      }
      __syncthreads();
      if (t1 + 128 * g7 >= 4 && t1 + 128 * g7 <= 4097) {
        A[(0 * 4096 + (12 * b0 + 2)) * 4096 + (t1 + 128 * g7 - 3)] = (0.2f * ((((A[(1 * 4096 + (12 * b0 + 2)) * 4096 + (t1 + 128 * g7 - 3)] + A[(1 * 4096 + (12 * b0 + 2)) * 4096 + (t1 + 128 * g7 - 4)]) + A[(1 * 4096 + (12 * b0 + 2)) * 4096 + (t1 + 128 * g7 - 2)]) + A[(1 * 4096 + (12 * b0 + 3)) * 4096 + (t1 + 128 * g7 - 3)]) + A[(1 * 4096 + (12 * b0 + 1)) * 4096 + (t1 + 128 * g7 - 3)]));
        if (b0 <= 340) {
          A[(0 * 4096 + (12 * b0 + 3)) * 4096 + (t1 + 128 * g7 - 3)] = (0.2f * ((((A[(1 * 4096 + (12 * b0 + 3)) * 4096 + (t1 + 128 * g7 - 3)] + A[(1 * 4096 + (12 * b0 + 3)) * 4096 + (t1 + 128 * g7 - 4)]) + A[(1 * 4096 + (12 * b0 + 3)) * 4096 + (t1 + 128 * g7 - 2)]) + A[(1 * 4096 + (12 * b0 + 4)) * 4096 + (t1 + 128 * g7 - 3)]) + A[(1 * 4096 + (12 * b0 + 2)) * 4096 + (t1 + 128 * g7 - 3)]));
          A[(0 * 4096 + (12 * b0 + 4)) * 4096 + (t1 + 128 * g7 - 3)] = (0.2f * ((((A[(1 * 4096 + (12 * b0 + 4)) * 4096 + (t1 + 128 * g7 - 3)] + A[(1 * 4096 + (12 * b0 + 4)) * 4096 + (t1 + 128 * g7 - 4)]) + A[(1 * 4096 + (12 * b0 + 4)) * 4096 + (t1 + 128 * g7 - 2)]) + A[(1 * 4096 + (12 * b0 + 5)) * 4096 + (t1 + 128 * g7 - 3)]) + A[(1 * 4096 + (12 * b0 + 3)) * 4096 + (t1 + 128 * g7 - 3)]));
          A[(0 * 4096 + (12 * b0 + 5)) * 4096 + (t1 + 128 * g7 - 3)] = (0.2f * ((((A[(1 * 4096 + (12 * b0 + 5)) * 4096 + (t1 + 128 * g7 - 3)] + A[(1 * 4096 + (12 * b0 + 5)) * 4096 + (t1 + 128 * g7 - 4)]) + A[(1 * 4096 + (12 * b0 + 5)) * 4096 + (t1 + 128 * g7 - 2)]) + A[(1 * 4096 + (12 * b0 + 6)) * 4096 + (t1 + 128 * g7 - 3)]) + A[(1 * 4096 + (12 * b0 + 4)) * 4096 + (t1 + 128 * g7 - 3)]));
          A[(0 * 4096 + (12 * b0 + 6)) * 4096 + (t1 + 128 * g7 - 3)] = (0.2f * ((((A[(1 * 4096 + (12 * b0 + 6)) * 4096 + (t1 + 128 * g7 - 3)] + A[(1 * 4096 + (12 * b0 + 6)) * 4096 + (t1 + 128 * g7 - 4)]) + A[(1 * 4096 + (12 * b0 + 6)) * 4096 + (t1 + 128 * g7 - 2)]) + A[(1 * 4096 + (12 * b0 + 7)) * 4096 + (t1 + 128 * g7 - 3)]) + A[(1 * 4096 + (12 * b0 + 5)) * 4096 + (t1 + 128 * g7 - 3)]));
        }
      }
      __syncthreads();
    }
}