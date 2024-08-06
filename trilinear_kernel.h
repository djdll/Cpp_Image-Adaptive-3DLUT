#ifndef _TRILINEAR_KERNEL
#define _TRILINEAR_KERNEL

#include <THC/THC.h>

__global__ void TriLinearForward(const int nthreads, const float* lut, const float* image, float* output, const int dim, const int shift, const float binsize, const int width, const int height, const int batch);

int TriLinearForwardLaucher(const float* lut, const float* image, float* output, const int lut_dim, const int shift, const float binsize, const int width, const int height, const int batch, cudaStream_t stream);


#endif

