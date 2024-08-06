#ifndef TRILINEAR_CUDA_H
#define TRILINEAR_CUDA_H

#include <torch/extension.h>

int trilinear_forward_cuda(torch::Tensor lut, torch::Tensor image, torch::Tensor output,
                           int lut_dim, int shift, float binsize, int width, int height, int batch);

#endif

