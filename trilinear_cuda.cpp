#include "trilinear_kernel.h"
#include <torch/extension.h>
#include <THC/THC.h>
#include <math.h>
#include <float.h>
//#include "cuda_runtime.h"
int trilinear_forward_cuda(torch::Tensor lut, torch::Tensor image, torch::Tensor output, int lut_dim, int shift, float binsize, int width, int height, int batch)
{
    // Grab the input tensor
    float * lut_flat = lut.data_ptr<float>();
    float * image_flat = image.data_ptr<float>();
    float * output_flat = output.data_ptr<float>();

    TriLinearForwardLaucher(lut_flat, image_flat, output_flat, lut_dim, shift, binsize, width, height, batch, at::cuda::getCurrentCUDAStream());

    return 1;
}


