#include "ClampingPluginKernel.h"

__global__ void clampingKernel(int nthreads, const float* input, float* output)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i > nthreads)
        return;

    output[i] = input[i] > 0.000001 ? input[i] : 0.000001;
}

cudaError_t clamping(int num_of_elements, const float* input, float* output, cudaStream_t*  stream)
{
    if(!input || !output)
        return cudaErrorInvalidDevicePointer;

    if(num_of_elements <= 0)
        return cudaErrorInvalidValue;

    const dim3 _gridDim((num_of_elements - 1) / 512 + 1, 1, 1);
    clampingKernel<<<_gridDim,512, 0, *stream>>>(num_of_elements, input, output);

    return CUDA(cudaGetLastError());
}
