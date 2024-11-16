#include "rfft2d.h"
#include <math.h>
#define MAX_THREAD 512

__global__ void array_sqrt(float *arr, int ninputs, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < ninputs) {
        arr[i] /= N;
    }
}

using nvinfer1::plugin::RFFT2D;

int RFFT2D::enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc, 
    const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept {
    int dsize = inputDesc[0].dims.nbDims;
    int in_c = inputDesc[0].dims.d[dsize - 3];
    int in_h = inputDesc[0].dims.d[dsize - 2];
    int in_w = inputDesc[0].dims.d[dsize - 1];
    int out_w = in_w / 2 + 1;
    int out_cnt = out_w * in_h * 2;

    int nThreads = MAX_THREAD;
    int ngrid = (int)((float)out_cnt / nThreads) + 1;

    cufftPlan2d(&handle_, in_h, in_w, CUFFT_R2C);

    float *input_arr;
    float *real = (float *)inputs[0];
    cufftComplex *comp = (cufftComplex *)outputs[0];
    cudaMalloc(&input_arr, sizeof(float) * in_h * in_w);
    int in_size = in_h * out_w * 2;
    int N = std::sqrt(in_h * in_w);
    for (size_t i = 0; i < in_c; i++) {
        // solve 64bit alignment
        cudaMemcpy(input_arr, real, sizeof(float) * in_h * in_w, cudaMemcpyDeviceToDevice);

        cufftExecR2C(handle_, input_arr, comp);
        cudaStreamSynchronize(stream);
        array_sqrt<<<ngrid, nThreads, 0, stream>>>((float *)comp, in_size, N);
        cudaStreamSynchronize(stream);

        real += in_h * in_w;
        comp += in_h * out_w;
    }

    cufftDestroy(handle_);
    cudaFree(input_arr);
    return 0;
}