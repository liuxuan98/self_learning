#include <math.h>
#include "irfft2d.h"
#define MAX_THREAD 512

__global__ void array_divide(float *arr, int ninputs, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < ninputs) {
        arr[i] /= N;
    }
}

using nvinfer1::plugin::IRFFT2D;

int IRFFT2D::enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept {
    int dsize = inputDesc[0].dims.nbDims;
    int dsize2 = inputDesc[1].dims.nbDims;

    int in_h = inputDesc[0].dims.d[dsize - 3];
    int in_w = inputDesc[0].dims.d[dsize - 2];
    int in_c = inputDesc[0].dims.d[dsize - 4];
    int out_w = inputDesc[1].dims.d[dsize2 - 1];
    int out_cnt = in_h * in_w;
    cufftPlan2d(&handle_, in_h, out_w, CUFFT_C2R);

    int nThreads = MAX_THREAD;
    int ngrid = (int)((float)in_h * out_w / nThreads) + 1;


    cufftComplex *comp = (cufftComplex *)inputs[0];
    float *real = (float *)outputs[0];
    cufftComplex *tmp_in;
    cudaMalloc(&tmp_in, sizeof(cufftComplex) * in_h * in_w);
    int out_size = std::sqrt(in_h * out_w);
    for (int i = 0; i < in_c; i++) {
        // solve 64bit alignment
        cudaMemcpy(tmp_in, comp, sizeof(cufftComplex) * in_h * in_w, cudaMemcpyDeviceToDevice);

        cufftExecC2R(handle_, tmp_in, real);
        cudaStreamSynchronize(stream);
        array_divide<<<ngrid, nThreads, 0, stream>>>((float *)real, in_h * out_w, out_size);
        cudaStreamSynchronize(stream);
        comp += in_h * in_w;
        real += in_h * out_w;
    }
    cudaFree(tmp_in);
    cufftDestroy(handle_);

    return 0;
}
