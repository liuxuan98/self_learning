#include "irfft2d.h"

#include <cufft.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cassert>
#include <cstring>
#include <iostream>

//#define CHECK_CURFFT(call) do{cufftResult_t status_ = call;if( status_ != CUFFT_SUCCESS ){fprintf(stderr, "CUFFT Error at line %d: ", __LINE__, status_);exit(1);}} while(0)

using nvinfer1::plugin::IRFFT2D;
using nvinfer1::plugin::IRFFT2DCreator;

namespace {
    const char* IRFFT2D_PLUGIN_VERSION{"1"};
    const char* IRFFT2D_PLUGIN_NAME{"irfft2d"};
} // namespace

nvinfer1::PluginFieldCollection IRFFT2DCreator::mFC{};
std::vector<nvinfer1::PluginField> IRFFT2DCreator::mPluginAttributes;

IRFFT2D::IRFFT2D(const std::string name) : handle_(NULL), layer_name_(name) {
}

void IRFFT2D::serialize(void* buffer) const noexcept {
}

size_t IRFFT2D::getSerializationSize() const noexcept {
    return 0;
}

int IRFFT2D::initialize() noexcept {
    return 0;
}

void IRFFT2D::terminate() noexcept {
}

nvinfer1::DimsExprs IRFFT2D::getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs* inputs,
    int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept {
    nvinfer1::DimsExprs ret(inputs[1]);
    return ret;
}

void IRFFT2D::setPluginNamespace(const char* pluginNamespace) noexcept {
    mPluginNamespace = pluginNamespace;
}

const char* IRFFT2D::getPluginNamespace() const noexcept {
    return mPluginNamespace.c_str();
}

nvinfer1::DataType IRFFT2D::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept {
    return nvinfer1::DataType::kFLOAT;
}

bool IRFFT2D::supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept {
    assert(nbInputs == 2 && nbOutputs == 1 && pos < nbInputs + nbOutputs);
    if (pos == 0 || pos == 1) {
        return (inOut[pos].type == nvinfer1::DataType::kFLOAT && inOut[pos].format == nvinfer1::TensorFormat::kLINEAR);
    } else {
        return inOut[pos].type == inOut[0].type && inOut[pos].format == inOut[0].format;
    }

    return true;
}

void IRFFT2D::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept {
    assert(in && nbInputs == 2);
    assert(out && nbOutputs == 1); 
}

size_t IRFFT2D::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept {
    int nb_dim = inputs[0].dims.nbDims;
    size_t workspace = inputs[0].dims.d[nb_dim - 2] * inputs[0].dims.d[nb_dim - 3] * 2;
    return workspace;
}

void IRFFT2D::attachToContext(cudnnContext* cudnnContext, cublasContext* cublasContext, nvinfer1::IGpuAllocator* gpuAllocator) noexcept {
}

void IRFFT2D::detachFromContext() noexcept {
}

const char* IRFFT2D::getPluginType() const noexcept {
    return IRFFT2D_PLUGIN_NAME;
}

const char* IRFFT2D::getPluginVersion() const noexcept {
    return IRFFT2D_PLUGIN_VERSION;
}

void IRFFT2D::destroy() noexcept {
    delete this;
}

nvinfer1::IPluginV2DynamicExt* IRFFT2D::clone() const noexcept {
    IRFFT2D *obj = new IRFFT2D(layer_name_);
    obj->setPluginNamespace(mPluginNamespace.c_str());

    return obj;
}

IRFFT2DCreator::IRFFT2DCreator() {
}

const char* IRFFT2DCreator::getPluginName() const noexcept {
    return IRFFT2D_PLUGIN_NAME;
}

const char* IRFFT2DCreator::getPluginVersion() const noexcept {
    return IRFFT2D_PLUGIN_VERSION;
}

const nvinfer1::PluginFieldCollection* IRFFT2DCreator::getFieldNames() noexcept {
    return &mFC;
}

nvinfer1::IPluginV2DynamicExt* IRFFT2DCreator::createPlugin(const char* name, 
    const nvinfer1::PluginFieldCollection* fc) noexcept {
    IRFFT2D *obj = new IRFFT2D(name);
    obj->setPluginNamespace(mNamespace.c_str());

    return obj;
}

nvinfer1::IPluginV2DynamicExt* IRFFT2DCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept {    
    IRFFT2D* obj = new IRFFT2D(name);
    obj->setPluginNamespace(mNamespace.c_str());

    return obj;
}