#include "rfft2d.h"

#include <cufft.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cassert>
#include <cstring>
#include <iostream>

//#define CHECK_CURFFT(call) do{cufftResult_t status_ = call;if( status_ != CUFFT_SUCCESS ){fprintf(stderr, "CUFFT Error at line %d: ", __LINE__, status_);exit(1);}} while(0)

using nvinfer1::plugin::RFFT2D;
using nvinfer1::plugin::RFFT2DCreator;

namespace {
    const char* RFFT2D_PLUGIN_VERSION{"1"};
    const char* RFFT2D_PLUGIN_NAME{"rfft2d"};
} // namespace

nvinfer1::PluginFieldCollection RFFT2DCreator::mFC{};
std::vector<nvinfer1::PluginField> RFFT2DCreator::mPluginAttributes;

RFFT2D::RFFT2D(const std::string name) : handle_(NULL), layer_name_(name) {
}

void RFFT2D::serialize(void* buffer) const noexcept {
}

size_t RFFT2D::getSerializationSize() const noexcept {
    return 0;
}

int RFFT2D::initialize() noexcept {
    return 0;
}

void RFFT2D::terminate() noexcept {
}

nvinfer1::DimsExprs RFFT2D::getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs* inputs,
    int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept {
    nvinfer1::DimsExprs ret;
    ret.nbDims = inputs[0].nbDims + 1;
    for (int i = 0; i < inputs[0].nbDims; i++) {
        ret.d[i] = inputs[0].d[i];
    }

    const auto *con_two = exprBuilder.constant(2);
    const auto *con_one = exprBuilder.constant(1);
    ret.d[ret.nbDims - 1] = con_two;
    ret.d[ret.nbDims - 2] = exprBuilder.operation(nvinfer1::DimensionOperation::kSUM,
        *exprBuilder.operation(nvinfer1::DimensionOperation::kFLOOR_DIV, 
        *inputs[0].d[inputs[0].nbDims - 1], *con_two), *con_one);

    return ret;
}

void RFFT2D::setPluginNamespace(const char* pluginNamespace) noexcept {
    mPluginNamespace = pluginNamespace;
}

const char* RFFT2D::getPluginNamespace() const noexcept {
    return mPluginNamespace.c_str();
}

nvinfer1::DataType RFFT2D::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept {
    return nvinfer1::DataType::kFLOAT;
}

bool RFFT2D::supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept {
    assert(nbInputs == 1 && nbOutputs == 1 && pos < nbInputs + nbOutputs);
    if (pos == 0) {
        return (inOut[pos].type == nvinfer1::DataType::kFLOAT && inOut[pos].format == nvinfer1::TensorFormat::kLINEAR);
    } else {
        return inOut[pos].type == inOut[0].type && inOut[pos].format == inOut[0].format;
    }

    return true;
}

void RFFT2D::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept {
    assert(in && nbInputs == 1);
    assert(out && nbOutputs == 1); 
}

size_t RFFT2D::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept {

    int nb_dim = inputs[0].dims.nbDims;
    size_t workspace = inputs[0].dims.d[nb_dim - 1] * inputs[0].dims.d[nb_dim - 2];
    return workspace;
}

void RFFT2D::attachToContext(cudnnContext* cudnnContext, cublasContext* cublasContext, nvinfer1::IGpuAllocator* gpuAllocator) noexcept {
}

void RFFT2D::detachFromContext() noexcept {
}

const char* RFFT2D::getPluginType() const noexcept {
    return RFFT2D_PLUGIN_NAME;
}

const char* RFFT2D::getPluginVersion() const noexcept {
    return RFFT2D_PLUGIN_VERSION;
}

void RFFT2D::destroy() noexcept {
    delete this;
}

nvinfer1::IPluginV2DynamicExt* RFFT2D::clone() const noexcept {
    RFFT2D *obj = new RFFT2D(layer_name_);
    obj->setPluginNamespace(mPluginNamespace.c_str());

    return obj;
}

RFFT2DCreator::RFFT2DCreator() {
}

const char* RFFT2DCreator::getPluginName() const noexcept {
    return RFFT2D_PLUGIN_NAME;
}

const char* RFFT2DCreator::getPluginVersion() const noexcept {
    return RFFT2D_PLUGIN_VERSION;
}

const nvinfer1::PluginFieldCollection* RFFT2DCreator::getFieldNames() noexcept {
    return &mFC;
}

nvinfer1::IPluginV2DynamicExt* RFFT2DCreator::createPlugin(const char* name, 
    const nvinfer1::PluginFieldCollection* fc) noexcept {
    RFFT2D *obj = new RFFT2D(name);
    obj->setPluginNamespace(mNamespace.c_str());

    return obj;
}

nvinfer1::IPluginV2DynamicExt* RFFT2DCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept {    
    RFFT2D* obj = new RFFT2D(name);
    obj->setPluginNamespace(mNamespace.c_str());

    return obj;
}