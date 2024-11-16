#ifndef LIUXUAN_RFFT_2D_PLUGIN_H
#define LIUXUAN_RFFT_2D_PLUGIN_H

#include <cufft.h>
#include <cuda.h>
#include <vector>
#include <string>

#include <NvInfer.h>
namespace nvinfer1
{
namespace plugin
{
class RFFT2D: public nvinfer1::IPluginV2DynamicExt
{
public:
    RFFT2D(const std::string name);
    ~RFFT2D() override = default;

    nvinfer1::DimsExprs getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs* inputs,
        int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept override;
    size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
        const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept override;
    int enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
        const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;
    IPluginV2DynamicExt* clone() const noexcept override;

    void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
        const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept override;
    bool supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept override;
    void attachToContext(cudnnContext* cudnnContext, cublasContext* cublasContext, nvinfer1::IGpuAllocator* gpuAllocator) noexcept override;
    nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept override;

    const char* getPluginType() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    const char* getPluginNamespace() const noexcept override;
    void terminate() noexcept override;
    int initialize() noexcept override;
    void serialize(void* buffer) const noexcept override;
    size_t getSerializationSize() const noexcept override;
    void destroy() noexcept override;
    void detachFromContext() noexcept override;
    void setPluginNamespace(const char* pluginNamespace) noexcept override;
    int getNbOutputs() const noexcept override { return 1; }

private:
    std::string layer_name_;

    std::string mPluginNamespace;
    std::string mNamespace;
    cufftHandle handle_;
};


class RFFT2DCreator : public nvinfer1::IPluginCreator
{
public:
    RFFT2DCreator();
    ~RFFT2DCreator() override = default;

    const char* getPluginName() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override;
    nvinfer1::IPluginV2DynamicExt* createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) noexcept override;
    nvinfer1::IPluginV2DynamicExt* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override;
    void setPluginNamespace(const char* libNamespace) noexcept override {
        mNamespace = libNamespace;
    }

    const char* getPluginNamespace() const noexcept override {
        return mNamespace.c_str();
    }

private:
    static nvinfer1::PluginFieldCollection mFC;
    static std::vector<nvinfer1::PluginField> mPluginAttributes;
    std::string mNamespace;
    std::string mPluginName;
};
} // namespace plugin
} // namespace nvinfer1

#endif // WS_RFFT_2D_PLUGIN_H