#ifndef MNN_NETWORK_H_
#define MNN_NETWORK_H_
#include <vector>

#include "magic_xe_blob.h"
#include "magic_xe_json.h"
#include "mnn_include.h"
#include "plugin/magic_xe_plugin.h"
#include "utils/magic_xe_json_utils.h"
namespace magic_xe {
namespace mnn {
class MnnNetwork {
public:
    MnnNetwork();
    ~MnnNetwork();

    MagicXEError Init(const MagicXEModel model, const MagicXECustomRuntimeV2 *runtime);
    void DeInit();

    MagicXEError Reshape(const char **name_arr, const MagicXEDims *dims_arr, size_t dims_size);
    MagicXEError Forward();
    MagicXEError InputBlobsGet(const MagicXEBlob ***blob_arr, size_t *blob_size);
    MagicXEError OutputBlobsGet(const MagicXEBlob ***blob_arr, size_t *blob_size);
    MagicXEError InputBlobGet(const char *input_name, MagicXEBlob **blob);
    MagicXEError OutputBlobGet(const char *output_name, const MagicXEBlob **blob);

    MagicXEError AddOutput(const std::string &output_name);

private:
    MagicXEError Reshape();
    MagicXEError InitWithJson(
        MagicXEModel model, const MagicXECustomRuntimeV2 *runtime, const MagicXEJsonHandle json_handle);

    MagicXEError ParseInputShapes(const MagicXEJsonHandle json_handle);

    MagicXEError CreateBlobArray();
    MagicXEError CreateInferRequest();
    void ClearBlobArray();

private:
    MNN::Interpreter *interpreter_ = nullptr;
    MNN::Session *session_ = nullptr;
    MagicXEDeviceTypeV2 device_type_;

    std::map<std::string, MagicXEDims> input_max_shapes_;
    std::map<std::string, MagicXEDims> input_min_shapes_;

    MagicXEBlob **input_blob_arr_ = nullptr;
    size_t input_blob_size_ = 0;

    MagicXEBlob **output_blob_arr_ = nullptr;
    size_t output_blob_size_ = 0;

    std::vector<std::string> save_tensors_;

    bool gpu_blob_ = false;
};

} // namespace mnn

} // namespace magic_xe

#endif //MNN_NETWORK_H_
