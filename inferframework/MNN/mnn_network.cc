#include "mnn_network.h"

#include "magic_xe_blob.h"
#include "magic_xe_buffer_v2.h"
#include "magic_xe_env.h"
#include "magic_xe_mnn_version.h"
#include "mnn_blob_converter.h"
#include "mnn_config_converter.h"
#include "plugin/magic_xe_plugin_template.h"
#include "utils/magic_xe_blob_utils.h"
#include "utils/magic_xe_file_utils.h"
#include "utils/magic_xe_type_utils.h"

MAGIC_XE_NEURAL_NET_STRUCT_TO_CLASS(MnnNetwork, mnn, MAGIC_XE_MODEL_TYPE_MNN);

namespace magic_xe {
namespace mnn {

MnnNetwork::MnnNetwork() {
}

MnnNetwork::~MnnNetwork() {

    ClearBlobArray();
    if (interpreter_ != nullptr) {
        delete interpreter_;
        interpreter_ = nullptr;
    }
}

void MnnNetwork::ClearBlobArray() {
    if (input_blob_arr_ != nullptr && input_blob_size_ != 0) {
        for (size_t i = 0; i < input_blob_size_; ++i) {
            if (input_blob_arr_[i]) {
                MagicXEBlobFree(input_blob_arr_[i]);
            }
        }
        free(input_blob_arr_);
        input_blob_arr_ = nullptr;
        input_blob_size_ = 0;
    }

    if (output_blob_arr_ != nullptr && output_blob_size_ != 0) {
        for (size_t i = 0; i < output_blob_size_; ++i) {
            if (output_blob_arr_[i]) {
                MagicXEBlobFree(output_blob_arr_[i]);
            }
        }
        free(output_blob_arr_);
        output_blob_arr_ = nullptr;
        output_blob_size_ = 0;
    }
}

MagicXEError MnnNetwork::Init(const MagicXEModel model, const MagicXECustomRuntimeV2 *runtime) {

    void *content = NULL;
    size_t content_size = 0;
    MagicXEError mgxe_err = MAGIC_XE_SUCCESS;
    mgxe_err = MagicXEModelFileContentGet(model, "inference.json", &content, &content_size);
    if (mgxe_err != MAGIC_XE_SUCCESS) {
        MAGIC_XE_LOGE("GetFileContent inference.json failed:%d", mgxe_err);
        return mgxe_err;
    }
    MagicXEJsonHandle json_handle = nullptr;
    if ((mgxe_err = MagicXEJsonCreate(nullptr, content, content_size, &json_handle)) != MAGIC_XE_SUCCESS) {
        MAGIC_XE_LOGE("neural_network.json MagicXEJsonUtilsCreate failed:%d", mgxe_err);
        goto __end;
    }
    if ((mgxe_err = ParseInputShapes(json_handle)) != MAGIC_XE_SUCCESS) {
        MAGIC_XE_LOGE("ParseInputShapes failed:%d", mgxe_err);
        goto __end;
    }
    if ((mgxe_err = InitWithJson(model, runtime, json_handle)) != MAGIC_XE_SUCCESS) {
        MAGIC_XE_LOGE("Init failed:%d", mgxe_err);
        goto __end;
    }

    if ((mgxe_err = Reshape()) != MAGIC_XE_SUCCESS) {
        MAGIC_XE_LOGE("CreateInferRequest failed:%d", mgxe_err);
        goto __end;
    }

    device_type_ = runtime->device_type;
    if (runtime->device_type == MAGIC_XE_DEVICE_TYPE_OPENCL) {
        interpreter_->updateCacheFile(session_);
    }

    if ((mgxe_err = CreateBlobArray()) != MAGIC_XE_SUCCESS) {
        MAGIC_XE_LOGE("Reshape failed:%d", mgxe_err);
        goto __end;
    }

__end:
    if (json_handle != nullptr) {
        MagicXEJsonDestory(&json_handle);
    }
    return mgxe_err;
}

void MnnNetwork::DeInit() {
}

MagicXEError MnnNetwork::InitWithJson(
    MagicXEModel model, const MagicXECustomRuntimeV2 *runtime, const MagicXEJsonHandle json_handle) {
    void *data = nullptr;
    size_t data_len = 0;
    MagicXEError mgxe_err = MagicXEModelFileContentGet(model, ".mnn", &data, &data_len);
    if (mgxe_err != MAGIC_XE_SUCCESS) {
        MAGIC_XE_LOGE("MagicXEModelGetFileContentBySuffix mnn failed:%d", mgxe_err);
        return MAGIC_XE_INVALID_PARAM;
    }

    interpreter_ = (MNN::Interpreter::createFromBuffer(data, data_len));
    if (interpreter_ == nullptr) {
        MAGIC_XE_LOGE("Init\n");
        return MAGIC_XE_INVALID_MODEL;
    }

    MagicXEJsonObject root_obj = MagicXEJsonRootGet(json_handle);
    MagicXEJsonObject network_obj = MagicXEJsonObjectGet(root_obj, "NetworkConfig");

    MagicXEDeviceTypeV2 device_type =
        runtime ? runtime->device_type :
                  MagicXEStringToDeviceType(MagicXEJsonStringGet(MagicXEJsonObjectGet(network_obj, "DeviceType")));
    //MagicXEDeviceTypeV2 device_type = MAGIC_XE_DEVICE_TYPE_X86;
    int num_threads =
        runtime ? runtime->num_thread : MagicXEJsonIntGet(MagicXEJsonObjectGet(network_obj, "ThreadNum"), 0);
    MagicXEPrecisionV2 precision =
        MagicXEStringToPrecision(MagicXEJsonStringGet(MagicXEJsonObjectGet(network_obj, "Precision")));
    if (device_type == MAGIC_XE_DEVICE_TYPE_OPENCL) {
        std::string mnn_cache_path = "";
#ifdef __APPLE__
        mnn_cache_path = getenv("TMPDIR");
#else
        mnn_cache_path = MagicXEFileUtilsGetExeDir();
#endif
        std::string cache_path = MagicXEJsonStringGet(MagicXEJsonObjectGet(network_obj, "CachePath"));
        mnn_cache_path += "/" + cache_path;

        interpreter_->setCacheFile(mnn_cache_path.c_str());
    }

    MNN::ScheduleConfig config;
    mgxe_err = MnnConfigConverter::ConvertFromNetworkConfig(config, device_type, num_threads, precision);
    if (mgxe_err != MAGIC_XE_SUCCESS) {
        MAGIC_XE_LOGE("MnnConfigConverter::ConvertFromNetworkConfig failed:%d.");
        return mgxe_err;
    }
    config.saveTensors = save_tensors_;
    if (device_type == MAGIC_XE_DEVICE_TYPE_OPENCL) {
        config.mode = MNN_GPU_TUNING_WIDE | MNN_GPU_MEMORY_IMAGE;
    }
    session_ = interpreter_->createSession(config);
    if (session_ == nullptr) {
        MAGIC_XE_LOGE("createSession Failed");
        return MAGIC_XE_INVALID_MODEL;
    }
    if (config.backendConfig != nullptr) {
        delete config.backendConfig;
        config.backendConfig = nullptr;
    }

    gpu_blob_ = runtime->use_gpu;

    return MAGIC_XE_SUCCESS;
}

MagicXEError MnnNetwork::ParseInputShapes(const MagicXEJsonHandle json_handle) {
    MagicXEError ret = MAGIC_XE_SUCCESS;

    MagicXEJsonObject root_obj = MagicXEJsonRootGet(json_handle);
    MagicXEJsonObject model_obj = MagicXEJsonObjectGet(root_obj, "ModelConfig");
    if (model_obj == NULL) {
        MAGIC_XE_LOGE("NetworkConfig is empty!");
        return MAGIC_XE_INVALID_PARAM;
    }
    MagicXEJsonObject max_shapes_arr = MagicXEJsonObjectGet(model_obj, "MaxShapes");
    if (max_shapes_arr == NULL) {
        return MAGIC_XE_SUCCESS;
    }

    unsigned int size = MagicXEJsonArraySize(max_shapes_arr);
    for (unsigned int i = 0; i < size; ++i) {
        MagicXEJsonObject shape_obj = MagicXEJsonArrayAt(max_shapes_arr, i);
        MagicXEJsonObject name_obj = MagicXEJsonObjectGet(shape_obj, "name");
        const char *name = MagicXEJsonStringGet(name_obj);
        if (name == NULL || strlen(name) <= 0 || strlen(name) > MAX_BLOB_NAME) {
            MAGIC_XE_LOGE("InputMaxShape Name is empty or > MAX_DIMS_NAME:%d!", MAX_BLOB_NAME);
            ret = MAGIC_XE_INVALID_MODEL;
            goto __end;
        }
        MagicXEJsonObject dims_arr = MagicXEJsonObjectGet(shape_obj, "dims");
        if (dims_arr == NULL) {
            MAGIC_XE_LOGE("InputMaxShape Dims is Error!");
            ret = MAGIC_XE_INVALID_MODEL;
            goto __end;
        }
        unsigned int dims_size = MagicXEJsonArraySize(dims_arr);
        if (dims_size <= 0 || dims_size > MAX_DIMS_SIZE) {
            MAGIC_XE_LOGE("InputMaxShape Dims size:%d is empty or > MAX_DIMS_SIZE:%d!", dims_size, MAX_DIMS_SIZE);
            ret = MAGIC_XE_INVALID_MODEL;
            goto __end;
        }

        MagicXEDims dims;
        dims.size = dims_size;

        for (int j = 0; j < dims_size; ++j) {
            dims.value[j] = MagicXEJsonIntGet(MagicXEJsonArrayAt(dims_arr, j), 0);
        }

        input_max_shapes_[name] = dims;
    }

__end:
    return MAGIC_XE_SUCCESS;
}

MagicXEError MnnNetwork::CreateBlobArray() {
    MagicXEError ret = MAGIC_XE_SUCCESS;
    ClearBlobArray();

    if (nullptr == interpreter_) {
        MAGIC_XE_LOGE("MNN:Interpreter is nullptr");
        return MAGIC_XE_INVALID_MODEL;
    }
    std::map<std::string, MNN::Tensor *> inputs = interpreter_->getSessionInputAll(session_);
    int input_count = (int)inputs.size();
    if (input_count <= 0) {
        MAGIC_XE_LOGE("getSessionInputAll size:%d failed", input_count);
        return MAGIC_XE_INVALID_MODEL;
    }

    input_blob_arr_ = (MagicXEBlob **)malloc(sizeof(MagicXEBlob *) * input_count);
    if (input_blob_arr_ == nullptr) {
        MAGIC_XE_LOGE("Input blobs malloc MagicXEBlob:%d * size:%zu failed", sizeof(MagicXEBlob), input_count);
        return MAGIC_XE_OUT_OF_MEMORY;
    }
    memset(input_blob_arr_, 0, sizeof(MagicXEBlob *) * input_count);

    for (auto input : inputs) {
        const char *name = input.first.c_str();
        if (name == NULL || strlen(name) <= 0 || strlen(name) > MAX_BLOB_NAME) {
            MAGIC_XE_LOGE("GetInputName:%s len:%d failed", name != NULL ? name : "", name != NULL ? strlen(name) : 0);
            ClearBlobArray();
            return MAGIC_XE_INVALID_MODEL;
        }

        MNN::Tensor *tensor = input.second;
        ret =
            MnnBlobConverter::CreateOrUpdateBlob(&input_blob_arr_[input_blob_size_++], *tensor, name, true, gpu_blob_);
        if (ret != MAGIC_XE_SUCCESS) {
            MAGIC_XE_LOGE("MnnBlobConverter::CreateOrUpdateBlob failed:%d.", ret);
            ClearBlobArray();
            return ret;
        }
    }

    std::map<std::string, MNN::Tensor *> outputs = interpreter_->getSessionOutputAll(session_);
    int output_count = (int)outputs.size();
    if (output_count <= 0) {
        MAGIC_XE_LOGE("getSessionOutputAll size:%d failed", output_count);
        ClearBlobArray();
        return MAGIC_XE_INVALID_MODEL;
    }

    output_blob_arr_ = (MagicXEBlob **)malloc(sizeof(MagicXEBlob *) * output_count);
    if (output_blob_arr_ == nullptr) {
        MAGIC_XE_LOGE("Output blobs malloc MagicXEBlob:%d * size:%zu failed", sizeof(MagicXEBlob), output_count);
        ClearBlobArray();
        return MAGIC_XE_OUT_OF_MEMORY;
    }
    memset(output_blob_arr_, 0, sizeof(MagicXEBlob *) * output_count);
    bool alloc = input_blob_arr_[0]->device_type != MAGIC_XE_DEVICE_TYPE_OPENCL;
    for (auto output : outputs) {
        const char *name = output.first.c_str();
        if (name == NULL || strlen(name) <= 0 || strlen(name) > MAX_BLOB_NAME) {
            MAGIC_XE_LOGE("GetOutputName:%s len:%d failed", name != NULL ? name : "", name != NULL ? strlen(name) : 0);
            return MAGIC_XE_INVALID_MODEL;
        }

        MNN::Tensor *tensor = output.second;
        ret = MnnBlobConverter::CreateOrUpdateBlob(
            &output_blob_arr_[output_blob_size_++], *tensor, name, alloc, gpu_blob_);
        if (ret != MAGIC_XE_SUCCESS) {
            MAGIC_XE_LOGE("MnnBlobConverter::CreateOrUpdateBlob failed:%d.", ret);
            ClearBlobArray();
            return ret;
        }
    }

    return MAGIC_XE_SUCCESS;
}

MagicXEError MnnNetwork::CreateInferRequest() {
    return MAGIC_XE_SUCCESS;
}

MagicXEError MnnNetwork::Reshape(const char **name_arr, const MagicXEDims *dims_arr, size_t dims_size) {

    bool session_flag = false;
    for (int i = 0; i < dims_size; ++i) {
        const char *name = name_arr[i];
        const MagicXEDims *dims = &dims_arr[i];
        bool tensor_flag = false;
        auto input = interpreter_->getSessionInput(session_, name);
        if (nullptr == input) {
            MAGIC_XE_LOGE("Reshape Failed\n");
            return MAGIC_XE_INVALID_MODEL;
        }

        auto origin_shape = input->shape();
        if (dims->size != origin_shape.size()) {
            tensor_flag = true;
        } else {
            for (int i = 0; i < dims->size; ++i) {
                if (dims->value[i] != origin_shape[i]) {
                    tensor_flag = true;
                    break;
                }
            }
        }

        if (tensor_flag) {
            std::vector<int> shape(dims->size, 0);
            for (int i = 0; i < dims->size; ++i) {
                shape[i] = dims->value[i];
            }
            interpreter_->resizeTensor(input, shape);
            session_flag = true;
        }
    }

    if (session_flag) {
        interpreter_->resizeSession(session_);
        CreateBlobArray();
    }

    if (device_type_ == MAGIC_XE_DEVICE_TYPE_OPENCL) {
        interpreter_->updateCacheFile(session_);
    }

    return MAGIC_XE_SUCCESS;
}

MagicXEError MnnNetwork::Reshape() {
    MagicXEError ret = MAGIC_XE_SUCCESS;

    for (auto it : input_max_shapes_) {
        const char *name = it.first.c_str();
        const MagicXEDims &dims = it.second;

        auto input = interpreter_->getSessionInput(session_, name);
        if (nullptr == input) {
            MAGIC_XE_LOGE("Reshape Failed\n");
            return MAGIC_XE_INVALID_MODEL;
        }

        std::vector<int> shape(dims.size, 0);
        for (int i = 0; i < dims.size; ++i) {
            shape[i] = dims.value[i];
        }
        interpreter_->resizeTensor(input, shape);
    }

    interpreter_->resizeSession(session_);

    return ret;
}

MagicXEError MnnNetwork::Forward() {
    MagicXEError ret = MAGIC_XE_SUCCESS;
    for (size_t i = 0; i < input_blob_size_; ++i) {
        MagicXEBlob *blob = input_blob_arr_[i];

        char *name = blob->name;
        MNN::Tensor *input = interpreter_->getSessionInput(session_, name);
        if (input == nullptr) {
            MAGIC_XE_LOGE("getSessionInput Failed");
            return MAGIC_XE_INVALID_MODEL;
        }

        MNN::Tensor *tensor = MnnBlobConverter::ConvertFromBlob(ret, blob);
        if (tensor == nullptr) {
            return MAGIC_XE_INVALID_MODEL;
        }

        bool res = input->copyFromHostTensor(tensor);

        delete tensor;
    }

    MNN::ErrorCode mnn_ret = interpreter_->runSession(session_);
    if (0 != mnn_ret) {
        MAGIC_XE_LOGE("Run Session Failed\n");
        return MAGIC_XE_INVALID_MODEL;
    }

    return MAGIC_XE_SUCCESS;
}

MagicXEError MnnNetwork::InputBlobsGet(const MagicXEBlob ***blob_arr, size_t *blob_size) {
    if (blob_arr == nullptr || blob_size == nullptr) {
        return MAGIC_XE_INVALID_PARAM;
    }
    if (input_blob_arr_ == nullptr || input_blob_size_ <= 0) {
        MAGIC_XE_LOGE("input_blob_arr_:%p is nullptr or input_blob_size_:%zu <= 0", input_blob_arr_, input_blob_size_);
        return MAGIC_XE_INVALID_MODEL;
    }

    *blob_arr = (const MagicXEBlob **)input_blob_arr_;
    *blob_size = input_blob_size_;
    return MAGIC_XE_SUCCESS;
}

MagicXEError MnnNetwork::OutputBlobsGet(const MagicXEBlob ***blob_arr, size_t *blob_size) {
    if (blob_arr == nullptr || blob_size == nullptr) {
        return MAGIC_XE_INVALID_PARAM;
    }
    if (output_blob_arr_ == nullptr || output_blob_size_ <= 0) {
        MAGIC_XE_LOGE(
            "output_blob_arr_:%p is nullptr or output_blob_size_:%zu <= 0", output_blob_arr_, output_blob_size_);
        return MAGIC_XE_INVALID_MODEL;
    }

    *blob_arr = (const MagicXEBlob **)output_blob_arr_;
    *blob_size = output_blob_size_;
    return MAGIC_XE_SUCCESS;
}

MagicXEError MnnNetwork::InputBlobGet(const char *input_name, MagicXEBlob **blob) {

    if (input_blob_arr_ == nullptr || blob == nullptr || input_blob_size_ <= 0) {
        MAGIC_XE_LOGE("blob:%p input_blob_arr_ is nullptr or input_blob_size_:%d is empty", blob, input_blob_size_);
        return MAGIC_XE_INVALID_MODEL;
    }

    int index = -1;
    *blob = FindBlobAndIndexByName(input_blob_arr_, input_blob_size_, input_name, &index);
    if (*blob == NULL) {
        MAGIC_XE_LOGE("Not Find Blob name:%s", input_name != nullptr ? input_name : "");
        return MAGIC_XE_INVALID_PARAM;
    }

    return MAGIC_XE_SUCCESS;
}

MagicXEError MnnNetwork::OutputBlobGet(const char *output_name, const MagicXEBlob **blob) {
    if (output_blob_arr_ == nullptr) {
        MAGIC_XE_LOGE("output_blob_arr_ is nullptr");
        return MAGIC_XE_INVALID_MODEL;
    }
    if (output_blob_size_ <= 0) {
        MAGIC_XE_LOGE("output_blob_size_:%d is empty", output_blob_size_);
        return MAGIC_XE_INVALID_MODEL;
    }

    int index = -1;
    MagicXEBlob *find_blob = FindBlobAndIndexByName(output_blob_arr_, output_blob_size_, output_name, &index);
    if (find_blob == NULL) {
        MAGIC_XE_LOGE("Not Find Blob name:%s", output_name);
        return MAGIC_XE_INVALID_PARAM;
    }

    MNN::Tensor *output = interpreter_->getSessionOutput(session_, find_blob->name);
    if (output == nullptr) {
        MAGIC_XE_LOGE("Name:%s getSessionOutput Failed", find_blob->name);
        return MAGIC_XE_INVALID_MODEL;
    }

    MagicXEError mgxe_err = MnnBlobConverter::CopyToBlob(find_blob, *output, find_blob->data_format);
    if (mgxe_err != MAGIC_XE_SUCCESS) {
        MAGIC_XE_LOGE("Name:%s CopyToBlob Failed:%d", find_blob->name, mgxe_err);
        return mgxe_err;
    }

    *blob = find_blob;
    return MAGIC_XE_SUCCESS;
}

MagicXEError MnnNetwork::AddOutput(const std::string &output_name) {
    save_tensors_.emplace_back(output_name);
    return MAGIC_XE_SUCCESS;
}

} // namespace mnn
} // namespace magic_xe
