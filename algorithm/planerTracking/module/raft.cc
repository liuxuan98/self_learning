#include "raft.h"

namespace magic_xe {
namespace raft_algorithm {
static MagicXEError MatToBlob(cv::Mat &src, MagicXEBlob *dst) {

    int num_channel = src.channels();
    int rtype = CV_MAKETYPE(CV_32F, num_channel);
    src.convertTo(src, rtype);

    float *dst_ptr = (float *)MagicXEBufferDataGetV2(dst->buffer);
    if (dst_ptr == NULL) {
        MAGIC_XE_LOGE("Blob dst handle:%d.", MAGIC_XE_INVALID_PARAM);
        return MAGIC_XE_INVALID_PARAM;
    }

    switch (dst->data_format) {
    case MAGIC_XE_DATA_FORMAT_NHWC: {
        memcpy(dst_ptr, src.data, src.total() * src.elemSize()); //image_float.rows * image_float.step;
    } break;
    case MAGIC_XE_DATA_FORMAT_NCHW:
    default: {
        int rows = src.rows;
        int cols = src.cols;
        std::vector<cv::Mat> channels;
        for (int c = 0; c < num_channel; c++) {
            cv::Mat tmp(rows, cols, CV_32FC1, (void *)dst_ptr);
            channels.emplace_back(tmp);
            dst_ptr += rows * cols;
        }
        cv::split(src, channels);
    } break;
    }

    return MAGIC_XE_SUCCESS;
}

void rearrange(const cv::Mat &input_mat, cv::Mat &result) {

    std::vector<cv::Mat> channels;
    cv::split(input_mat, channels);

    for (size_t i = 0; i < channels.size(); ++i) {
        cv::Mat &tmp_channel = channels[i];
        tmp_channel = tmp_channel.reshape(1, 1);
    }
    cv::vconcat(channels, result);
}

MagicXEError GetFeaturemapCoords(int h, int w, cv::Mat &coords) {
    if (coords.rows != 2 || coords.cols != h * w || coords.channels() != 1) {
        MAGIC_XE_LOGE("coords.rows != 2 or coords.cols != h * w or coords.channels() != 1!");
        return MAGIC_XE_INVALID_PARAM;
    }

    //cl优化
    float x_value = 0.f;
    float y_value = 0.f;
    for (int i = 0; i < coords.cols; ++i) {
        if (i != 0 && i % w == 0) {
            x_value = 0;
            ++y_value;
        }
        coords.at<float>(0, i) = x_value;
        coords.at<float>(1, i) = y_value;
        ++x_value;
    }

    return MAGIC_XE_SUCCESS;
}

cv::Mat Raft::Sigmoid(const cv::Mat &input) {
    cv::Mat output;
    cv::exp(-input, output);
    output = 1 / (1 + output);
    return output;
}
Raft::Raft() {
}

Raft::~Raft() {
    if (raft_network_ != nullptr) {
        MagicXENeuralNetClose(&raft_network_);
        raft_network_ = nullptr;
    }
}

MagicXEError Raft::Init(const char *models_json) {
    MagicXEError mgxe_err = MAGIC_XE_SUCCESS;

    MagicXEModelJsonArray *models = nullptr;
    mgxe_err = MagicXEModelUtilsModelsParser(models_json, &models);
    if (mgxe_err != MAGIC_XE_SUCCESS) {
        MAGIC_XE_LOGE("MagicXEModelUtilsModelsParser failed:%d", mgxe_err);
        return mgxe_err;
    }

    if (models->arr_size != 1) {
        MAGIC_XE_LOGE("models->arr_size != 1");
        return MAGIC_XE_INVALID_PARAM;
    }
    mgxe_err = MagicXENeuralNetOpen(models->arr[0].model_path, &models->arr[0].runtime, &raft_network_);
    if (mgxe_err != MAGIC_XE_SUCCESS) {
        MAGIC_XE_LOGE("MagicXENeuralNetworkOpen failed!");
        goto __end;
    }

    mgxe_err = MagicXENeuralNetInputBlobGet(raft_network_, "src_img", &src_input_blob_);
    if (mgxe_err != MAGIC_XE_SUCCESS) {
        MAGIC_XE_LOGE("MagicXENeuralNetInputBlobGet failed:%d", mgxe_err);
        goto __end;
    }

    mgxe_err = MagicXENeuralNetInputBlobGet(raft_network_, "dst_img", &dst_input_blob_);
    if (mgxe_err != MAGIC_XE_SUCCESS) {
        MAGIC_XE_LOGE("MagicXENeuralNetInputBlobGet failed:%d", mgxe_err);
        goto __end;
    }

__end:
    if (mgxe_err != MAGIC_XE_SUCCESS) {
        if (raft_network_ != nullptr) {
            MagicXENeuralNetClose(&raft_network_);
            raft_network_ = nullptr;
        }
    }
    MagicXEModelUtilsModelsFree(&models);
    return mgxe_err;
}
MagicXEError Raft::Process(const cv::Mat &src_img,
    const cv::Mat &dst_img,
    cv::Mat &template_coords,
    cv::Mat &cur_pw_coords,
    cv::Mat &weights) {
    MagicXEError mgxe_err = MAGIC_XE_SUCCESS;
    //INSERT_TIME_POINT_START("Host", "raft infer pre_process");
    if (src_img.size() != dst_img.size()) {
        MAGIC_XE_LOGE("src_img.size() != dst_img.size()");
        return MAGIC_XE_INVALID_PARAM;
    }

    int h = src_img.rows;
    int w = src_img.cols;

    cv::Mat src_img_rgb;
    cv::cvtColor(src_img, src_img_rgb, cv::COLOR_BGR2RGB); //video_common.

    cv::Mat dst_img_rgb;
    cv::cvtColor(dst_img, dst_img_rgb, cv::COLOR_BGR2RGB);

    int pad_ht = (((h / 8) + 1) * 8 - h) % 8;
    int pad_wd = (((w / 8) + 1) * 8 - w) % 8;

    std::vector<int> pad_vec = {pad_wd / 2, pad_wd - pad_wd / 2, pad_ht / 2, pad_ht - pad_ht / 2};
    if (pad_vec[0] > 0 || pad_vec[1] > 0 || pad_vec[2] > 0 || pad_vec[3] > 0) {
        //tocrh.F.padding.
        cv::copyMakeBorder(
            src_img_rgb, src_img_rgb, pad_vec[2], pad_vec[3], pad_vec[0], pad_vec[1], cv::BORDER_REPLICATE);

        cv::copyMakeBorder(
            dst_img_rgb, dst_img_rgb, pad_vec[2], pad_vec[3], pad_vec[0], pad_vec[1], cv::BORDER_REPLICATE);
    }

    //INSERT_TIME_POINT_END("Host", "raft infer pre_process");

    //model infer
    //INSERT_TIME_POINT_START("Host", "raft infer SetInputBlob");
    mgxe_err = SetInputBlob(src_img_rgb, dst_img_rgb);
    //INSERT_TIME_POINT_END("Host", "raft infer SetInputBlob");

    MAGIC_XE_IF_ERROR_RETURN(mgxe_err, "SetInputBlob failed:%d.", mgxe_err);

    //INSERT_TIME_POINT_START("Host", "raft infer Forward");
    mgxe_err = Forward();
    //INSERT_TIME_POINT_END("Host", "raft infer Forward");

    MAGIC_XE_IF_ERROR_RETURN(mgxe_err, "Forward failed:%d.", mgxe_err);

    cv::Mat flow_up_mat;
    cv::Mat weights_up_mat;
    //INSERT_TIME_POINT_START("Host", "raft infer Post");
    mgxe_err = Post(flow_up_mat, weights_up_mat);
    //INSERT_TIME_POINT_END("Host", "raft infer Post");

    MAGIC_XE_IF_ERROR_RETURN(mgxe_err, "Post failed:%d.", mgxe_err);
    //INSERT_TIME_POINT_START("Host", "raft infer post_process");
    //unpadding
    UnPadding(flow_up_mat, pad_vec);
    UnPadding(weights_up_mat, pad_vec);
    //sigmoid
    cv::Mat weights_up_sigmoid = Sigmoid(weights_up_mat);

    cv::Mat flat_flow;
    rearrange(flow_up_mat, flat_flow);
    rearrange(weights_up_sigmoid, weights);
    h = flow_up_mat.rows;
    w = flow_up_mat.cols;

    cv::Mat src_coords_mat(2, h * w, CV_32F);
    GetFeaturemapCoords(h, w, src_coords_mat);

    template_coords = src_coords_mat.clone();
    cv::add(flat_flow, src_coords_mat, cur_pw_coords);
    //INSERT_TIME_POINT_END("Host", "raft infer post_process");
    return mgxe_err;
}

MagicXEError Raft::SetInputBlob(cv::Mat &src_img, cv::Mat &dst_img) {
    MagicXEError mgxe_err = MAGIC_XE_SUCCESS;

    int h = src_img.rows;
    int w = src_img.cols;

    /**
	 * Reshape
	 */
    auto &src_dims = src_input_blob_->dims.value;
    auto &dst_dims = dst_input_blob_->dims.value;
    if (src_dims[2] != h || src_dims[3] != w) {
        const char *names[] = {"src_img", "dst_img"};
        MagicXEDims dims_new[2];
        dims_new[0].size = dims_new[1].size = 4;
        dims_new[0].value[0] = dims_new[1].value[0] = 1;
        dims_new[0].value[1] = dims_new[1].value[1] = 3;
        dims_new[0].value[2] = dims_new[1].value[2] = h;
        dims_new[0].value[3] = dims_new[1].value[3] = w;

        mgxe_err = MagicXENeuralNetReshape(raft_network_, names, dims_new, 2);
        MAGIC_XE_IF_ERROR_RETURN(mgxe_err, "MagicXENeuralNetReshape failed:%d.", mgxe_err);
    }

    mgxe_err = MagicXENeuralNetInputBlobGet(raft_network_, "src_img", &src_input_blob_);
    if (mgxe_err != MAGIC_XE_SUCCESS) {
        MAGIC_XE_LOGE("MagicXENeuralNetInputBlobGet failed:%d", mgxe_err);
        return mgxe_err;
    }
    mgxe_err = MagicXENeuralNetInputBlobGet(raft_network_, "dst_img", &dst_input_blob_);
    if (mgxe_err != MAGIC_XE_SUCCESS) {
        MAGIC_XE_LOGE("MagicXENeuralNetInputBlobGet failed:%d", mgxe_err);
        return mgxe_err;
    }

    mgxe_err = MatToBlob(src_img, src_input_blob_);
    MAGIC_XE_IF_ERROR_RETURN(mgxe_err, "MatToBlob failed:%d.", mgxe_err);

    mgxe_err = MatToBlob(dst_img, dst_input_blob_);
    MAGIC_XE_IF_ERROR_RETURN(mgxe_err, "MatToBlob failed:%d.", mgxe_err);

    return mgxe_err;
}

MagicXEError Raft::Forward() {
    MagicXEError mgxe_err = MAGIC_XE_SUCCESS;

    mgxe_err = MagicXENeuralNetForward(raft_network_);
    MAGIC_XE_IF_ERROR_RETURN(mgxe_err, "MagicXENeuralNetForward failed:%d.", mgxe_err);

    mgxe_err = MagicXENeuralNetOutputBlobGet(raft_network_, "flow_up", (const MagicXEBlob **)&flow_up_blob_);
    MAGIC_XE_IF_ERROR_RETURN(mgxe_err, "MagicXENeuralNetOutputBlobGet flow_high failed:%d.", mgxe_err);

    mgxe_err = MagicXENeuralNetOutputBlobGet(raft_network_, "weights_up", (const MagicXEBlob **)&weights_up_blob_);
    MAGIC_XE_IF_ERROR_RETURN(mgxe_err, "MagicXENeuralNetOutputBlobGet weights_up failed:%d.", mgxe_err);

    return mgxe_err;
}

MagicXEError Raft::Post(cv::Mat &flow_up_mat, cv::Mat &weights_up_mat) {
    MagicXEError mgxe_err = MAGIC_XE_SUCCESS;

    //flow_low_blob_ 1 2 27 48

    //flow_up_blob_ 1 2 216 384
    int n = flow_up_blob_->dims.value[0];
    int c = flow_up_blob_->dims.value[1];
    int h = flow_up_blob_->dims.value[2];
    int w = flow_up_blob_->dims.value[3];

    float *flow_up_ptr = (float *)MagicXEBufferDataGetV2(flow_up_blob_->buffer);
    std::vector<cv::Mat> channel_mat;
    //channel_mat.clear();
    for (int i = 0; i < c; ++i) {
        channel_mat.emplace_back(cv::Mat(h, w, CV_32FC1, (void *)(flow_up_ptr + i * h * w)));
    }
    cv::merge(channel_mat, flow_up_mat);

    // weight_low_blob_ 1 1 27 48

    // weight_up_blob_ 1 1 216 384
    n = weights_up_blob_->dims.value[0];
    c = weights_up_blob_->dims.value[1];
    h = weights_up_blob_->dims.value[2];
    w = weights_up_blob_->dims.value[3];
    float *weights_up_ptr = (float *)MagicXEBufferDataGetV2(weights_up_blob_->buffer);
    channel_mat.clear();
    for (int i = 0; i < c; ++i) {
        channel_mat.emplace_back(cv::Mat(h, w, CV_32FC1, (void *)(weights_up_ptr + i * h * w)));
    }
    cv::merge(channel_mat, weights_up_mat);

    //const_volume_blob_ 1296,1,27,48. not using.

    return mgxe_err;
}
MagicXEError Raft::InputPadding(cv::Mat &pad_img, const std::vector<int> &pad_size) {
    if (pad_size.size() != 4) {
        return MAGIC_XE_INVALID_PARAM;
    }

    cv::copyMakeBorder(pad_img, pad_img, pad_size[2], pad_size[3], pad_size[0], pad_size[1], cv::BORDER_REPLICATE);

    return MAGIC_XE_SUCCESS;
}

MagicXEError Raft::UnPadding(cv::Mat &unpad_img, const std::vector<int> &pad_size) {
    if (unpad_img.empty()) {
        MAGIC_XE_LOGE("unpad_img is empty.");
        return MAGIC_XE_INVALID_PARAM;
    }
    int ht = unpad_img.rows;
    int wd = unpad_img.cols;

    std::vector<int> c = {pad_size[2], ht - pad_size[3], pad_size[0], wd - pad_size[1]};
    unpad_img = unpad_img(cv::Range(c[0], c[1]), cv::Range(c[2], c[3]));

    return MAGIC_XE_SUCCESS;
}

} // namespace raft_algorithm
} // namespace magic_xe
