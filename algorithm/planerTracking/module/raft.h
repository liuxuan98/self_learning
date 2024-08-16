#ifndef RAFT_HPP
#define RAFT_HPP

#include <iostream>
#include <string>
#include <vector>

#include "magic_xe_blob.h"
#include "magic_xe_common_v2.h"
#include "magic_xe_model.h"
#include "opencv2/opencv.hpp"
#include "plugin/magic_xe_neural_net.h"
#include "utils/magic_xe_json_utils.h"

namespace magic_xe {
namespace raft_algorithm {

class Raft {

public:
    Raft();
    ~Raft();

    MagicXEError Init(const char *models_json);

    MagicXEError Process(const cv::Mat &src_img,
        const cv::Mat &dst_img,
        cv::Mat &template_coords,
        cv::Mat &cur_pw_coords,
        cv::Mat &weights);

private:
    MagicXEError InputPadding(cv::Mat &pad_img, const std::vector<int> &pad_size);

    MagicXEError UnPadding(cv::Mat &unpad_img, const std::vector<int> &pad_size);

    MagicXEError SetInputBlob(cv::Mat &src_img, cv::Mat &dst_img);

    MagicXEError Forward();

    MagicXEError Post(cv::Mat &flow_up_mat, cv::Mat &weights_up_mat);

    cv::Mat Sigmoid(const cv::Mat &input);

private:
    MagicXENeuralNet raft_network_ = nullptr; //use shared_ptr.
    MagicXEBlob *src_input_blob_ = nullptr;
    MagicXEBlob *dst_input_blob_ = nullptr;

    MagicXEBlob *flow_up_blob_ = nullptr;
    MagicXEBlob *weights_up_blob_ = nullptr;
};

} // namespace raft_algorithm
} // namespace magic_xe

#endif
