#include "plane_track.h"

#include <Eigen/Dense>
#include <Eigen/QR>
#include <algorithm>
#include <cmath>
#include <opencv2/core/eigen.hpp>

#include "magic_xe_json.h"
#include "plugin/magic_xe_plugin_template.h"
//#include "time_profile.h"
#include "utils/magic_xe_json_utils.h"

//using namespace magic_xe::tool;

MAGIC_XE_ADAPTER_STRUCT_TO_CLASS(PlaneTrackAlgorithm, plane_track, MAGIC_XE_PLUGIN_ALGORITHM_TYPE_VIDEO);

namespace magic_xe {
namespace plane_track {

void CvMatConvertFrame(const cv::Mat &mat, MagicXEFrameV2 *frame) {
    cv::Mat temp = mat;
    size_t copy_size = temp.cols * temp.elemSize();
    uint8_t *data = frame->data[0];
    for (int i = 0; i < frame->video.height; ++i) {
        uchar *row_ptr = temp.ptr<uchar>(i);
        memcpy(data, row_ptr, copy_size);
        data += frame->linesize[0];
    }
}
cv::Mat warp_image(cv::Mat src_img, cv::Mat dst_img, const std::vector<cv::Point> points) {
    // 获取目标图像的宽度和高度
    int h = src_img.rows;
    int w = src_img.cols;
    // 定义目标图像的四个顶点
    cv::Point2f src_points[4] = {cv::Point2f(0, 0), cv::Point2f(w, 0), cv::Point2f(w, h), cv::Point2f(0, h)};
    // 定义选取范围的四个顶点
    cv::Point2f dst_points[4] = {points[0], points[1], points[2], points[3]};
    // 计算透视变换矩阵
    cv::Mat matrix = cv::getPerspectiveTransform(src_points, dst_points);
    // 进行透视变换
    cv::Mat warped_image;
    cv::warpPerspective(src_img, warped_image, matrix, cv::Size(dst_img.cols, dst_img.rows));
    return warped_image;
}

cv::Mat triv_tracker_vis(cv::Mat frame, cv::Mat init_mask, cv::Mat warpimage, cv::Mat H_2init) {
    cv::Mat current_mask;
    cv::Mat H_2init_inv;
    cv::invert(H_2init, H_2init_inv);
    cv::warpPerspective(init_mask, current_mask, H_2init_inv, frame.size(), cv::INTER_NEAREST);

    cv::Mat current_warpimage;
    cv::warpPerspective(warpimage, current_warpimage, H_2init_inv, frame.size(), cv::INTER_NEAREST);

    cv::Mat rgb_mask;
    cv::cvtColor(current_mask, rgb_mask, cv::COLOR_GRAY2BGR);

    cv::Mat warped_image_masked;
    cv::bitwise_and(current_warpimage, rgb_mask, warped_image_masked);

    cv::Mat dst_image_masked;
    cv::bitwise_not(rgb_mask, rgb_mask);
    cv::bitwise_and(frame, rgb_mask, dst_image_masked);

    double alpha = 1.0;
    cv::Mat blended_image;
    cv::addWeighted(warped_image_masked, alpha, dst_image_masked, 1 - alpha, 0, blended_image);

    cv::Mat result_img;
    cv::add(blended_image, dst_image_masked, result_img);

    return result_img;
}

cv::Mat PerspectiveTransPoints(const cv::Mat &points, const cv::Mat &h2init) {
    cv::Mat h_2init_inv;
    cv::invert(h2init, h_2init_inv);
    cv::Mat points_homogeneous;
    cv::copyMakeBorder(points, points_homogeneous, 0, 0, 0, 1, cv::BORDER_CONSTANT, 1.0f);
    cv::Mat transformed_points_homogeneous = (h_2init_inv * points_homogeneous.t()).t();
    cv::Mat dividend = transformed_points_homogeneous(cv::Range::all(), cv::Range(0, 2));
    cv::Mat divide = transformed_points_homogeneous(cv::Range::all(), cv::Range(2, 3));
    if (dividend.cols != divide.cols) {
        cv::hconcat(divide, divide, divide);
    }
    cv::Mat res = dividend / divide;
    return res;
}
MagicXEError ConvertPointsfromhomogeneous(cv::Mat &points, cv::Mat &res) {
    float eps = 1e-8;

    if (points.empty()) {
        MAGIC_XE_LOGE("input mat(points) is empty!");
        return MAGIC_XE_INVALID_PARAM;
    }
    cv::Mat z_vec = points.row(points.rows - 1);
    cv::Mat mask = cv::abs(z_vec) > eps;
    // compute accel scale
    cv::Mat scale;
    cv::Mat ones = cv::Mat::ones(z_vec.size(), CV_32FC1);
    cv::add(z_vec, cv::Scalar(eps), z_vec);
    cv::Mat ture_mat = 1 / z_vec;

    // depends on mask to select elements.
    ture_mat.copyTo(scale, mask);
    ones.copyTo(scale, ~mask);
    cv::vconcat(scale, scale, scale);
    res = scale.mul(points(cv::Range(0, points.rows - 1), cv::Range::all()));

    return MAGIC_XE_SUCCESS;
}

cv::Mat composeHomography(const std::vector<cv::Mat> &Hs) {
    cv::Mat result = cv::Mat::eye(3, 3, CV_32F);
    for (auto it = Hs.rbegin(); it != Hs.rend(); ++it) {
        if (it->empty()) {
            return cv::Mat();
        }
        result = result * (*it);
    }
    result /= (result.at<float>(2, 2));
    return result;
}

void qrDecomposition(const cv::Mat &A, cv::Mat &Q, cv::Mat &R) {
    Eigen::MatrixXf eigenA;
    cv::cv2eigen(A, eigenA);
    int cols = A.cols;
    Eigen::HouseholderQR<Eigen::MatrixXf> qr(eigenA);
    Eigen::MatrixXf Qeigen = qr.householderQ();
    Eigen::MatrixXf Reigen = qr.matrixQR().triangularView<Eigen::Upper>();

    cv::eigen2cv(Qeigen, Q);
    cv::eigen2cv(Reigen, R);
    Q = Q(cv::Range(0, Q.rows), cv::Range(0, cols));
    R = R(cv::Range(0, cols), cv::Range(0, R.cols));
}

cv::Mat solveHomography(const cv::Mat &A, const cv::Mat &b, const cv::Mat &transform1, const cv::Mat &transform2) {
    cv::Mat Q, R;
    qrDecomposition(A, Q, R);
    cv::Mat lhs = R;
    cv::Mat rhs = Q.t() * b;

    cv::Mat solution;
    cv::solve(lhs, rhs, solution, cv::DECOMP_NORMAL);

    cv::copyMakeBorder(solution, solution, 0, 1, 0, 0, cv::BORDER_CONSTANT, 1.0f);

    cv::Mat H = solution.reshape(1, 3);
    H = transform2.inv() * (H * transform1);

    float eps = std::numeric_limits<float>::epsilon();
    H /= (H.at<float>(2, 2) + eps);

    return H;
}
void RowDimNorm(const cv::Mat &src, int normType, cv::Mat &res) {
    float norm = 0.0;

    for (int i = 0; i < src.rows; ++i) {
        cv::Mat row = src.row(i);
        norm = cv::norm(row, normType);
        res.at<float>(i, 0) = norm;
    }
}

inline int64_t rightmost_zero(const int64_t n) {
    int64_t z, i;
    for (z = n, i = 0; z % 2 == 1; z /= 2, i++) {
        ;
    }
    return i;
}

PlaneTrackParam::PlaneTrackParam() {
    config_json_ = R"({"input_downscale_factor": 5})";
    input_downscale_factor_ = 5;
}
PlaneTrackParam::~PlaneTrackParam() {
}
void PlaneTrackParam::Reset() {
}
PlaneTrackAlgorithm::PlaneTrackAlgorithm() {
    param_.reset(new PlaneTrackParam());
}
PlaneTrackAlgorithm::~PlaneTrackAlgorithm() {
    DeInit();
}

const char *PlaneTrackAlgorithm::ModelsSchemaGet() {
    const char *models_schema = R"({
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "models.json",
        "type": "object",
        "description": "Plane Track Algorithm  model file configuration,which include only one model",

        "properties": {
            "models": {
                "type": "object",
                "description": "PlaneTrackAlgorithm Model",
                "items":[{
                        "type": "object",
                        "properties": {
                            "model_path": {"type": "string",
                            "default_openvino":"WM_PlaneTrack_X86_CPU_OpenVino_FP16_v1.0.0.1.model",
                            "default_mnn":"WM_PlaneTrack_Arm_Mnn_FP16_v1.0.0.1.model"
                            },
                            "device_type": {"type": "int", "default_value": 16, "description": "MagicXEGetDefaultCpuDeviceTypeV2() to get default cpu device type"},
                            "num_thread": {"type": "int", "default_value": 4},
                            "use_gpu": {"type": "bool", "default_value": false}
                        }
                    }
                ]
            }
        }
    })";
    return models_schema;
}
const char *PlaneTrackAlgorithm::ConfigSchemaGet() {
    const char *config_schema = R"({
        "$schema": "https://json-schema.org/draft/2019-09/schema",
        "title": "config.json",
        "description": "plane track config,include downscale_factor only is 3 or 5 or 8.",
        "type": "object",
        "properties": {
            "downscale_factor": {"type": "int", "minimum": 3, "maximum": 8,"default_value":5}
        },
        "process_attrs": {
            "input": {"pix_fmt":1,"mem_type":0,"width":-1,"height":-1}
        }
    })";
    return config_schema;
}

const char *PlaneTrackAlgorithm::DataSchemaGet(bool is_input) {

    if (is_input) {
        const char *data_schema = R"({
        "$schema": "https://json-schema.org/draft/2019-09/schema",
        "title": "data_schema.json",
        "description": "Palne Track inputdata schema",
        "type": "object",
        "properties": {
            "four_point_array": {
                "description": "four point array,include users input palne four point",
                "type": "array",
                "items": {
                    "type": "MagicXEPointV2",
                    "properties": {
                        "x": {"type": "MAGIC_XE_VALUE_TYPE_FLOAT"},
                        "y": {"type": "MAGIC_XE_VALUE_TYPE_FLOAT"}
                    }
                }
            }
        }
    })";
        return data_schema;
    } else {
        const char *data_schema = R"({
        "$schema": "https://json-schema.org/draft/2019-09/schema",
        "title": "data_schema.json",
        "description": "Palne Track outputdata schema",
        "type": "object",
        "properties": {
            "coords": {
                "description": "four point array,plane track algorithm output four point",
                "type": "array",
                "items": {
                    "type": "MagicXEPointV2",
                    "properties": {
                        "x": {"type": "MAGIC_XE_VALUE_TYPE_FLOAT"},
                        "y": {"type": "MAGIC_XE_VALUE_TYPE_FLOAT"}
                    }
                }
            }
        }
    })";
        return data_schema;
    }
}

MagicXEError PlaneTrackAlgorithm::Init(const char *models_json) {
    MagicXEError mgxe_err = MAGIC_XE_SUCCESS;

    if (models_json == nullptr) {
        MAGIC_XE_LOGE("models_json is nullptr:%d!", MAGIC_XE_INVALID_PARAM);
        return MAGIC_XE_INVALID_PARAM;
    }

    raft_ = nullptr;
    raft_.reset(new Raft());
    if (raft_ == nullptr) {
        MAGIC_XE_LOGE("raft_ is nullptr:%d!", MAGIC_XE_INVALID_PARAM);
        return MAGIC_XE_INVALID_PARAM;
    }
    mgxe_err = raft_->Init(models_json);
    if (mgxe_err != MAGIC_XE_SUCCESS) {
        MAGIC_XE_LOGE("raft_->Init(models_json) failed:%d!", mgxe_err);
        return mgxe_err;
    }

    return mgxe_err;
}

void PlaneTrackAlgorithm::DeInit() {
}

MagicXEError PlaneTrackAlgorithm::ParamSet(const char *config_json) {

    if (config_json == nullptr) {
        MAGIC_XE_LOGE("config_json:%p has nullptr", config_json);
        return MAGIC_XE_INVALID_PARAM;
    }

    MagicXEJsonHandle json_handle = nullptr;
    MagicXEError mgxe_err = MagicXEJsonCreate(nullptr, (void *)config_json, strlen(config_json), &json_handle);
    if (mgxe_err != MAGIC_XE_SUCCESS) {
        MAGIC_XE_LOGE("MagicXEJsonUtilsCreate:%s failed:%d", config_json, mgxe_err);
        return mgxe_err;
    }

    MagicXEJsonObject root_obj = MagicXEJsonRootGet(json_handle);
    MagicXEJsonObject config_obj = MagicXEJsonObjectGet(root_obj, "downscale_factor");
    if (config_obj != nullptr) {

        int downscale_factor = MagicXEJsonIntGet(config_obj, 5);
        if (downscale_factor != 3 && downscale_factor != 5 && downscale_factor != 8) {
            MAGIC_XE_LOGE("set downscale factor is invalid:%d,should be 3 or 5 or 8.", downscale_factor);
            MagicXEJsonDestory(&json_handle);
            return MAGIC_XE_INVALID_PARAM;
        }
        param_->input_downscale_factor_ = downscale_factor;
        param_->config_json_ = std::string(config_json);

    } else {
        MAGIC_XE_LOGE("config_obj is nullptr.");
        MagicXEJsonDestory(&json_handle);
        return MAGIC_XE_INVALID_PARAM;
    }

    MagicXEJsonDestory(&json_handle);

    return mgxe_err;
}

const char *PlaneTrackAlgorithm::ParamGet() {
    return param_->config_json_.c_str();
}

void PlaneTrackAlgorithm::Reset() {
    param_->Reset();

    track_init_ = false;

    prev_img_.release();
    prev_H2init_.release();
    last_good_H2init_.release();

    template_img_.release();
    np_template_mask_.release();
    template_mask_.release();

    n_lost_ = 0;
    lost_ = false;
    std::vector<cv::Mat>().swap(point_vec);
}

void PlaneTrackAlgorithm::PointGenMask(const std::vector<cv::Point> &points, cv::Mat &mask) {
    std::vector<std::vector<cv::Point>> contours = {points};
    cv::fillPoly(mask, contours, cv::Scalar(255));
}

MagicXEError PlaneTrackAlgorithm::InitMask(cv::Mat &img, const std::vector<cv::Point> &points) {

    cv::Mat mask = cv::Mat::zeros(img.size(), CV_8UC1);
    PointGenMask(points, mask);
    init_mask_ = mask.clone();
    warp_img_ = cv::imread("E:/MagicXE_VideoSegment/videotrack/data/plane_track/1.jpg");
    warp_img_ = warp_image(warp_img_, mask, points);

    int h = img.rows;
    int w = img.cols;
    if (h <= 0 || w <= 0) {
        MAGIC_XE_LOGE("h:%d w:%d is invalid!", h, w);
        return MAGIC_XE_INVALID_PARAM;
    }

    if (h * w <= 1280 * 720) {
        cv::resize(img,
            img,
            cv::Size(),
            1.0 / (param_->input_downscale_factor_ - 1),
            1.0 / (param_->input_downscale_factor_ - 1));
        cv::resize(mask,
            mask,
            cv::Size(),
            1.0 / (param_->input_downscale_factor_ - 1),
            1.0 / (param_->input_downscale_factor_ - 1));

    }
#if defined(__APPLE__) && defined(__aarch64__)

    else if (h * w > 2160 * 3200) {
        int input_downscale_factor = param_->input_downscale_factor_;

        int new_w = 2160;
        int new_h = h * new_w / w;
        if (w < h) {
            new_h = 2160;
            new_w = w * new_h / h;
        }
        new_w = new_w / input_downscale_factor;
        new_h = new_h / input_downscale_factor;

        cv::resize(img, img, cv::Size(new_w, new_h));
        cv::resize(mask, mask, cv::Size(new_w, new_h));
    }
#else
    else if (h * w > 2160 * 3840) {

        int input_downscale_factor = param_->input_downscale_factor_;

        int new_w = 2160;
        int new_h = h * new_w / w;
        if (w < h) {
            new_h = 2160;
            new_w = w * new_h / h;
        }
        new_w = new_w / input_downscale_factor;
        new_h = new_h / input_downscale_factor;

        cv::resize(img, img, cv::Size(new_w, new_h));
        cv::resize(mask, mask, cv::Size(new_w, new_h));
    }

#endif
    else {
        cv::resize(img, img, cv::Size(), 1.0 / param_->input_downscale_factor_, 1.0 / param_->input_downscale_factor_);
        cv::resize(
            mask, mask, cv::Size(), 1.0 / param_->input_downscale_factor_, 1.0 / param_->input_downscale_factor_);
    }

    h = img.rows;
    w = img.cols;

    if ((0 < h && h <= 216) && (0 < w && w <= 384)) {
        cv::resize(img, img, cv::Size(384, 216));
        cv::resize(mask, mask, cv::Size(384, 216));
    }

    template_img_ = img;
    np_template_mask_ = mask;

    //THRESH_BINARY
    cv::threshold(mask, template_mask_, 1, 1, cv::THRESH_BINARY);

    std::vector<cv::Vec4i> hierarchy;
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

    if (contours.size() != 1) {
        MAGIC_XE_LOGE("findContours contours size not equal 1!");
        return MAGIC_XE_INVALID_PARAM;
    }
    // tracker state.
    prev_H2init_ = cv::Mat::eye(3, 3, CV_32F);
    last_good_H2init_ = cv::Mat::eye(3, 3, CV_32F);
    prev_img_ = img;
    lost_ = false;
    n_lost_ = 0;

    return MAGIC_XE_SUCCESS;
}
MagicXEError PlaneTrackAlgorithm::Process(const MagicXEFrameV2 *input_frame,
    const MagicXEFrameV2 *assist,
    const MagicXEDictionary *input_data,
    MagicXEFrameV2 *output,
    MagicXEDictionary **output_data) {

    if (input_frame == nullptr) {
        MAGIC_XE_LOGE("input_frame is nullptr:%d!", MAGIC_XE_INVALID_PARAM);
        return MAGIC_XE_INVALID_PARAM;
    }

    if (input_frame->video.format != MAGIC_XE_PIX_FMT_RGB24 && input_frame->video.format != MAGIC_XE_PIX_FMT_BGR24) {
        MAGIC_XE_LOGE(
            "input_frame format is not MAGIC_XE_PIX_FMT_RGB24 or MAGIC_XE_PIX_FMT_BGR24:%d!", MAGIC_XE_INVALID_PARAM);
        return MAGIC_XE_INVALID_PARAM;
    }

    MagicXEError mgxe_err = MAGIC_XE_SUCCESS;

    if (input_data != nullptr && track_init_ == false) {
        //first frame save,first frame process.
        cv::Mat frame;
        frame = cv::Mat(input_frame->video.height,
            input_frame->video.width,
            CV_8UC3,
            input_frame->data[0],
            input_frame->linesize[0])
                    .clone();

        if (frame.empty()) {
            MAGIC_XE_LOGE("input frame mat is empty!");
            return MAGIC_XE_INVALID_PARAM;
        }

        if (input_frame->video.format == MAGIC_XE_PIX_FMT_RGB24) {
            cv::cvtColor(frame, frame, cv::COLOR_RGB2BGR);
        }
        const MagicXEArray *points_array = MagicXEDictionaryArrayGet(input_data, "four_point_array", nullptr);
        int point_size = points_array->size;
        if (point_size != 4) {
            MAGIC_XE_LOGE("input_data points size:%d is not 4!", point_size);
            return MAGIC_XE_INVALID_PARAM;
        }
        std::vector<cv::Point> points_vec;
        for (int i = 0; i < point_size; ++i) {
            cv::Mat point_mat(1, 2, CV_32F);
            point_mat.at<float>(0, 0) = points_array->values[i].point_value->x;
            point_mat.at<float>(0, 1) = points_array->values[i].point_value->y;
            point_vec.emplace_back(point_mat);
            cv::Point point = cv::Point(points_array->values[i].point_value->x, points_array->values[i].point_value->y);
            points_vec.emplace_back(point);
        }

        mgxe_err = InitMask(frame, points_vec);
        if (mgxe_err != MAGIC_XE_SUCCESS) {
            MAGIC_XE_LOGE("InitMask failed:%d!", mgxe_err);
            return mgxe_err;
        }
        track_init_ = true;

        return mgxe_err;
    }

    //INSERT_TIME_POINT_START("Host", "Pre-processing");
    int h = input_frame->video.height;
    int w = input_frame->video.width;
    cv::Mat input_img;
    input_img = cv::Mat(
        input_frame->video.height, input_frame->video.width, CV_8UC3, input_frame->data[0], input_frame->linesize[0]);
    origin_frame_ = input_img.clone();
    if (input_frame->video.format == MAGIC_XE_PIX_FMT_RGB24) {
        cv::cvtColor(input_img, input_img, cv::COLOR_RGB2BGR);
    }

    if (h * w <= 1280 * 720) {
        cv::resize(input_img,
            input_img,
            cv::Size(),
            1.0 / (param_->input_downscale_factor_ - 1),
            1.0 / (param_->input_downscale_factor_ - 1));

    }
#if defined(__APPLE__) && defined(__aarch64__)

    else if (h * w > 2160 * 3200) {
        int input_downscale_factor = param_->input_downscale_factor_;

        int new_w = 2160;
        int new_h = h * new_w / w;
        if (w < h) {
            new_h = 2160;
            new_w = w * new_h / h;
        }
        new_w = new_w / input_downscale_factor;
        new_h = new_h / input_downscale_factor;

        cv::resize(input_img, input_img, cv::Size(new_w, new_h));
    }
#else
    else if (h * w > 2160 * 3840) {
        int input_downscale_factor = param_->input_downscale_factor_;

        int new_w = 2160;
        int new_h = h * new_w / w;
        if (w < h) {
            new_h = 2160;
            new_w = w * new_h / h;
        }
        new_w = new_w / input_downscale_factor;
        new_h = new_h / input_downscale_factor;

        cv::resize(input_img, input_img, cv::Size(new_w, new_h));
    }

#endif
    else {
        cv::resize(input_img,
            input_img,
            cv::Size(),
            1.0 / param_->input_downscale_factor_,
            1.0 / param_->input_downscale_factor_);
    }

    if ((0 < input_img.rows && input_img.rows <= 216) && (0 < input_img.cols && input_img.cols <= 384)) {
        cv::resize(input_img, input_img, cv::Size(384, 216));
    }

    float scaled_h = 1.0 * input_img.rows / h;
    float scaled_w = 1.0 * input_img.cols / w;

    if (n_lost_ > 10) {
        last_good_H2init_ = cv::Mat::eye(3, 3, CV_32F);
    }

    cv::Mat prewarp_H = last_good_H2init_;

    cv::Mat prewarped_input;
    cv::warpPerspective(
        input_img, prewarped_input, prewarp_H, cv::Size(input_img.cols, input_img.rows), cv::INTER_LINEAR);

    cv::Mat pw_mask;
    cv::warpPerspective(cv::Mat::ones(input_img.size(), CV_8UC1),
        pw_mask,
        prewarp_H,
        cv::Size(input_img.cols, input_img.rows),
        cv::INTER_LINEAR);

    cv::threshold(pw_mask, pw_mask, 0, 1, cv::THRESH_BINARY);

    cv::Mat cur_pw_coords;
    cv::Mat template_coords;
    cv::Mat weights;

    //INSERT_TIME_POINT_END("Host", "Pre-processing");

    //INSERT_TIME_POINT_START("Host", "raft infer");

    mgxe_err = raft_->Process(template_img_, prewarped_input, template_coords, cur_pw_coords, weights);
    if (mgxe_err != MAGIC_XE_SUCCESS) {
        MAGIC_XE_LOGE("raft_->Process(input_img, prewarped_input) failed:%d!", mgxe_err);
        return mgxe_err;
    }
    //INSERT_TIME_POINT_END("Host", "raft infer");

    //INSERT_TIME_POINT_START("Host", "Post-processing");

    std::vector<std::vector<float>> temp_coords_vec;
    std::vector<std::vector<float>> cur_pw_coords_vec;
    std::vector<std::vector<float>> weights_vec;

    mgxe_err =
        MaskCoords(template_coords, cur_pw_coords, weights, pw_mask, temp_coords_vec, cur_pw_coords_vec, weights_vec);
    if (mgxe_err != MAGIC_XE_SUCCESS) {
        MAGIC_XE_LOGE("MaskCoords failed!");
        return mgxe_err;
    }

    if (temp_coords_vec[0].size() <= 20 || cur_pw_coords_vec[0].size() <= 20 || weights_vec[0].size() <= 20) {
        MAGIC_XE_LOGE("Tracking object mask lost or not found,track failed!");
        return MAGIC_XE_NOT_FOUND;
    }

    cv::Mat temp_coords_mat(cv::Size(500, (int)temp_coords_vec.size()), CV_32F);
    cv::Mat cur_pw_coords_mat(cv::Size(500, (int)cur_pw_coords_vec.size()), CV_32F);
    cv::Mat weights_out(cv::Size(500, (int)weights_vec.size()), CV_32F);

    mgxe_err =
        SubSampler(temp_coords_vec, cur_pw_coords_vec, weights_vec, temp_coords_mat, cur_pw_coords_mat, weights_out);
    if (mgxe_err != MAGIC_XE_SUCCESS) {
        MAGIC_XE_LOGE("SubSampler failed");
        return mgxe_err;
    }

    cv::Mat H_prewarped2init;
    mgxe_err = FindHomography(cur_pw_coords_mat, temp_coords_mat, weights_out, H_prewarped2init);
    if (mgxe_err != MAGIC_XE_SUCCESS) {
        MAGIC_XE_LOGE("FindHomography failed:%d.", mgxe_err);
        return mgxe_err;
    }

    std::vector<cv::Mat> Hs;
    Hs.emplace_back(prewarp_H);
    Hs.emplace_back(H_prewarped2init);
    cv::Mat H_global_cur2init = composeHomography(Hs);

    bool global_sucess = false;
    mgxe_err = RedetSucessFn(H_prewarped2init, temp_coords_mat, cur_pw_coords_mat, global_sucess);
    if (mgxe_err != MAGIC_XE_SUCCESS) {
        MAGIC_XE_LOGE("RedetSucessFn failed:%d.", mgxe_err);
        return mgxe_err;
    }

    cv::Mat h_cur2init;
    if (global_sucess) {
        h_cur2init = H_global_cur2init;
        n_lost_ = 0;
        lost_ = false;

    } else {
        lost_ = true;
        n_lost_ += 1;

        cv::Mat prev_coords;
        cv::Mat cur_coords;
        cv::Mat weights_tmp;

        mgxe_err = raft_->Process(prev_img_, input_img, prev_coords, cur_coords, weights_tmp);
        if (mgxe_err != MAGIC_XE_SUCCESS) {
            MAGIC_XE_LOGE("raft_->Process(prev_img_, input_img) failed:%d!", mgxe_err);
            return mgxe_err;
        }
        std::vector<std::vector<float>> prev_coords_vec;
        std::vector<std::vector<float>> cur_coords_vec;
        std::vector<std::vector<float>> weights_vec;

        mgxe_err = MaskCoordsFlow(prev_coords, cur_coords, weights_tmp, prev_coords_vec, cur_coords_vec, weights_vec);
        if (mgxe_err != MAGIC_XE_SUCCESS) {
            MAGIC_XE_LOGE("MaskCoordsFlow failed!");
            return mgxe_err;
        }
        if (prev_coords_vec[0].size() <= 20 || cur_coords_vec[0].size() <= 20 || weights_vec[0].size() <= 20) {
            MAGIC_XE_LOGE("Tracking object mask lost or not found,track failed!");
            return MAGIC_XE_NOT_FOUND;
        }

        cv::Mat prev_coords_mat(cv::Size(500, (int)prev_coords_vec.size()), CV_32F);
        cv::Mat cur_coords_mat(cv::Size(500, (int)cur_coords_vec.size()), CV_32F);
        cv::Mat weights_out(cv::Size(500, (int)weights_vec.size()), CV_32F);

        mgxe_err =
            SubSampler(prev_coords_vec, cur_coords_vec, weights_vec, prev_coords_mat, cur_coords_mat, weights_out);
        if (mgxe_err != MAGIC_XE_SUCCESS) {
            MAGIC_XE_LOGE("SubSampler failed");
            return mgxe_err;
        }

        cv::Mat H_flow;
        mgxe_err = FindHomography(cur_coords_mat, prev_coords_mat, weights_out, H_flow);
        if (mgxe_err != MAGIC_XE_SUCCESS) {
            MAGIC_XE_LOGE("FindHomography failed:%d.", mgxe_err);
            return mgxe_err;
        }

        std::vector<cv::Mat> Hs;
        Hs.emplace_back(H_flow);
        Hs.emplace_back(prev_H2init_);
        cv::Mat H_local_cur2init = composeHomography(Hs);
        h_cur2init = H_local_cur2init.clone();
    }

    prev_img_ = input_img.clone();
    cv::Mat prev_h2init2 = prev_H2init_;
    prev_H2init_ = h_cur2init.clone();

    if (!lost_) {
        last_good_H2init_ = h_cur2init.clone();
    }

    cv::Mat H_downscale = cv::Mat::eye(3, 3, CV_32FC1);
    cv::Mat H_upscale = cv::Mat::eye(3, 3, CV_32FC1);

    H_downscale.at<float>(0, 0) = scaled_h;
    H_downscale.at<float>(1, 1) = scaled_w;

    H_upscale.at<float>(0, 0) = 1.0 / scaled_h;
    H_upscale.at<float>(1, 1) = 1.0 / scaled_w;
    std::vector<cv::Mat> Hs_tmp;
    Hs_tmp.emplace_back(H_downscale);
    Hs_tmp.emplace_back(h_cur2init);
    Hs_tmp.emplace_back(H_upscale);
    h_cur2init = composeHomography(Hs_tmp);

    cv::Mat frame_vis = triv_tracker_vis(origin_frame_, init_mask_, warp_img_, h_cur2init);
    cv::Mat out_frame =
        cv::Mat(output->video.height, output->video.width, CV_8UC3, output->data[0], output->linesize[0]);
    CvMatConvertFrame(frame_vis, output);

    cv::Mat points_mat(4, 2, CV_32F);
    cv::vconcat(point_vec, points_mat);
    cv::Mat points_result = PerspectiveTransPoints(points_mat, h_cur2init);

    //INSERT_TIME_POINT_END("Host", "Post-processing");

    if (points_result.rows == 4 && points_result.cols == 2) {
        *output_data = MagicXEDictionaryAlloc(1);
        MagicXEArray *points_array = MagicXEArrayAlloc(MAGIC_XE_VALUE_TYPE_POINT, 4);
        for (int i = 0; i < 4; i++) {
            MagicXEPointV2 point = {points_result.at<float>(i, 0), points_result.at<float>(i, 1)};
            MAGIC_XE_ARRAY_INSERT(points_array, i, MAGIC_XE_VALUE_TYPE_POINT, &point);
        }
        MAGIC_XE_DICT_INSERT(*output_data, "coords", MAGIC_XE_VALUE_TYPE_ARRAY, points_array);
        MagicXEArrayFree(&points_array);

    } else {
        MAGIC_XE_LOGE("PerspectiveTransPoints failed,track failed!");
        return MAGIC_XE_INVALID_PARAM;
    }

    return mgxe_err;
}

MagicXEError PlaneTrackAlgorithm::FindHomography(
    const cv::Mat &pts_a, const cv::Mat &pts_b, const cv::Mat &weights, cv::Mat &H_prewarped2init) {
    MagicXEError mgxe_err = MAGIC_XE_SUCCESS;
    cv::Mat pts_a_t = pts_a.t(); // 1*N*2
    cv::Mat pts_b_t = pts_b.t(); // 1*N*2

    mgxe_err = FindHomographyQR(pts_a_t, pts_b_t, weights, H_prewarped2init);
    MAGIC_XE_IF_ERROR_RETURN(mgxe_err, "FindHomographyQR failed:%d.", mgxe_err);

    return mgxe_err;
}

MagicXEError PlaneTrackAlgorithm::FindHomographyQR(
    const cv::Mat &pts_a, const cv::Mat &pts_b, const cv::Mat &weights, cv::Mat &H_prewarped2init) {
    MagicXEError mgxe_err = MAGIC_XE_SUCCESS;
    if (pts_a.size() != pts_b.size()) {
        MAGIC_XE_LOGE("pts_a size != pts_b size!");
        return MAGIC_XE_INVALID_PARAM;
    }

    if (pts_a.cols != 2) {
        MAGIC_XE_LOGE("pts_a cols != 2!");
        return MAGIC_XE_INVALID_PARAM;
    }

    if (pts_a.rows < 4) {
        MAGIC_XE_LOGE("pts_a rows <4!");
        return MAGIC_XE_INVALID_PARAM;
    }

    cv::Mat a_point1_norm;
    cv::Mat a_tans1;
    mgxe_err = NormalizePoints(pts_a, a_point1_norm, a_tans1);
    MAGIC_XE_IF_ERROR_RETURN(mgxe_err, "NormalizePoints failed:%d.", mgxe_err);
    cv::Mat a_point2_norm;
    cv::Mat a_tans2;
    mgxe_err = NormalizePoints(pts_b, a_point2_norm, a_tans2);
    MAGIC_XE_IF_ERROR_RETURN(mgxe_err, "NormalizePoints failed:%d.", mgxe_err);

    int mid = a_point1_norm.cols / 2;
    cv::Mat x1 = a_point1_norm(cv::Range(0, a_point1_norm.rows), cv::Range(0, mid));
    cv::Mat y1 = a_point1_norm(cv::Range(0, a_point1_norm.rows), cv::Range(mid, a_point1_norm.cols));
    cv::Mat x2 = a_point2_norm(cv::Range(0, a_point2_norm.rows), cv::Range(0, mid));
    cv::Mat y2 = a_point2_norm(cv::Range(0, a_point2_norm.rows), cv::Range(mid, a_point2_norm.cols));
    cv::Mat zero_mat = cv::Mat::zeros(x1.size(), x1.type());
    cv::Mat one_mat = cv::Mat::ones(x1.size(), x1.type());
    std::vector<cv::Mat> mat_stack1;
    mat_stack1.emplace_back(zero_mat);
    mat_stack1.emplace_back(zero_mat);
    mat_stack1.emplace_back(zero_mat);
    mat_stack1.emplace_back(-x1);
    mat_stack1.emplace_back(-y1);
    mat_stack1.emplace_back(-one_mat);
    mat_stack1.emplace_back(y2.mul(x1));
    mat_stack1.emplace_back(y2.mul(y1));
    cv::Mat ax;
    cv::hconcat(mat_stack1, ax);
    mat_stack1.clear();
    std::vector<cv::Mat> mat_stack2;
    mat_stack2.emplace_back(x1);
    mat_stack2.emplace_back(y1);
    mat_stack2.emplace_back(one_mat);
    mat_stack2.emplace_back(zero_mat);
    mat_stack2.emplace_back(zero_mat);
    mat_stack2.emplace_back(zero_mat);
    mat_stack2.emplace_back((-x2).mul(x1));
    mat_stack2.emplace_back((-x2).mul(y1));
    cv::Mat ay;
    cv::hconcat(mat_stack2, ay);
    mat_stack2.clear();

    cv::Mat A;
    cv::hconcat(ax, ay, A);
    A = A.reshape(0, A.rows * 2);
    cv::Mat bx = -y2;
    cv::Mat by = x2;
    cv::Mat b;
    cv::hconcat(bx, by, b);
    b = b.reshape(0, b.rows * 2);

    if (!weights.empty()) {
        cv::Mat two_weights = weights.reshape(0, weights.cols);
        cv::hconcat(two_weights, two_weights, two_weights);
        two_weights = two_weights.reshape(0, two_weights.total());
        two_weights = cv::repeat(two_weights, 1, A.cols);
        A = two_weights.mul(A);
        b = (two_weights.col(0)).mul(b);
    }

    H_prewarped2init = solveHomography(A, b, a_tans1, a_tans2);

    return mgxe_err;
}

MagicXEError PlaneTrackAlgorithm::RedetSucessFn(const cv::Mat &H_prewarped2init,
    const cv::Mat &template_coords,
    const cv::Mat &cur_pw_coords,
    bool &global_h_success) {

    cv::Mat pts_a(cur_pw_coords);
    cv::copyMakeBorder(pts_a, pts_a, 0, 1, 0, 0, cv::BORDER_CONSTANT, 1.0f);

    cv::Mat gt_h(H_prewarped2init);
    cv::Mat pts_b(template_coords);
    cv::Mat proj_pts = gt_h * pts_a;
    cv::Mat proj_pts_res;
    MagicXEError mgxe_err = ConvertPointsfromhomogeneous(proj_pts, proj_pts_res);
    MAGIC_XE_IF_ERROR_RETURN(mgxe_err, "ConvertPointsfromhomogeneous failed:%d.", mgxe_err);
    //  proj_pts and pts_B diff ,and squared
    if (proj_pts_res.size() != pts_b.size()) {
        MAGIC_XE_LOGE("proj_pts_res and pts_b size not equal.");
        return MAGIC_XE_INVALID_PARAM;
    }
    cv::Mat diff_squared;
    cv::pow(proj_pts_res - pts_b, 2, diff_squared);
    // flow xy dims sum ,according to rows
    cv::Mat sum;
    cv::reduce(diff_squared, sum, 0, cv::REDUCE_SUM);

    cv::Mat L2_err;
    cv::sqrt(sum, L2_err);

    cv::Mat inliers_mat;
    cv::threshold(L2_err, inliers_mat, 5, 1, cv::THRESH_BINARY_INV);
    cv::Scalar inlier_frac = cv::mean(inliers_mat);
    global_h_success = inlier_frac.val[0] > 0.2;

    return mgxe_err;
}
//point :B N 2,point_norm:B N 2,transformation:B 3 3.
MagicXEError PlaneTrackAlgorithm::NormalizePoints(const cv::Mat &point, cv::Mat &point_norm, cv::Mat &transformation) {
    float eps = 1e-8f;
    int h = point.rows;
    int w = point.cols;
    cv::Mat x_mean;
    cv::reduce(point, x_mean, 0, cv::REDUCE_AVG); //B 1 *2;
    cv::Mat tmp = cv::repeat(x_mean, h, 1);
    cv::Mat diff = point - tmp;
    cv::Mat norm(diff.rows, 1, CV_32F);
    RowDimNorm(diff, cv::NORM_L2, norm);
    cv::Scalar cvscale = cv::mean(norm);
    float scale = std::sqrt(2.0) / (cvscale.val[0] + eps);

    cv::Mat transform(3, 3, CV_32F);
    transform.at<float>(0, 0) = scale;
    transform.at<float>(0, 1) = 0.f;
    transform.at<float>(0, 2) = -scale * x_mean.at<float>(0, 0);
    transform.at<float>(1, 0) = 0.f;
    transform.at<float>(1, 1) = scale;
    transform.at<float>(1, 2) = -scale * x_mean.at<float>(0, 1);
    transform.at<float>(2, 0) = 0.f;
    transform.at<float>(2, 1) = 0.f;
    transform.at<float>(2, 2) = 1.f;
    transformation = transform.clone();
    cv::Mat points_1_h;
    cv::copyMakeBorder(point, points_1_h, 0, 0, 0, 1, cv::BORDER_CONSTANT, 1.0f);
    cv::Mat points_0_h = points_1_h * transform.t();
    point_norm = points_0_h(cv::Range(0, points_0_h.rows), cv::Range(0, points_0_h.cols - 1));

    return MAGIC_XE_SUCCESS;
}

void GetTemplateMask(const cv::Mat &template_mask, const cv::Mat &template_coords, cv::Mat &out_template_mask) {

    int num_coords = template_coords.cols;

    for (int i = 0; i < num_coords; ++i) {
        int x = static_cast<int>(template_coords.at<float>(0, i));
        int y = static_cast<int>(template_coords.at<float>(1, i));

        out_template_mask.at<bool>(0, i) = template_mask.at<bool>(y, x);
    }
}

MagicXEError PlaneTrackAlgorithm::MaskCoordsFlow(const cv::Mat &prev_coords,
    const cv::Mat &cur_coords,
    const cv::Mat &weights,
    std::vector<std::vector<float>> &prev_coords_vec,
    std::vector<std::vector<float>> &cur_coords_vec,
    std::vector<std::vector<float>> &weights_vec) {

    if (prev_coords.rows != 2 || cur_coords.rows != 2) {
        MAGIC_XE_LOGE("cur_pw_coords or temp_coords rows not equal 2.");
        return MAGIC_XE_INVALID_PARAM;
    }

    cv::Mat prev_mask;
    cv::warpPerspective(np_template_mask_, prev_mask, prev_H2init_.inv(), np_template_mask_.size(), cv::INTER_NEAREST);
    prev_mask = prev_mask > 0;
    cv::Mat in_mask(1, prev_coords.cols, prev_mask.type());
    GetTemplateMask(prev_mask, prev_coords, in_mask);

    prev_coords_vec.resize(prev_coords.rows);
    cur_coords_vec.resize(cur_coords.rows);
    weights_vec.resize(weights.rows);

    for (int i = 0; i < in_mask.cols; i++) {
        bool value = *in_mask.ptr<bool>(0, i);
        if (value) {
            for (int j = 0; j < prev_coords.rows; j++) {
                prev_coords_vec[j].emplace_back(prev_coords.at<float>(j, i));
            }
            for (int j = 0; j < cur_coords.rows; j++) {
                cur_coords_vec[j].emplace_back(cur_coords.at<float>(j, i));
            }
            for (int j = 0; j < weights.rows; j++) {
                weights_vec[j].emplace_back(weights.at<float>(j, i));
            }
        }
    }

    return MAGIC_XE_SUCCESS;
}

MagicXEError PlaneTrackAlgorithm::MaskCoords(const cv::Mat &temp_coords,
    const cv::Mat &cur_pw_coords,
    const cv::Mat &weights,
    const cv::Mat &pw_mask,
    std::vector<std::vector<float>> &temp_coords_vec,
    std::vector<std::vector<float>> &cur_pw_coords_vec,
    std::vector<std::vector<float>> &weights_vec) {

    if (cur_pw_coords.rows != 2 || temp_coords.rows != 2) {
        MAGIC_XE_LOGE("cur_pw_coords or temp_coords rows not equal 2.");
        return MAGIC_XE_INVALID_PARAM;
    }

    cv::Mat in_template_mask(1, temp_coords.cols, template_mask_.type());
    GetTemplateMask(template_mask_, temp_coords, in_template_mask);

    if (!pw_mask.empty()) {
        int h = pw_mask.rows;
        int w = pw_mask.cols;
        cv::Mat cur_coords = cur_pw_coords.clone();
        cv::Mat cur_coords_int;
        cv::Mat cur_coords_oob;
        cur_coords.forEach<float>(
            [](float &pixel, const int *position) -> void { pixel = cv::saturate_cast<float>(cvRound(pixel)); });

        cur_coords.convertTo(cur_coords_int, CV_32S);

        cv::bitwise_or(cur_pw_coords.row(0) < 0, cur_pw_coords.row(1) < 0, cur_coords_oob);

        cv::Mat cur_coords_obb_tmp;
        cv::bitwise_or(cur_coords_int.row(0) >= w, cur_coords_int.row(1) >= h, cur_coords_obb_tmp);
        cv::bitwise_or(cur_coords_oob, cur_coords_obb_tmp, cur_coords_oob);

        cv::Mat in_pw_mask = cv::Mat::ones(cur_coords_oob.size(), CV_8U); // all true mask

        in_pw_mask.setTo(0, cur_coords_oob);

        cv::Mat in_mask;
        cv::bitwise_and(in_template_mask, in_pw_mask, in_mask);

        temp_coords_vec.resize(temp_coords.rows);
        cur_pw_coords_vec.resize(cur_pw_coords.rows);
        weights_vec.resize(weights.rows);

        for (int i = 0; i < in_mask.cols; i++) {
            bool value = *in_mask.ptr<bool>(0, i);
            if (value) {
                for (int j = 0; j < temp_coords.rows; j++) {
                    temp_coords_vec[j].emplace_back(temp_coords.at<float>(j, i));
                }
                for (int j = 0; j < cur_pw_coords.rows; j++) {
                    cur_pw_coords_vec[j].emplace_back(cur_pw_coords.at<float>(j, i));
                }
                for (int j = 0; j < weights.rows; j++) {
                    weights_vec[j].emplace_back(weights.at<float>(j, i));
                }
            }
        }
    } else {
        MAGIC_XE_LOGE("pw_mask is empty!");
        return MAGIC_XE_INVALID_PARAM;
    }

    return MAGIC_XE_SUCCESS;
}

void multiplyAndRound(const std::vector<float> &input, int constant, std::vector<int32_t> &output) {
    size_t i = 1;
    for (const auto &element : input) {
        float multiplied = element * constant;
        int32_t rounded = static_cast<int32_t>(cvRound(multiplied));
        output[i++] = rounded;
    }
}

MagicXEError PlaneTrackAlgorithm::SubSampler(const std::vector<std::vector<float>> &coords_a,
    const std::vector<std::vector<float>> &coords_b,
    const std::vector<std::vector<float>> &weights,
    cv::Mat &coords_a_out,
    cv::Mat &coords_b_out,
    cv::Mat &weights_out) {
    MagicXEError mgxe_err = MAGIC_XE_SUCCESS;

    if (coords_a.empty() || coords_b.empty() || weights.empty()) {
        MAGIC_XE_LOGE("coords_a, coords_b, weights is empty!");
        return MAGIC_XE_INVALID_PARAM;
    }

    if (coords_a.size() != coords_b.size()) {
        MAGIC_XE_LOGE("coords_a, coords_b size is not equal!");
        return MAGIC_XE_INVALID_PARAM;
    }

    int n_pts = coords_a[0].size();

    if (weights.size() != 1 && weights[0].size() != n_pts) {
        MAGIC_XE_LOGE("weights size is not equal (1,n_pts)!");
        return MAGIC_XE_INVALID_PARAM;
    }

    int to_draw = 500;

    if (to_draw >= n_pts) {
        cv::vconcat(coords_a, coords_a_out);
        cv::vconcat(coords_b, coords_b_out);
        cv::vconcat(weights, weights_out);
        return MAGIC_XE_SUCCESS;
    }

    std::vector<float> index(to_draw - 1);
    mgxe_err = SobelEngineDraw(to_draw - 1, 0, index);
    MAGIC_XE_IF_ERROR_RETURN(mgxe_err, "SobelEngineDraw failed:%d.", mgxe_err);
    std::vector<int32_t> index_int(to_draw, 0);
    multiplyAndRound(index, n_pts, index_int);
    std::sort(index_int.begin(), index_int.end());

    int32_t index_value = 0;
    for (size_t i = 0; i < index_int.size(); i++) {
        index_value = index_int[i];
        for (size_t j = 0; j < coords_a.size(); j++) {
            coords_a_out.at<float>(j, i) = coords_a[j][index_value];
            coords_b_out.at<float>(j, i) = coords_b[j][index_value];
        }
        for (size_t j = 0; j < weights.size(); j++) {
            weights_out.at<float>(j, i) = weights[j][index_value];
        }
    }

    return mgxe_err;
}

MagicXEError PlaneTrackAlgorithm::SobelEngineDraw(int64_t n, int64_t num_generated, std::vector<float> &res) {

    int64_t l;
    int64_t wqasi_data = 0;
    int64_t sobolsate_col_stride = 1;
    int64_t result_row_stride = 1;
    for (int64_t i = 0; i < n; i++, num_generated++) {
        l = rightmost_zero(num_generated);
        wqasi_data ^= sobolstate_data[l * sobolsate_col_stride];
        res[i * result_row_stride] = wqasi_data * RECIPD;
    }
    return MAGIC_XE_SUCCESS;
}

} // namespace plane_track
} // namespace magic_xe
