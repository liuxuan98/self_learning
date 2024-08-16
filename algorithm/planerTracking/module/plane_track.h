#ifndef PLANE_TRACK_H
#define PLANE_TRACK_H

#include <memory>
#include <string>
#include <vector>

#include "magic_xe_array.h"
#include "magic_xe_dictionary.h"
#include "magic_xe_frame_convert_color.h"
#include "magic_xe_frame_v2.h"
#include "magic_xe_videotrack_version.h"
#include "opencv2/opencv.hpp"
#include "plugin/magic_xe_plugin.h"
#include "raft.h"

namespace magic_xe {
namespace plane_track {

constexpr int64_t MAXBIT = 30;
constexpr int64_t LARGEST_NUMBER = 1 << MAXBIT;
constexpr float RECIPD = 1.0 / LARGEST_NUMBER;

using namespace magic_xe::raft_algorithm;

class PlaneTrackParam {

public:
    PlaneTrackParam();
    ~PlaneTrackParam();

    void Reset();

    int input_downscale_factor_ = 5;

    std::string config_json_;
};

class PlaneTrackAlgorithm {
public:
    PlaneTrackAlgorithm();
    ~PlaneTrackAlgorithm();

    static const char *ModelsSchemaGet();
    static const char *ConfigSchemaGet();
    static const char *DataSchemaGet(bool is_input);

    MagicXEError Init(const char *models_json);
    void DeInit();

    MagicXEError ParamSet(const char *config_json);
    const char *ParamGet();

    void Reset();

    MagicXEError Process(const MagicXEFrameV2 *input_frame,
        const MagicXEFrameV2 *assist,
        const MagicXEDictionary *input_data,
        MagicXEFrameV2 *output,
        MagicXEDictionary **output_data);

    MagicXEError MaskCoords(const cv::Mat &temp_coords,
        const cv::Mat &cur_pw_coords,
        const cv::Mat &weights,
        const cv::Mat &pw_mask,
        std::vector<std::vector<float>> &temp_coords_vec,
        std::vector<std::vector<float>> &cur_pw_coords_vec,
        std::vector<std::vector<float>> &weights_vec);

    MagicXEError MaskCoordsFlow(const cv::Mat &prev_coords,
        const cv::Mat &cur_coords,
        const cv::Mat &weights,
        std::vector<std::vector<float>> &prev_coords_vec,
        std::vector<std::vector<float>> &cur_coords_vec,
        std::vector<std::vector<float>> &weights_vec);

private:
    MagicXEError InitMask(cv::Mat &img, const std::vector<cv::Point> &points);

    MagicXEError SubSampler(const std::vector<std::vector<float>> &coords_a,
        const std::vector<std::vector<float>> &coords_b,
        const std::vector<std::vector<float>> &weights,
        cv::Mat &coords_a_out,
        cv::Mat &coords_b_out,
        cv::Mat &weights_out);

    MagicXEError FindHomography(
        const cv::Mat &pts_a, const cv::Mat &pts_b, const cv::Mat &weights, cv::Mat &H_prewarped2init);

    MagicXEError FindHomographyQR(
        const cv::Mat &pts_a, const cv::Mat &pts_b, const cv::Mat &weights, cv::Mat &H_prewarped2init);

    MagicXEError RedetSucessFn(const cv::Mat &H_prewarped2init,
        const cv::Mat &template_coords,
        const cv::Mat &cur_pw_coords,
        bool &global_h_success);

    void PointGenMask(const std::vector<cv::Point> &points, cv::Mat &mask);

    MagicXEError SobelEngineDraw(int64_t n, int64_t num_generated, std::vector<float> &res);

    MagicXEError NormalizePoints(const cv::Mat &point, cv::Mat &point_norm, cv::Mat &transformation);

private:
    std::shared_ptr<PlaneTrackParam> param_;
    std::shared_ptr<Raft> raft_;

    cv::Mat np_template_mask_;
    cv::Mat template_mask_; //save first frame.
    cv::Mat template_img_;

    cv::Mat last_good_H2init_;
    cv::Mat prev_H2init_;
    cv::Mat prev_img_;

    bool lost_ = false;
    int n_lost_ = 0;

    std::vector<cv::Mat> point_vec;

    cv::Mat warp_img_;
    cv::Mat init_mask_;
    cv::Mat origin_frame_;

    bool track_init_ = false;

    const std::vector<int64_t> sobolstate_data{536870912,
        268435456,
        134217728,
        67108864,
        33554432,
        16777216,
        8388608,
        4194304,
        2097152,
        1048576,
        524288,
        262144,
        131072,
        65536,
        32768,
        16384,
        8192,
        4096,
        2048,
        1024,
        512,
        256,
        128,
        64,
        32,
        16,
        8,
        4,
        2,
        1};
};

} // namespace plane_track
} // namespace magic_xe

#endif // PLANE_TRACK_H
