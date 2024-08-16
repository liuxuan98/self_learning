#include <iostream>
#include <opencv2/opencv.hpp>

#include "magic_xe_array.h"
#include "magic_xe_dictionary.h"
#include "magic_xe_env.h"
#include "magic_xe_frame_v2.h"
#include "plugin/magic_xe_avdecoder.h"
#include "plugin/magic_xe_avencoder.h"
#include "plugin/magic_xe_plugin.h"
#include "sample_utils.h"
#include "time_profile.h"
#include "utils/magic_xe_file_utils.h"

using namespace magic_xe::tool;

MagicXEError ShowResult(MagicXEFrameV2 *frame, MagicXEDictionary *output_data) {

    MagicXEError ret = MAGIC_XE_SUCCESS;

    cv::Mat input_image = cv::Mat(frame->video.height, frame->video.width, CV_8UC3, frame->data[0], frame->linesize[0]);
    if (output_data != nullptr) {
        const MagicXEArray *coords_points = MagicXEDictionaryArrayGet(output_data, "coords", nullptr);
        for (int i = 0; i < coords_points->size; ++i) {
            float x = coords_points->values[i].point_value->x;
            float y = coords_points->values[i].point_value->y;
            cv::circle(input_image, cv::Point2f(x, y), 5, cv::Scalar(0, 255, 0), -1);
        }
    }
    return ret;
}
int main(int argc, char **argv) {
    printf("demo start!\n");
    if (argc != 13) {
        printf("Usage: <model_path> <video_path/image_directory> <save_path> <downscale_factor> <x1> <y1> <x2> <y2> "
               "<x3> <y3> <x4> <y4>");
        return -1;
    }

    for (int i = 0; i < argc; ++i) {
        printf("%s\n", argv[i]);
    }

    std::string model_path = argv[1];
    std::string data_path = argv[2];
    std::string save_path = argv[3];
    int downscale_factor = atoi(argv[4]);
    float x1, y1;
    float x2, y2;
    float x3, y3;
    float x4, y4;
    if (argc >= 5) {
        x1 = atof(argv[5]);
        y1 = atof(argv[6]);
        x2 = atof(argv[7]);
        y2 = atof(argv[8]);
        x3 = atof(argv[9]);
        y3 = atof(argv[10]);
        x4 = atof(argv[11]);
        y4 = atof(argv[12]);
    }

    if (!IsPathExist(save_path)) {
        MakeDirectory(save_path, true);
    }

    MagicXEEnv env;
    memset(&env, 0, sizeof(MagicXEEnv));
    env.size = sizeof(MagicXEEnv);
    env.license_file = "./WonderShare";
    env.log_path = "";
    env.plugin_path = ".";

    MagicXEError ret = MAGIC_XE_SUCCESS;
    ret = MagicXEEnvInitialize(&env);

    if (ret != MAGIC_XE_SUCCESS) {
        MAGIC_XE_LOGE("MagicXEEnvInitialize failed %d !", ret);
        return ret;
    }

#ifdef __APPLE__
#ifdef __aarch64__
    ret = MagicXEPluginLoad("magic_xe_mnn");
    if (ret != MAGIC_XE_SUCCESS) {
        MAGIC_XE_LOGE("MagicXEPluginLoad(magic_xe_mnn) faile:%d", ret);
        return ret;
    }

    ret = MagicXEPluginLoad("magic_xe_opencl");
    if (ret != MAGIC_XE_SUCCESS) {
        MAGIC_XE_LOGE("MagicXEPluginLoad(magic_xe_opencl) faile:%d", ret);
        return ret;
    }

    std::string model_json = "{\"models\":[{\"model_path\":\"" + std::string(model_path)
                             + "\",\"model_type\":" + std::to_string(MAGIC_XE_MODEL_TYPE_MNN) + ",\"device_type\":"
                             + std::to_string(MAGIC_XE_DEVICE_TYPE_ARM) + ",\"device_id\":0,\"num_thread\":4}]}";
#else
    ret = MagicXEPluginLoad("magic_xe_openvino");
    if (ret != MAGIC_XE_SUCCESS) {
        MAGIC_XE_LOGE("MagicXEPluginLoad(magic_xe_openvino) faile:%d", ret);
        return ret;
    }

    std::string model_json = "{\"models\":[{\"model_path\":\"" + std::string(model_path) + "\",\"device_type\":"
                             + std::to_string(MAGIC_XE_DEVICE_TYPE_X86) + ",\"device_id\":0,\"num_thread\":4}]}";

#endif
#else
    ret = MagicXEPluginLoad("magic_xe_openvino");
    if (ret != MAGIC_XE_SUCCESS) {
        MAGIC_XE_LOGE("MagicXEPluginLoad(magic_xe_openvino) faile:%d", ret);
        return ret;
    }

    std::string model_json = "{\"models\":[{\"model_path\":\"" + std::string(model_path) + "\",\"device_type\":"
                             + std::to_string(MAGIC_XE_DEVICE_TYPE_X86) + ",\"device_id\":0,\"num_thread\":4}]}";

#endif

    ret = MagicXEPluginLoad("magic_xe_avcodec");
    if (ret != MAGIC_XE_SUCCESS) {
        MAGIC_XE_LOGE("MagicXEPluginLoad(magic_xe_avcodec) faile:%d", ret);
        return ret;
    }

    ret = MagicXEPluginLoad("magic_xe_plane_track");
    if (ret != MAGIC_XE_SUCCESS) {
        MAGIC_XE_LOGE("MagicXEPluginLoad(magic_xe_plane_track) faile:%d", ret);
        return ret;
    }

    MagicXEAdapter adapter = nullptr;
    ret = MagicXEAdapterOpen("magic_xe_plane_track", model_json.c_str(), &adapter);
    if (ret != MAGIC_XE_SUCCESS) {
        MAGIC_XE_LOGE("MagicXEAdapterOpen failed:%d", ret);
        return ret;
    }

    //const char *
    std::string config = "{\"downscale_factor\":" + std::to_string(downscale_factor) + "}";
    ret = MagicXEAdapterConfigSet(adapter, config.c_str());
    if (ret != MAGIC_XE_SUCCESS) {
        MAGIC_XE_LOGE("MagicXEAdapterConfigSet failed:%d", ret);
        return ret;
    }

    const char *param = MagicXEAdapterConfigGet(adapter);
    printf("param:%s\n", param);
    //为了对齐采用opencv解码
    //cv::VideoCapture capture;
    //cv::Mat input_mat;
    //capture.open(data_path);
    MagicXEAVDecoder decoder;
    MagicXEAVEncoder encoder;

    ret = InitDecoderEncoder(&decoder, &encoder, data_path, save_path, MAGIC_XE_PIX_FMT_BGR24, MAGIC_XE_PIX_FMT_BGR24);
    if (ret != MAGIC_XE_SUCCESS) {
        MAGIC_XE_LOGE("InitDecoderEncoder failed:%d", ret);
        return ret;
    }
    // input dict.
    MagicXEDictionary *input_data = MagicXEDictionaryAlloc(1);
    //set point
    MagicXEArray *points_array = MagicXEArrayAlloc(MAGIC_XE_VALUE_TYPE_POINT, 4);
    MagicXEPointV2 point1 = {x1, y1};
    //MagicXEPointV2 point1 = {197, 159 };
    //MagicXEPointV2 point1 = {471, 158};
    MAGIC_XE_ARRAY_INSERT(points_array, 0, MAGIC_XE_VALUE_TYPE_POINT, &point1);
    MagicXEPointV2 point2 = {x2, y2};
    //MagicXEPointV2 point2 = {1467, 260};
    //MagicXEPointV2 point2 = {636, 170};

    MAGIC_XE_ARRAY_INSERT(points_array, 1, MAGIC_XE_VALUE_TYPE_POINT, &point2);
    MagicXEPointV2 point3 = {x3, y3};
    //MagicXEPointV2 point3 = {1512, 936};
    //MagicXEPointV2 point3 = {636, 281};
    MAGIC_XE_ARRAY_INSERT(points_array, 2, MAGIC_XE_VALUE_TYPE_POINT, &point3);
    MagicXEPointV2 point4 = {x4, y4};
    //MagicXEPointV2 point4 = {88, 997};
    //MagicXEPointV2 point4 = {473, 281};

    MAGIC_XE_ARRAY_INSERT(points_array, 3, MAGIC_XE_VALUE_TYPE_POINT, &point4);

    //insert
    MAGIC_XE_DICT_INSERT(input_data, "four_point_array", MAGIC_XE_VALUE_TYPE_ARRAY, points_array);
    MagicXEArrayFree(&points_array);

    MagicXEFrameV2 *frame = nullptr;
    size_t idx = 0;
    MagicXEFrameV2 *output_frame = nullptr;
    INSERT_TIME_POINT_START("Host", "Total process");
    while (MagicXEAVDecoderNextDecode(decoder, (const MagicXEFrameV2 **)&frame) == MAGIC_XE_SUCCESS) {
        //while (capture.read(input_mat)) {

        if ((ret = MagicXEFrameCloneV2(frame, &output_frame)) != MAGIC_XE_SUCCESS) {
            MAGIC_XE_LOGE("MagicXEFrameCloneV2 failed:%d", ret);
            return ret;
        }
        MagicXEDictionary *output_data = nullptr;
        // void *pixel_array[3] = {input_mat.data, nullptr, nullptr};
        // int stride_array[3] = {(int)input_mat.step, 0, 0};
        // frame = MagicXEVideoFrameMakeV2(pixel_array,
        //     MAGIC_XE_PIX_FMT_BGR24,
        //     MAGIC_XE_MEM_TYPE_HOST,
        //     input_mat.cols,
        //     input_mat.rows,
        //     stride_array,
        //     0);
        // frame->pts = idx;
        // frame->index = idx;
        printf("frame idx :%zd\n", idx);
        INSERT_TIME_POINT_START("Host", "Single process");
        ret = MagicXEAdapterProcess(adapter, frame, nullptr, input_data, output_frame, &output_data);
        INSERT_TIME_POINT_END("Host", "Single process");
        if (ret == MAGIC_XE_NOT_FOUND) {
            continue;
        }

        if (ret != MAGIC_XE_SUCCESS) {
            MAGIC_XE_LOGE("MagicXEAdapterProcess failed:%d", ret);
            return ret;
        }
        //std::string output_image = save_path + "/" + std::to_string(idx) + "test33_mnn_iter2_res.png";
        if (output_frame) {
            ret = MagicXEAVEncoderFrameEncode(encoder, output_frame);
            MagicXEFrameFreeV2(output_frame);
            if (ret != MAGIC_XE_SUCCESS) {
                MAGIC_XE_LOGE("MagicXEAVEncoderFrameEncode failed:%d", ret);
                return ret;
            }
        }
        // if (output_data != NULL) {
        //     MagicXEFrameV2 *out_frame = NULL;
        //     MagicXEFrameCloneV2(frame, &out_frame);
        //     ret = ShowResult(out_frame, output_data);
        //     ret = MagicXEAVEncoderFrameEncode(encoder, out_frame);
        //     MagicXEFrameFreeV2(out_frame);
        //     MagicXEDictionaryFree(&output_data);
        //     if (ret != MAGIC_XE_SUCCESS) {
        //         MAGIC_XE_LOGE("MagicXEAVEncoderFrameEncode failed:%d", ret);
        //         return ret;
        //     }

        //     // ret = MagicXEAVEncoderImageEncode(
        //     //     "magic_xe_avcodec", out_frame, output_image.c_str(), nullptr, nullptr, nullptr);
        // }
        //MagicXEFrameFreeV2(frame);

        ++idx;
    }
    //}
    INSERT_TIME_POINT_END("Host", "Total process");

    ret = MagicXEAVEncoderFrameEncode(encoder, nullptr);
    if (ret != MAGIC_XE_SUCCESS) {
        MAGIC_XE_LOGE("MagicXEAVEncoderFrameEncode failed:%d", ret);
        return ret;
    }

    MagicXEAVEncoderClose(&encoder);
    MagicXEAVDecoderClose(&decoder);

    MagicXEAdapterClose(&adapter);
#ifdef __APPLE__
#ifdef __aarch64__
    MagicXEPluginUnload("magic_xe_mnn");
    MagicXEPluginUnload("magic_xe_opencl");
#else
    MagicXEPluginUnload("magic_xe_openvino");
#endif
#else
    MagicXEPluginUnload("magic_xe_openvino");
#endif

    MagicXEPluginUnload("magic_xe_plane_track");
    MagicXEPluginUnload("magic_xe_avcodec");
    MagicXEEnvDeinitialize();
    DOWNLOAD_TIME_PROFILE(save_path + "/plane_track_time_profile");
    printf("demo end!\n");

    return 0;
}