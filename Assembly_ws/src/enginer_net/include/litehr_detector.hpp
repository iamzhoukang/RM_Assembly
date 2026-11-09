#pragma once

#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <openvino/openvino.hpp>

namespace enginer_net
{

struct LiteHRConfig
{
    std::string model_path;
    std::string device = "CPU";
    cv::Size input_size{256, 256};  // 宽、高。
    cv::Vec3f mean{123.675f, 116.28f, 103.53f};
    cv::Vec3f std{58.395f, 57.12f, 57.375f};
    double bbox_padding = 1.10;
};

struct LiteHRResult
{
    std::vector<cv::Point2f> keypoints;  // 模型通道顺序，坐标位于原图。
    std::vector<float> scores;          // 对应热图峰值，尚未做可见性过滤。
};

class LiteHRDetector
{
public:
    explicit LiteHRDetector(const LiteHRConfig& config);

    LiteHRResult predictBBox(const cv::Mat& image_bgr, const cv::Vec4f& bbox_xyxy);

private:
    struct PreparedImage
    {
        ov::Tensor tensor;
        cv::Mat inverse_affine;  // 将裁剪图坐标变回原图的 2×3 仿射矩阵。
    };

    PreparedImage preprocess(const cv::Mat& image_bgr, const cv::Vec4f& bbox_xyxy);
    LiteHRResult decodeHeatmaps(ov::Tensor output, const cv::Mat& inverse_affine) const;

    ov::CompiledModel compiled_model_;
    ov::InferRequest infer_request_;
    cv::Size input_size_;
    cv::Vec3f mean_;
    cv::Vec3f std_;
    double bbox_padding_;
};

}  // namespace enginer_net
