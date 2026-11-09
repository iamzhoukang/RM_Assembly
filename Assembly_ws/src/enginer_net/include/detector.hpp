#pragma once

#include <optional>
#include <string>
#include <vector>

#include "litehr_detector.hpp"
#include "yolo_detector.hpp"

namespace enginer_net
{

struct DetectorConfig
{
    std::string yolo_model;
    std::string yolo_device = "CPU";
    double yolo_conf = 0.55;
    LiteHRConfig hrnet;  // 复用已完成的 LiteHRNet 配置。

    int bbox_class_id = 1;
    double crop_margin = 0.0;
    std::vector<std::string> keypoint_names{
        "TL", "TR", "BL", "BR", "ring", "light_BR",
        "light_TR", "shell_R", "shell_M", "shell_L", "light_TL", "light_BL"};

    double fallback_score_threshold = 0.01;
    int fallback_min_keypoints = 4;
    double fallback_crop_margin = 0.5;

    bool drop_low_confidence_keypoints = true;
    double min_keypoint_score = 0.0;
    std::optional<double> visibility_score_threshold = 0.60;  // nullopt 时使用 min_keypoint_score。
    int min_visible_keypoints = 5;
};

struct DetectionMetadata
{
    std::string source = "yolo_hrnet";
    double yolo_score;
    int bbox_class_id;
    std::string bbox_source;
    YoloDetections yolo_detections;
    std::vector<std::string> all_keypoint_names;
    std::vector<std::string> status_keypoint_names;
    std::vector<cv::Point2d> status_keypoints_2d;
    std::vector<double> status_keypoint_scores;
    double visibility_score_threshold;
    std::vector<std::string> selected_keypoint_names;
};

struct KeypointObservation
{
    std::vector<cv::Point2d> keypoints_2d;  // 筛选后的原图坐标，与名称、分数一一对应。
    std::vector<std::string> keypoint_names;
    std::vector<double> confidences;
    cv::Vec4f bbox_xyxy;
    DetectionMetadata metadata;
};

class Detector
{
public:
    explicit Detector(const DetectorConfig& config);

    KeypointObservation detect(const cv::Mat& image_bgr);  // 每次处理一帧，顺序调用。

private:
    struct DetectedInstance
    {
        cv::Vec4f bbox_xyxy;
        float yolo_score;
        LiteHRResult keypoints;
        int bbox_class_id;
        std::string bbox_source;
    };

    struct PipelineResult
    {
        std::vector<DetectedInstance> instances;
        YoloDetections yolo_detections;
    };

    PipelineResult predictInstances(const cv::Mat& image_bgr);
    cv::Vec4f expandBBox(const cv::Vec4f& box, const cv::Size& image_size, double margin) const;
    std::optional<cv::Vec4f> keypointBBox(const LiteHRResult& points, const cv::Size& image_size) const;

    DetectorConfig config_;
    YoloDetector yolo_;
    LiteHRDetector litehr_;
    double visibility_score_threshold_;
};

}  // namespace enginer_net
