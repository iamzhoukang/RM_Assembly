#include "detector.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace enginer_net
{

Detector::Detector(const DetectorConfig& config)
    : config_(config),
      yolo_(config.yolo_model, config.yolo_device),
      litehr_(config.hrnet),
      visibility_score_threshold_(
          config.visibility_score_threshold.value_or(config.min_keypoint_score))
{
}

cv::Vec4f Detector::expandBBox(
    const cv::Vec4f& box, const cv::Size& image_size, double margin) const
{
    if (margin <= 0.0) {
        return box;
    }
    const float extra_x = (box[2] - box[0]) * static_cast<float>(margin);
    const float extra_y = (box[3] - box[1]) * static_cast<float>(margin);
    return {
        std::clamp(box[0] - extra_x, 0.0f, static_cast<float>(image_size.width)),
        std::clamp(box[1] - extra_y, 0.0f, static_cast<float>(image_size.height)),
        std::clamp(box[2] + extra_x, 0.0f, static_cast<float>(image_size.width)),
        std::clamp(box[3] + extra_y, 0.0f, static_cast<float>(image_size.height))};
}

std::optional<cv::Vec4f> Detector::keypointBBox(
    const LiteHRResult& points, const cv::Size& image_size) const
{
    if (points.keypoints.size() != points.scores.size()) {
        throw std::invalid_argument("关键点分数数量必须与关键点数量一致");
    }
    std::vector<cv::Point2f> visible;
    for (size_t i = 0; i < points.keypoints.size(); ++i) {
        if (points.scores[i] >= static_cast<float>(config_.fallback_score_threshold)) {
            visible.push_back(points.keypoints[i]);
        }
    }
    if (static_cast<int>(visible.size()) < config_.fallback_min_keypoints) {
        return std::nullopt;
    }
    // 对应 NumPy 对空数组执行 min/max 时的失败，避免 C++ 访问空容器。
    if (visible.empty()) {
        throw std::invalid_argument("无法从空关键点集合计算回退框");
    }

    float x1 = visible.front().x;
    float y1 = visible.front().y;
    float x2 = x1;
    float y2 = y1;
    for (const auto& point : visible) {
        if (!std::isfinite(point.x) || !std::isfinite(point.y)) {
            return std::nullopt;
        }
        x1 = std::min(x1, point.x);
        y1 = std::min(y1, point.y);
        x2 = std::max(x2, point.x);
        y2 = std::max(y2, point.y);
    }
    if (x2 <= x1 || y2 <= y1) {
        return std::nullopt;
    }
    return cv::Vec4f(
        std::clamp(x1, 0.0f, static_cast<float>(image_size.width)),
        std::clamp(y1, 0.0f, static_cast<float>(image_size.height)),
        std::clamp(x2, 0.0f, static_cast<float>(image_size.width)),
        std::clamp(y2, 0.0f, static_cast<float>(image_size.height)));
}

Detector::PipelineResult Detector::predictInstances(const cv::Mat& image_bgr)
{
    PipelineResult result;
    const auto yolo_result = yolo_.predict(image_bgr, static_cast<float>(config_.yolo_conf));
    if (yolo_result.detections.empty()) {
        return result;
    }

    result.yolo_detections.names = yolo_result.names;
    std::vector<YoloDetection> candidates;
    for (const auto& detection : yolo_result.detections) {
        // 对应 _extract_all_detections()，这里使用 Python float 的双精度比较。
        if (!(static_cast<double>(detection.score) < config_.yolo_conf)) {
            result.yolo_detections.detections.push_back(detection);
        }
        if (detection.class_id == config_.bbox_class_id &&
            detection.score >= static_cast<float>(config_.yolo_conf)) {
            candidates.push_back(detection);
        }
    }

    int bbox_class_id = config_.bbox_class_id;
    std::string bbox_source = "yolo";
    double crop_margin = config_.crop_margin;
    if (candidates.empty()) {
        const cv::Vec4f full_image(
            0, 0, static_cast<float>(image_bgr.cols), static_cast<float>(image_bgr.rows));
        const auto coarse = litehr_.predictBBox(image_bgr, full_image);
        const auto box = keypointBBox(coarse, image_bgr.size());
        if (!box) {
            return result;
        }
        const float score_sum = std::accumulate(coarse.scores.begin(), coarse.scores.end(), 0.0f);
        const float score = score_sum / static_cast<float>(coarse.scores.size());
        candidates.push_back({*box, score, -1});
        bbox_class_id = -1;
        bbox_source = "hrnet_fallback";
        crop_margin = config_.fallback_crop_margin;
    }

    // 与港科一致，所有候选框完成关键点推理后，再由 detect() 选择目标。
    for (const auto& candidate : candidates) {
        const auto box = expandBBox(candidate.xyxy, image_bgr.size(), crop_margin);
        auto points = litehr_.predictBBox(image_bgr, box);
        result.instances.push_back({box, candidate.score, std::move(points), bbox_class_id, bbox_source});
    }
    return result;
}

KeypointObservation Detector::detect(const cv::Mat& image_bgr)
{
    if (image_bgr.dims != 2 || image_bgr.channels() != 3) {
        throw std::invalid_argument("双网络检测器要求输入 H×W×3 的三通道图像");
    }
    cv::Mat image = image_bgr;
    if (image.depth() != CV_8U) {
        // 对应 np.clip(image, 0, 255).astype(uint8)，小数截断而非四舍五入。
        cv::Mat values;
        image.convertTo(values, CV_64F);
        image.create(image_bgr.size(), CV_8UC3);
        for (int y = 0; y < image.rows; ++y) {
            const double* source = values.ptr<double>(y);
            auto* target = image.ptr<unsigned char>(y);
            for (int x = 0; x < image.cols * 3; ++x) {
                double value = std::clamp(source[x], 0.0, 255.0);
                if (std::isnan(value)) {
                    value = 0.0;
                }
                target[x] = static_cast<unsigned char>(value);
            }
        }
    }

    const auto pipeline = predictInstances(image);
    if (pipeline.instances.empty()) {
        throw std::runtime_error("YOLO-HRNet 未找到目标实例");
    }
    const auto best = std::max_element(
        pipeline.instances.begin(), pipeline.instances.end(),
        [](const DetectedInstance& a, const DetectedInstance& b) {
            return a.yolo_score < b.yolo_score;
        });
    const auto& points = best->keypoints;
    if (points.keypoints.size() != points.scores.size()) {
        throw std::invalid_argument("关键点分数数量必须与关键点数量一致");
    }

    std::map<std::string, size_t> index_by_name;
    for (size_t i = 0; i < config_.keypoint_names.size(); ++i) {
        index_by_name[config_.keypoint_names[i]] = i;
    }
    KeypointObservation observation;
    observation.bbox_xyxy = best->bbox_xyxy;
    auto& metadata = observation.metadata;
    metadata.yolo_score = best->yolo_score;
    metadata.bbox_class_id = best->bbox_class_id;
    metadata.bbox_source = best->bbox_source;
    metadata.yolo_detections = pipeline.yolo_detections;
    metadata.all_keypoint_names = config_.keypoint_names;
    metadata.status_keypoint_names = config_.keypoint_names;
    metadata.visibility_score_threshold = visibility_score_threshold_;

    double minimum_score = std::numeric_limits<double>::infinity();
    bool below_minimum = false;
    for (const auto& name : config_.keypoint_names) {
        const size_t index = index_by_name.at(name);
        if (index >= points.keypoints.size()) {
            throw std::out_of_range("关键点名称对应的索引超出模型输出点数");
        }
        const cv::Point2d point(points.keypoints[index].x, points.keypoints[index].y);
        const double score = points.scores[index];
        metadata.status_keypoints_2d.push_back(point);
        metadata.status_keypoint_scores.push_back(score);
        if (std::isnan(score) || score < minimum_score) {
            minimum_score = score;
        }
        if (score < config_.min_keypoint_score) {
            below_minimum = true;
        }
        if (!config_.drop_low_confidence_keypoints || score >= visibility_score_threshold_) {
            observation.keypoints_2d.push_back(point);
            observation.keypoint_names.push_back(name);
            observation.confidences.push_back(score);
        }
    }

    if (config_.drop_low_confidence_keypoints) {
        if (static_cast<int>(observation.keypoint_names.size()) < config_.min_visible_keypoints) {
            throw std::runtime_error(
                "YOLO-HRNet 可见关键点数量不足：当前=" +
                std::to_string(observation.keypoint_names.size()) +
                "，至少需要=" + std::to_string(config_.min_visible_keypoints));
        }
    } else if (below_minimum) {
        std::ostringstream message;
        message << "YOLO-HRNet 关键点分数低于阈值：最低分数="
                << std::fixed << std::setprecision(3) << minimum_score
                << "，阈值=" << config_.min_keypoint_score;
        throw std::runtime_error(message.str());
    }
    metadata.selected_keypoint_names = observation.keypoint_names;
    return observation;
}

}  // namespace enginer_net
