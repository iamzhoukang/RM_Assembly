#include "visual_part/VisualDetector.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>

const char *KeypointNameToString(VisualKeypointName name) noexcept {
    switch (name) {
    case VisualKeypointName::PillarTopLeft:
        return "TL";
    case VisualKeypointName::PillarTopRight:
        return "TR";
    case VisualKeypointName::PillarBottomLeft:
        return "BL";
    case VisualKeypointName::PillarBottomRight:
        return "BR";
    case VisualKeypointName::PillarRing:
        return "ring";
    case VisualKeypointName::ExchangeLightBottomRight:
        return "light_BR";
    case VisualKeypointName::ExchangeLightTopRight:
        return "light_TR";
    case VisualKeypointName::ExchangeShellRight:
        return "shell_R";
    case VisualKeypointName::ExchangeShellMiddle:
        return "shell_M";
    case VisualKeypointName::ExchangeShellLeft:
        return "shell_L";
    case VisualKeypointName::ExchangeLightTopLeft:
        return "light_TL";
    case VisualKeypointName::ExchangeLightBottomLeft:
        return "light_BL";
    case VisualKeypointName::Count:
        break;
    }
    return "unknown";
}

const VisualKeypoint &VisualResult::keypoint(VisualKeypointName name) const {
    const std::size_t index = static_cast<std::size_t>(name);
    return keypoints.at(index);
}

VisualDetector::VisualDetector(const std::string &yoloModelPath, const std::string &litehrModelPath,
                               float yoloScoreThreshold, int targetClassId, float bboxPadding,
                               float keypointScoreThreshold, const std::string &yoloDevice,
                               const std::string &litehrDevice, bool useHrnetFallback)
    : use_hrnet_fallback_(useHrnetFallback), target_class_id_(targetClassId),
      keypoint_score_threshold_(keypointScoreThreshold),
      yolo_detector_(yoloModelPath, yoloScoreThreshold, yoloDevice),
      litehr_detector_(litehrModelPath, bboxPadding, litehrDevice) {
    if (targetClassId < 0) {
        throw std::invalid_argument("目标类别 ID 不能小于 0");
    }
    if (!std::isfinite(keypointScoreThreshold) || keypointScoreThreshold < 0.0F) {
        throw std::invalid_argument("关键点置信度阈值必须是非负有限数");
    }
}

const YoloResult *VisualDetector::selectBestTarget() const noexcept {
    const YoloResult *best = nullptr;

    for (const YoloResult &candidate : yolo_detector_.detectResults) {

        if (candidate.class_id != target_class_id_) {
            continue;
        }

        if (best == nullptr || candidate.confidence > best->confidence) {
            best = &candidate;
        }
    }

    return best;
}

bool VisualDetector::findFallbackBox(cv::Mat &frame, cv::Rect &box) {
    if (!litehr_detector_.Detect(frame, cv::Rect(0, 0, frame.cols, frame.rows))) return false;

    float left = frame.cols;
    float top = frame.rows;
    float right = 0;
    float bottom = 0;
    int count = 0;
    for (const KeypointResult &point : litehr_detector_.keypointResults) {
        if (!std::isfinite(point.confidence) || !std::isfinite(point.point.x) ||
            !std::isfinite(point.point.y) || point.confidence < keypoint_score_threshold_)
            continue;
        left = std::min(left, point.point.x);
        top = std::min(top, point.point.y);
        right = std::max(right, point.point.x);
        bottom = std::max(bottom, point.point.y);
        ++count;
    }
    if (count < 4) return false;

    left = std::clamp(left, 0.0F, static_cast<float>(frame.cols));
    top = std::clamp(top, 0.0F, static_cast<float>(frame.rows));
    right = std::clamp(right, 0.0F, static_cast<float>(frame.cols));
    bottom = std::clamp(bottom, 0.0F, static_cast<float>(frame.rows));
    box = cv::Rect(cvFloor(left), cvFloor(top), cvCeil(right) - cvFloor(left),
                   cvCeil(bottom) - cvFloor(top));
    return box.width > 0 && box.height > 0;
}

VisualResult VisualDetector::Detect(cv::Mat &frame) {
    VisualResult result;
    // 和参考项目相同：YOLO 完全没有检测框时直接结束；缺整体框时才尝试回退。
    if (!yolo_detector_.Detect(frame)) return result;
    result.yolo_results = yolo_detector_.detectResults;

    const YoloResult *target = selectBestTarget();
    if (target != nullptr) {
        result.class_id = target->class_id;
        result.confidence = target->confidence;
        result.box = target->box;
    } else {
        if (!use_hrnet_fallback_ || !findFallbackBox(frame, result.box)) return result;
        result.used_fallback = true;
        // 回退框来自关键点，不伪装成 YOLO 的 exchange 检测结果。
        result.class_id = -1;
    }
    if (!litehr_detector_.Detect(frame, result.box)) return result;

    const std::vector<KeypointResult> &points = litehr_detector_.keypointResults;
    if (points.size() != kVisualKeypointCount) {
        throw std::runtime_error("LiteHR 返回的关键点数量不是 12");
    }
    // LiteHR 后处理按 0～11 顺序输出，数组下标就是语义编号。
    for (std::size_t i = 0; i < points.size(); ++i) {
        const KeypointResult &point = points[i];
        if (!std::isfinite(point.confidence) || !std::isfinite(point.point.x) ||
            !std::isfinite(point.point.y)) {
            throw std::runtime_error("LiteHR 关键点包含无效数值");
        }
        VisualKeypoint &keypoint = result.keypoints[i];
        keypoint.name = static_cast<VisualKeypointName>(i);
        keypoint.point = point.point;
        keypoint.confidence = point.confidence;
        keypoint.valid = point.confidence >= keypoint_score_threshold_;
        if (keypoint.valid) ++result.valid_keypoint_count;
    }
    if (result.used_fallback && result.valid_keypoint_count < 4) return result;

    result.success = true;
    return result;
}
