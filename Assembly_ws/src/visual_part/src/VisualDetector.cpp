#include "visual_part/VisualDetector.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>


const char *KeypointNameToString(
    VisualKeypointName name) noexcept
{
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

const VisualKeypoint &VisualResult::keypoint(VisualKeypointName name) const
{
    const std::size_t index = static_cast<std::size_t>(name);
    return keypoints.at(index);
}

namespace{
    int validateTargetClassId(int targetClassId)
    {
         if (targetClassId < 0) {
        throw std::invalid_argument(
            "目标类别 ID 不能小于 0");
    }

        return targetClassId;
    }

    float validateKeypointScoreThreshold(float threshold)
    {
         if (!std::isfinite(threshold) ||
        threshold < 0.0F) {
        throw std::invalid_argument(
            "关键点置信度阈值必须是大于等于 0 的有限数值");
    }

        return threshold;
    }
}//namespace

VisualDetector::VisualDetector(
    const std::string &yoloModelPath,
    const std::string &litehrModelPath,
    float yoloScoreThreshold,
    int targetClassId,
    float bboxPadding,
    float keypointScoreThreshold,
    const std::string &yoloDevice,
    const std::string &litehrDevice)
    : target_class_id_(
          validateTargetClassId(targetClassId)),
      keypoint_score_threshold_(
          validateKeypointScoreThreshold(
              keypointScoreThreshold)),
      yolo_detector_(
          yoloModelPath,
          yoloScoreThreshold,
          yoloDevice),
      litehr_detector_(
          litehrModelPath,
          bboxPadding,
          litehrDevice)
{
}

const YoloResult *VisualDetector::selectBestTarget() const noexcept
{
    const YoloResult *best = nullptr;

    for (const YoloResult &candidate :
         yolo_detector_.detectResults) {

        if (candidate.class_id != target_class_id_) {
            continue;
        }

        if (best == nullptr ||
            candidate.confidence > best->confidence) {
            best = &candidate;
        }
    }

    return best;
}

std::array<VisualKeypoint,kVisualKeypointCount>VisualDetector::organizeKeypoints(
    const std::vector<KeypointResult> &rawKeypoints
) const
{
    if (rawKeypoints.size() != kVisualKeypointCount) {
        throw std::runtime_error(
            "LiteHR 返回的关键点数量不是 12");
    }

    std::array<
        VisualKeypoint,
        kVisualKeypointCount> organized{};

    std::array<
        bool,
        kVisualKeypointCount> seen{};

    for (const KeypointResult &raw : rawKeypoints) {
        if (raw.keypoint_id < 0) {
            throw std::runtime_error(
                "LiteHR 返回了负数关键点编号");
        }

        const std::size_t index =
            static_cast<std::size_t>(
                raw.keypoint_id);

        if (index >= kVisualKeypointCount) {
            throw std::runtime_error(
                "LiteHR 关键点编号超出 exchange12 定义");
        }

        if (seen[index]) {
            throw std::runtime_error(
                "LiteHR 返回了重复的关键点编号");
        }

        if (!std::isfinite(raw.confidence) ||
            !std::isfinite(raw.point.x) ||
            !std::isfinite(raw.point.y)) {
            throw std::runtime_error(
                "LiteHR 关键点包含无效数值");
        }

        seen[index] = true;

        organized[index] = VisualKeypoint{
            static_cast<VisualKeypointName>(index),
            raw.point,
            raw.confidence,
            raw.confidence >= keypoint_score_threshold_
        };
    }

    for (bool present : seen) {
        if (!present) {
            throw std::runtime_error(
                "LiteHR 返回的关键点不完整");
        }
    }

    return organized;
}

std::optional<VisualResult> VisualDetector::Detect(
    cv::Mat &frame)
{
    // 1. 拒绝空图像。
    if (frame.empty()) {
        return std::nullopt;
    }

    // 2. YOLO 检测整张图像。
    if (!yolo_detector_.Detect(frame)) {
        return std::nullopt;
    }

    // 3. 选择指定类别中置信度最高的目标。
    const YoloResult *bestTarget =
        selectBestTarget();

    if (bestTarget == nullptr) {
        return std::nullopt;
    }

    // 复制结果，后续不再依赖 vector 内部元素的地址。
    const YoloResult target = *bestTarget;

    // 4. 对目标框运行 LiteHRNet。
    if (!litehr_detector_.Detect(
            frame,
            target.box)) {
        return std::nullopt;
    }

     // 5. 按语义名称组织为固定的 12 个槽位。
    std::array<
        VisualKeypoint,
        kVisualKeypointCount> keypoints =
        organizeKeypoints(
            litehr_detector_.keypointResults);

    // 6. 统计达到置信度阈值的有效关键点。
    std::size_t validKeypointCount = 0;

    for (const VisualKeypoint &keypoint : keypoints) {
        if (keypoint.valid) {
            ++validKeypointCount;
        }
    }

    // 7. 组合两个模型的结果。
    VisualResult result{
        target.class_id,
        target.confidence,
        target.box,
        std::move(keypoints),
        validKeypointCount
    };

    return result;
}

