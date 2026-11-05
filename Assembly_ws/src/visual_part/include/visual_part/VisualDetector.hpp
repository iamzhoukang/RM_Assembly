#pragma once

#include "visual_part/LiteHRDetector.hpp"
#include "visual_part/YoloDetector.hpp"

#include <array>
#include <cstddef>
#include <string>
#include <vector>

enum class VisualKeypointName : std::size_t {
    PillarTopLeft = 0, // TL
    PillarTopRight,    // TR
    PillarBottomLeft,  // BL
    PillarBottomRight, // BR
    PillarRing,        // ring

    ExchangeLightBottomRight, // light_BR
    ExchangeLightTopRight,    // light_TR
    ExchangeShellRight,       // shell_R
    ExchangeShellMiddle,      // shell_M
    ExchangeShellLeft,        // shell_L
    ExchangeLightTopLeft,     // light_TL
    ExchangeLightBottomLeft,  // light_BL

    Count
};

inline constexpr std::size_t kVisualKeypointCount =
    static_cast<std::size_t>(VisualKeypointName::Count);
static_assert(kVisualKeypointCount == 12);

// 把枚举类转换成调试显示名称
const char *KeypointNameToString(VisualKeypointName name) noexcept;

// 联合识别层输出的单个语义关键点。
struct VisualKeypoint {
    VisualKeypointName name;
    cv::Point2f point;
    float confidence;
    bool valid;
};

// YOLO 与 LiteHRNet 的联合识别结果。
struct VisualResult {
    bool success = false;
    bool used_fallback = false;
    std::vector<YoloResult> yolo_results;
    int class_id = -1;
    float confidence = 0.0F;
    cv::Rect box;

    std::array<VisualKeypoint, kVisualKeypointCount> keypoints{};

    std::size_t valid_keypoint_count = 0;

    const VisualKeypoint &keypoint(VisualKeypointName name) const;
};

class VisualDetector {
public:
    VisualDetector(const std::string &yoloModelPath, const std::string &litehrModelPath,
                   float yoloScoreThreshold, int targetClassId, float bboxPadding,
                   float keypointScoreThreshold, const std::string &yoloDevice = "CPU",
                   const std::string &litehrDevice = "CPU", bool useHrnetFallback = false);

    ~VisualDetector() = default;

    // 通过 result.success 判断是否找到目标。
    VisualResult Detect(cv::Mat &frame);

private:
    // 从 YOLO 结果中选择指定类别的最高分目标。
    const YoloResult *selectBestTarget() const noexcept;

    bool findFallbackBox(cv::Mat &frame, cv::Rect &box);

    bool use_hrnet_fallback_;
    int target_class_id_;
    float keypoint_score_threshold_;

    YoloDetector yolo_detector_;
    LiteHRDetector litehr_detector_;
};
