#pragma once

#include "visual_part/VisualDetector.hpp"

#include <opencv2/core.hpp>

#include <array>
#include <cstddef>
#include <optional>

// PnP 每帧求解使用的配置。
struct PnPOptions
{
    // 输入有效点和 RANSAC 内点都需达到此数量。
    std::size_t min_correspondences = 5;
    bool use_ransac = true;
    double ransac_reprojection_error_px = 3.0;
    double ransac_confidence = 0.99;
    int ransac_iterations = 100;
};

// PnP 成功求解得到的目标位姿。
struct PnPResult
{
    // 将目标坐标系中的点变换到相机坐标系：
    // point_camera = camera_from_target * point_target
    cv::Matx44d camera_from_target = cv::Matx44d::eye();

    double mean_reprojection_error_px = 0.0;

    std::size_t correspondence_count = 0;
    std::size_t inlier_count = 0;

    // 按 VisualKeypointName 的固定编号记录 RANSAC 内点。
    std::array<bool,kVisualKeypointCount> inlier_mask{};
};

class PnPSolver
{
public:
    // 三维点数组也按照 VisualKeypointName 的枚举编号排列。
    using ObjectPointArray = std::array<
        cv::Point3d,
        kVisualKeypointCount>;

    explicit PnPSolver(
        ObjectPointArray objectPoints,
        PnPOptions options = {});

    // 使用有效二维点、对应三维点和相机标定参数求解位姿。
    // 点数不足或求解失败时返回 std::nullopt。
    std::optional<PnPResult> Solve(
        const VisualResult &visualResult,
        const cv::Matx33d &cameraMatrix,
        const cv::Mat &distortionCoefficients) const;

private:
    ObjectPointArray object_points_;
    PnPOptions options_;
};