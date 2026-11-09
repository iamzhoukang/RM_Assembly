#pragma once

#include <map>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "detector.hpp"

namespace enginer_net
{

struct PnpConfig
{
    std::map<std::string, cv::Point3d> object_points;  // 目标坐标系下的三维点，单位：米。
    bool use_ransac = false;
    double ransac_reprojection_error_px = 8.0;
    double ransac_confidence = 0.99;
    int ransac_iterations = 100;
};

struct PnpResult
{
    cv::Matx44d transform;  // 相机 ← 目标：p_camera = R * p_target + t。
    double reprojection_error_px;  // 全部匹配点的平均重投影误差。
};

class PnpSolver
{
public:
    explicit PnpSolver(const PnpConfig& config);

    PnpResult estimate(const KeypointObservation& observation,
                       const cv::Mat& camera_matrix,
                       const cv::Mat& distortion = cv::Mat()) const;

private:
    struct MatchedPoints
    {
        std::vector<cv::Point3d> object_points;
        std::vector<cv::Point2d> image_points;
    };

    MatchedPoints matchedPoints(const KeypointObservation& observation) const;
    double minimumDepth(const std::vector<cv::Point3d>& object_points,
                        const cv::Mat& rvec, const cv::Mat& tvec) const;
    double meanReprojectionError(const MatchedPoints& points,
                                 const cv::Mat& camera, const cv::Mat& distortion,
                                 const cv::Mat& rvec, const cv::Mat& tvec) const;

    PnpConfig config_;
};

}  // namespace enginer_net
