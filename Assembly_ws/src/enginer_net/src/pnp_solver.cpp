#include "pnp_solver.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include <opencv2/calib3d.hpp>

namespace enginer_net
{

PnpSolver::PnpSolver(const PnpConfig& config) : config_(config)
{
    if (config_.object_points.empty()) {
        throw std::invalid_argument("三维点模型不能为空，每个名称必须对应一个三维点");
    }
}

PnpSolver::MatchedPoints PnpSolver::matchedPoints(const KeypointObservation& observation) const
{
    if (observation.keypoint_names.size() != observation.keypoints_2d.size()) {
        throw std::invalid_argument("关键点名称数量必须与二维坐标数量一致");
    }
    MatchedPoints points;
    for (size_t i = 0; i < observation.keypoint_names.size(); ++i) {
        const auto found = config_.object_points.find(observation.keypoint_names[i]);
        if (found != config_.object_points.end()) {
            points.object_points.push_back(found->second);
            points.image_points.push_back(observation.keypoints_2d[i]);
        }
    }
    if (points.object_points.size() < 4) {
        throw std::invalid_argument(
            "PnP 至少需要 4 对匹配关键点，当前=" + std::to_string(points.object_points.size()));
    }
    return points;
}

double PnpSolver::minimumDepth(const std::vector<cv::Point3d>& object_points,
                               const cv::Mat& rvec, const cv::Mat& tvec) const
{
    cv::Matx33d rotation;
    cv::Rodrigues(rvec, rotation);
    double minimum = std::numeric_limits<double>::infinity();
    for (const auto& point : object_points) {
        const double depth = rotation(2, 0) * point.x + rotation(2, 1) * point.y +
                             rotation(2, 2) * point.z + tvec.at<double>(2);
        // NumPy min 遇到 NaN 时返回 NaN，保留原版的比较行为。
        if (std::isnan(depth)) {
            return depth;
        }
        minimum = std::min(minimum, depth);
    }
    return minimum;
}

double PnpSolver::meanReprojectionError(const MatchedPoints& points,
                                        const cv::Mat& camera, const cv::Mat& distortion,
                                        const cv::Mat& rvec, const cv::Mat& tvec) const
{
    std::vector<cv::Point2d> projected;
    cv::projectPoints(points.object_points, rvec, tvec, camera, distortion, projected);
    double total = 0.0;
    for (size_t i = 0; i < projected.size(); ++i) {
        total += cv::norm(projected[i] - points.image_points[i]);
    }
    return total / static_cast<double>(projected.size());
}

PnpResult PnpSolver::estimate(const KeypointObservation& observation,
                              const cv::Mat& camera_matrix, const cv::Mat& distortion) const
{
    if (camera_matrix.dims != 2 || camera_matrix.rows != 3 ||
        camera_matrix.cols != 3 || camera_matrix.channels() != 1) {
        throw std::invalid_argument("相机内参矩阵必须是 3×3 的单通道矩阵");
    }
    cv::Mat camera;
    camera_matrix.convertTo(camera, CV_64F);
    cv::Mat dist = cv::Mat::zeros(1, 5, CV_64F);
    if (!distortion.empty()) {
        distortion.clone().reshape(1, 1).convertTo(dist, CV_64F);
    }
    const auto matched = matchedPoints(observation);
    MatchedPoints solve_points = matched;

    if (config_.use_ransac) {
        cv::Mat rvec, tvec;
        std::vector<int> inliers;
        const bool ok = cv::solvePnPRansac(
            matched.object_points, matched.image_points, camera, dist, rvec, tvec, false,
            config_.ransac_iterations, static_cast<float>(config_.ransac_reprojection_error_px),
            config_.ransac_confidence, inliers, cv::SOLVEPNP_SQPNP);
        if (!ok || inliers.size() < 4) {
            throw std::runtime_error("RANSAC 求解失败或 PnP 内点少于 4 个");
        }
        solve_points.object_points.clear();
        solve_points.image_points.clear();
        for (int index : inliers) {
            solve_points.object_points.push_back(matched.object_points[index]);
            solve_points.image_points.push_back(matched.image_points[index]);
        }
    }

    std::vector<cv::Mat> rotations, translations;
    const int count = cv::solvePnPGeneric(
        solve_points.object_points, solve_points.image_points, camera, dist,
        rotations, translations, false, cv::SOLVEPNP_SQPNP);
    if (count == 0) {
        throw std::runtime_error("SQPnP 求解失败");
    }

    bool found = false;
    double best_error = 0.0;
    cv::Mat best_rvec, best_tvec;
    for (size_t i = 0; i < rotations.size(); ++i) {
        cv::Mat rvec, tvec;
        rotations[i].reshape(1, 3).convertTo(rvec, CV_64F);
        translations[i].reshape(1, 3).convertTo(tvec, CV_64F);
        if (minimumDepth(solve_points.object_points, rvec, tvec) <= 0.0) {
            continue;
        }
        cv::solvePnPRefineLM(solve_points.object_points, solve_points.image_points,
                             camera, dist, rvec, tvec);
        if (minimumDepth(solve_points.object_points, rvec, tvec) <= 0.0) {
            continue;
        }
        const double error = meanReprojectionError(solve_points, camera, dist, rvec, tvec);
        if (!found || error < best_error) {
            found = true;
            best_error = error;
            best_rvec = rvec;
            best_tvec = tvec;
        }
    }
    if (!found) {
        throw std::runtime_error("PnP 候选位姿均未通过相机正深度检查");
    }

    cv::Matx33d rotation;
    cv::Rodrigues(best_rvec, rotation);
    PnpResult result;
    result.transform = cv::Matx44d::eye();
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            result.transform(row, col) = rotation(row, col);
        }
        result.transform(row, 3) = best_tvec.at<double>(row);
    }
    // 最终误差按全部匹配点计算，包括被 RANSAC 排除的点，与港科一致。
    result.reprojection_error_px = meanReprojectionError(matched, camera, dist, best_rvec, best_tvec);
    return result;
}

}  // namespace enginer_net
