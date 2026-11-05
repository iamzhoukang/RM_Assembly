#include "visual_part/PnPSolver.hpp"

#include <opencv2/calib3d.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace
{

PnPOptions validateOptions(PnPOptions options)
{
    if (options.min_correspondences < 4 ||
        options.min_correspondences > kVisualKeypointCount) {
        throw std::invalid_argument(
            "PnP 最少对应点数量必须在 4 到 12 之间");
    }

    if (!std::isfinite(options.ransac_reprojection_error_px) ||
        options.ransac_reprojection_error_px <= 0.0) {
        throw std::invalid_argument(
            "RANSAC 重投影误差阈值必须是正有限数");
    }

    if (!std::isfinite(options.ransac_confidence) ||
        options.ransac_confidence <= 0.0 ||
        options.ransac_confidence >= 1.0) {
        throw std::invalid_argument(
            "RANSAC 置信度必须位于 0 到 1 之间");
    }

    if (options.ransac_iterations <= 0) {
        throw std::invalid_argument(
            "RANSAC 迭代次数必须大于 0");
    }

    return options;
}

PnPSolver::ObjectPointArray validateObjectPoints(
    PnPSolver::ObjectPointArray objectPoints)
{
    for (const cv::Point3d &point : objectPoints) {
        if (!std::isfinite(point.x) ||
            !std::isfinite(point.y) ||
            !std::isfinite(point.z)) {
            throw std::invalid_argument(
                "PnP 三维点包含无效数值");
        }
    }

    return objectPoints;
}

bool pointsAreInFrontOfCamera(
    const std::vector<cv::Point3d> &objectPoints,
    const cv::Mat &rotationVector,
    const cv::Mat &translationVector)
{
    if (objectPoints.empty() ||
        rotationVector.total() != 3 ||
        translationVector.total() != 3) {
        return false;
    }

    cv::Mat rotationMatrix;
    cv::Rodrigues(rotationVector, rotationMatrix);

    cv::Mat rotation64;
    cv::Mat translation64;
    rotationMatrix.convertTo(rotation64, CV_64F);
    translationVector.reshape(1, 3).convertTo(
        translation64,
        CV_64F);

    if (!cv::checkRange(rotation64) ||
        !cv::checkRange(translation64)) {
        return false;
    }

    for (const cv::Point3d &point : objectPoints) {
        const double cameraZ =
            rotation64.at<double>(2, 0) * point.x +
            rotation64.at<double>(2, 1) * point.y +
            rotation64.at<double>(2, 2) * point.z +
            translation64.at<double>(2, 0);

        if (!std::isfinite(cameraZ) || cameraZ <= 0.0) {
            return false;
        }
    }

    return true;
}

double meanReprojectionError(
    const std::vector<cv::Point3d> &objectPoints,
    const std::vector<cv::Point2d> &imagePoints,
    const cv::Matx33d &cameraMatrix,
    const cv::Mat &distortionCoefficients,
    const cv::Mat &rotationVector,
    const cv::Mat &translationVector)
{
    if (objectPoints.empty() ||
        objectPoints.size() != imagePoints.size()) {
        return std::numeric_limits<double>::infinity();
    }

    std::vector<cv::Point2d> projectedPoints;
    cv::projectPoints(
        objectPoints,
        rotationVector,
        translationVector,
        cameraMatrix,
        distortionCoefficients,
        projectedPoints);

    if (projectedPoints.size() != imagePoints.size()) {
        return std::numeric_limits<double>::infinity();
    }

    double totalError = 0.0;

    for (std::size_t index = 0;
         index < imagePoints.size();
         ++index) {
        const double deltaX =
            projectedPoints[index].x - imagePoints[index].x;
        const double deltaY =
            projectedPoints[index].y - imagePoints[index].y;
        const double error = std::hypot(deltaX, deltaY);

        if (!std::isfinite(error)) {
            return std::numeric_limits<double>::infinity();
        }

        totalError += error;
    }

    return totalError /
           static_cast<double>(imagePoints.size());
}

bool makeCameraFromTarget(
    const cv::Mat &rotationVector,
    const cv::Mat &translationVector,
    cv::Matx44d &cameraFromTarget)
{
    if (rotationVector.total() != 3 ||
        translationVector.total() != 3) {
        return false;
    }

    cv::Mat rotationMatrix;
    cv::Rodrigues(rotationVector, rotationMatrix);

    cv::Mat rotation64;
    cv::Mat translation64;
    rotationMatrix.convertTo(rotation64, CV_64F);
    translationVector.reshape(1, 3).convertTo(
        translation64,
        CV_64F);

    if (!cv::checkRange(rotation64) ||
        !cv::checkRange(translation64)) {
        return false;
    }

    cameraFromTarget = cv::Matx44d::eye();
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            cameraFromTarget(row, column) =
                rotation64.at<double>(row, column);
        }
        cameraFromTarget(row, 3) =
            translation64.at<double>(row, 0);
    }

    return true;
}

}  // namespace

PnPSolver::PnPSolver(
    ObjectPointArray objectPoints,
    PnPOptions options)
    : object_points_(
          validateObjectPoints(std::move(objectPoints))),
      options_(validateOptions(options))
{
}

std::optional<PnPResult> PnPSolver::Solve(
    const VisualResult &visualResult,
    const cv::Matx33d &cameraMatrix,
    const cv::Mat &distortionCoefficients) const
{
    for (double value : cameraMatrix.val) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument(
                "相机内参矩阵包含无效数值");
        }
    }

    if (cameraMatrix(0, 0) <= 0.0 ||
        cameraMatrix(1, 1) <= 0.0 ||
        cameraMatrix(2, 2) <= 0.0) {
        throw std::invalid_argument(
            "相机内参矩阵的焦距或齐次尺度无效");
    }

    if (!distortionCoefficients.empty()) {
        if (distortionCoefficients.channels() != 1) {
            throw std::invalid_argument(
                "相机畸变参数必须是单通道矩阵");
        }

        const int distortionDepth =
            distortionCoefficients.depth();
        if (distortionDepth != CV_32F &&
            distortionDepth != CV_64F) {
            throw std::invalid_argument(
                "相机畸变参数必须使用 float 或 double");
        }

        const int distortionCount = static_cast<int>(
            distortionCoefficients.total());
        if (distortionCount != 4 &&
            distortionCount != 5 &&
            distortionCount != 8 &&
            distortionCount != 12 &&
            distortionCount != 14) {
            throw std::invalid_argument(
                "相机畸变参数数量必须是 4、5、8、12 或 14");
        }

        if (!cv::checkRange(distortionCoefficients)) {
            throw std::invalid_argument(
                "相机畸变参数包含无效数值");
        }
    }

    std::vector<cv::Point3d> objectPoints;
    std::vector<cv::Point2d> imagePoints;
    std::vector<std::size_t> semanticIndices;

    objectPoints.reserve(kVisualKeypointCount);
    imagePoints.reserve(kVisualKeypointCount);
    semanticIndices.reserve(kVisualKeypointCount);

    std::array<bool, kVisualKeypointCount> seenNames{};

    for (const VisualKeypoint &keypoint : visualResult.keypoints) {
        const std::size_t semanticIndex =
            static_cast<std::size_t>(keypoint.name);

        if (semanticIndex >= kVisualKeypointCount) {
            throw std::invalid_argument(
                "VisualResult 包含无效关键点名称");
        }

        if (seenNames[semanticIndex]) {
            throw std::invalid_argument(
                "VisualResult 包含重复关键点名称");
        }
        seenNames[semanticIndex] = true;

        if (!keypoint.valid) {
            continue;
        }

        if (!std::isfinite(keypoint.point.x) ||
            !std::isfinite(keypoint.point.y)) {
            throw std::invalid_argument(
                "有效二维关键点包含无效坐标");
        }

        objectPoints.push_back(object_points_[semanticIndex]);
        imagePoints.emplace_back(
            static_cast<double>(keypoint.point.x),
            static_cast<double>(keypoint.point.y));
        semanticIndices.push_back(semanticIndex);
    }

    if (objectPoints.size() !=
        visualResult.valid_keypoint_count) {
        throw std::invalid_argument(
            "VisualResult 的有效关键点计数不一致");
    }

    if (objectPoints.size() < options_.min_correspondences) {
        return std::nullopt;
    }

    std::vector<int> selectedIndices;

    if (options_.use_ransac) {
        cv::Mat ransacRotationVector;
        cv::Mat ransacTranslationVector;
        bool ransacSucceeded = false;

        try {
            ransacSucceeded = cv::solvePnPRansac(
                objectPoints,
                imagePoints,
                cameraMatrix,
                distortionCoefficients,
                ransacRotationVector,
                ransacTranslationVector,
                false,
                options_.ransac_iterations,
                options_.ransac_reprojection_error_px,
                options_.ransac_confidence,
                selectedIndices,
                cv::SOLVEPNP_SQPNP);
        } catch (const cv::Exception &) {
            return std::nullopt;
        }

        if (!ransacSucceeded) {
            return std::nullopt;
        }
    } else {
        selectedIndices.reserve(objectPoints.size());
        for (std::size_t index = 0;
             index < objectPoints.size();
             ++index) {
            selectedIndices.push_back(
                static_cast<int>(index));
        }
    }

    if (selectedIndices.size() < 4) {
        return std::nullopt;
    }

    std::vector<bool> selectedFlags(objectPoints.size(), false);
    std::array<bool, kVisualKeypointCount> inlierMask{};
    std::vector<cv::Point3d> solveObjectPoints;
    std::vector<cv::Point2d> solveImagePoints;

    solveObjectPoints.reserve(selectedIndices.size());
    solveImagePoints.reserve(selectedIndices.size());

    for (int selectedIndex : selectedIndices) {
        if (selectedIndex < 0 ||
            static_cast<std::size_t>(selectedIndex) >=
                objectPoints.size()) {
            return std::nullopt;
        }

        const std::size_t collectedIndex =
            static_cast<std::size_t>(selectedIndex);

        if (selectedFlags[collectedIndex]) {
            return std::nullopt;
        }
        selectedFlags[collectedIndex] = true;

        const std::size_t semanticIndex =
            semanticIndices[collectedIndex];
        inlierMask[semanticIndex] = true;

        solveObjectPoints.push_back(objectPoints[collectedIndex]);
        solveImagePoints.push_back(imagePoints[collectedIndex]);
    }

    std::vector<cv::Mat> rotationVectors;
    std::vector<cv::Mat> translationVectors;

    try {
        const int solutionCount = cv::solvePnPGeneric(
            solveObjectPoints,
            solveImagePoints,
            cameraMatrix,
            distortionCoefficients,
            rotationVectors,
            translationVectors,
            false,
            cv::SOLVEPNP_SQPNP);

        if (solutionCount <= 0 ||
            rotationVectors.empty() ||
            rotationVectors.size() != translationVectors.size()) {
            return std::nullopt;
        }
    } catch (const cv::Exception &) {
        return std::nullopt;
    }

    double bestInlierError =
        std::numeric_limits<double>::infinity();
    cv::Mat bestRotationVector;
    cv::Mat bestTranslationVector;

    for (std::size_t candidateIndex = 0;
         candidateIndex < rotationVectors.size();
         ++candidateIndex) {
        try {
            cv::Mat rotationVector;
            cv::Mat translationVector;
            rotationVectors[candidateIndex]
                .reshape(1, 3)
                .convertTo(rotationVector, CV_64F);
            translationVectors[candidateIndex]
                .reshape(1, 3)
                .convertTo(translationVector, CV_64F);

            if (!cv::checkRange(rotationVector) ||
                !cv::checkRange(translationVector) ||
                !pointsAreInFrontOfCamera(
                    solveObjectPoints,
                    rotationVector,
                    translationVector)) {
                continue;
            }

            cv::solvePnPRefineLM(
                solveObjectPoints,
                solveImagePoints,
                cameraMatrix,
                distortionCoefficients,
                rotationVector,
                translationVector);

            if (!cv::checkRange(rotationVector) ||
                !cv::checkRange(translationVector) ||
                !pointsAreInFrontOfCamera(
                    solveObjectPoints,
                    rotationVector,
                    translationVector)) {
                continue;
            }

            const double inlierError = meanReprojectionError(
                solveObjectPoints,
                solveImagePoints,
                cameraMatrix,
                distortionCoefficients,
                rotationVector,
                translationVector);

            if (inlierError < bestInlierError) {
                bestInlierError = inlierError;
                bestRotationVector = rotationVector.clone();
                bestTranslationVector = translationVector.clone();
            }
        } catch (const cv::Exception &) {
            continue;
        }
    }

    if (bestRotationVector.empty() ||
        bestTranslationVector.empty() ||
        !std::isfinite(bestInlierError)) {
        return std::nullopt;
    }

    double allPointError = 0.0;
    cv::Matx44d cameraFromTarget;

    try {
        allPointError = meanReprojectionError(
            objectPoints,
            imagePoints,
            cameraMatrix,
            distortionCoefficients,
            bestRotationVector,
            bestTranslationVector);

        if (!std::isfinite(allPointError) ||
            !makeCameraFromTarget(
                bestRotationVector,
                bestTranslationVector,
                cameraFromTarget)) {
            return std::nullopt;
        }
    } catch (const cv::Exception &) {
        return std::nullopt;
    }

    PnPResult result;
    result.camera_from_target = cameraFromTarget;
    result.mean_reprojection_error_px = allPointError;
    result.correspondence_count = objectPoints.size();
    result.inlier_count = selectedIndices.size();
    result.inlier_mask = inlierMask;

    return result;
}
