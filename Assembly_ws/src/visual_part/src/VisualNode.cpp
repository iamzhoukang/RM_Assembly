#include "visual_part/HkCamera.hpp"
#include "visual_part/PnPSolver.hpp"
#include "visual_part/VisualDetector.hpp"

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/transform_broadcaster.h>

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

class VisualNode final : public rclcpp::Node {
public:
    VisualNode();
    ~VisualNode() override;

private:
    std::filesystem::path resolvePackagePath(const std::string &configuredPath) const;
    void processFrame();
    void publishTransform(const PnPResult &result, const rclcpp::Time &stamp);
    void showDebugFrame(cv::Mat &frame, const VisualResult &visualResult,
                        const std::optional<PnPResult> &pnpResult) const;

    std::unique_ptr<HkCamera> camera_;
    cv::VideoCapture video_;
    cv::Mat image_;
    std::string input_source_;
    std::unique_ptr<VisualDetector> visual_detector_;
    std::unique_ptr<PnPSolver> pnp_solver_;

    cv::Matx33d camera_matrix_{};
    cv::Mat distortion_coefficients_;

    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    rclcpp::TimerBase::SharedPtr timer_;

    std::string parent_frame_;
    std::string child_frame_;
    bool show_image_ = false;
};

std::filesystem::path VisualNode::resolvePackagePath(const std::string &configuredPath) const {
    if (configuredPath.empty()) {
        throw std::invalid_argument("配置文件中的路径不能为空");
    }

    const std::filesystem::path inputPath{configuredPath};
    std::filesystem::path resolvedPath;

    if (inputPath.is_absolute()) {
        resolvedPath = inputPath;
    } else {
        const std::filesystem::path packageShare =
            ament_index_cpp::get_package_share_directory("visual_part");
        resolvedPath = packageShare / inputPath;
    }

    resolvedPath = resolvedPath.lexically_normal();
    if (!std::filesystem::is_regular_file(resolvedPath)) {
        throw std::runtime_error("文件不存在或不是普通文件: " + resolvedPath.string());
    }

    return resolvedPath;
}

VisualNode::VisualNode() : rclcpp::Node("visual_node") {
    input_source_ = declare_parameter<std::string>("input.source", "camera");
    const std::string inputPath = declare_parameter<std::string>("input.path", "");

    const std::string serialNumber = declare_parameter<std::string>("camera.serial_number", "");
    const bool autoWhiteBalance = declare_parameter<bool>("camera.auto_white_balance", true);
    const std::int64_t exposureTimeUs =
        declare_parameter<std::int64_t>("camera.exposure_time_us", 8000);
    const double gain = declare_parameter<double>("camera.gain", 0.7);
    const double gamma = declare_parameter<double>("camera.gamma", 1.0);
    const bool flip = declare_parameter<bool>("camera.flip", false);
    const bool mirror = declare_parameter<bool>("camera.mirror", false);

    if (exposureTimeUs <= 0 || exposureTimeUs > std::numeric_limits<int>::max()) {
        throw std::invalid_argument("camera.exposure_time_us 超出有效范围");
    }

    const std::string yoloModelConfiguredPath =
        declare_parameter<std::string>("yolo.model_path", "");
    const std::string yoloDevice = declare_parameter<std::string>("yolo.device", "CPU");
    const double yoloScoreThreshold = declare_parameter<double>("yolo.score_threshold", 0.60);
    const std::int64_t targetClassId = declare_parameter<std::int64_t>("yolo.target_class_id", 1);

    const std::string litehrModelConfiguredPath =
        declare_parameter<std::string>("litehr.model_path", "");
    const std::string litehrDevice = declare_parameter<std::string>("litehr.device", "CPU");
    const bool useHrnetFallback = declare_parameter<bool>("litehr.use_fallback", false);
    const double bboxPadding = declare_parameter<double>("litehr.bbox_padding", 1.10);
    const double keypointScoreThreshold =
        declare_parameter<double>("litehr.keypoint_score_threshold", 0.60);

    if (targetClassId < 0 || targetClassId > std::numeric_limits<int>::max()) {
        throw std::invalid_argument("yolo.target_class_id 超出有效范围");
    }

    const std::filesystem::path yoloModelPath = resolvePackagePath(yoloModelConfiguredPath);
    const std::filesystem::path litehrModelPath = resolvePackagePath(litehrModelConfiguredPath);

    const std::string pnpSchema = declare_parameter<std::string>("pnp.schema", "exchange_12pt");
    const std::int64_t minCorrespondences =
        declare_parameter<std::int64_t>("pnp.min_correspondences", 5);
    const bool useRansac = declare_parameter<bool>("pnp.use_ransac", true);
    const double ransacReprojectionError =
        declare_parameter<double>("pnp.ransac_reprojection_error_px", 3.0);
    const double ransacConfidence = declare_parameter<double>("pnp.ransac_confidence", 0.99);
    const std::int64_t ransacIterations =
        declare_parameter<std::int64_t>("pnp.ransac_iterations", 100);

    if (minCorrespondences < 4 ||
        minCorrespondences > static_cast<std::int64_t>(kVisualKeypointCount)) {
        throw std::invalid_argument("pnp.min_correspondences 必须在 4 到 12 之间");
    }
    if (ransacIterations <= 0 || ransacIterations > std::numeric_limits<int>::max()) {
        throw std::invalid_argument("pnp.ransac_iterations 超出有效范围");
    }

    PnPSolver::ObjectPointArray objectPoints{};
    const std::string pointParameterPrefix = "pnp.schemas." + pnpSchema + ".";

    for (std::size_t index = 0; index < kVisualKeypointCount; ++index) {
        const VisualKeypointName name = static_cast<VisualKeypointName>(index);
        const std::string parameterName = pointParameterPrefix + KeypointNameToString(name);
        const std::vector<double> coordinates =
            declare_parameter<std::vector<double>>(parameterName, std::vector<double>{});

        if (coordinates.size() != 3) {
            throw std::invalid_argument(parameterName + " 必须包含三个坐标");
        }
        objectPoints[index] = cv::Point3d{coordinates[0], coordinates[1], coordinates[2]};
    }

    PnPOptions pnpOptions;
    pnpOptions.min_correspondences = static_cast<std::size_t>(minCorrespondences);
    pnpOptions.use_ransac = useRansac;
    pnpOptions.ransac_reprojection_error_px = ransacReprojectionError;
    pnpOptions.ransac_confidence = ransacConfidence;
    pnpOptions.ransac_iterations = static_cast<int>(ransacIterations);
    pnp_solver_ = std::make_unique<PnPSolver>(objectPoints, pnpOptions);

    const std::vector<double> cameraMatrixValues =
        declare_parameter<std::vector<double>>("camera.camera_matrix", std::vector<double>{});
    if (cameraMatrixValues.size() != 9) {
        throw std::invalid_argument("camera.camera_matrix 必须包含 9 个数值");
    }
    camera_matrix_ =
        cv::Matx33d{cameraMatrixValues[0], cameraMatrixValues[1], cameraMatrixValues[2],
                    cameraMatrixValues[3], cameraMatrixValues[4], cameraMatrixValues[5],
                    cameraMatrixValues[6], cameraMatrixValues[7], cameraMatrixValues[8]};

    const std::vector<double> distortionValues = declare_parameter<std::vector<double>>(
        "camera.distortion_coefficients", std::vector<double>{});
    const std::size_t distortionCount = distortionValues.size();
    if (distortionCount != 4 && distortionCount != 5 && distortionCount != 8 &&
        distortionCount != 12 && distortionCount != 14) {
        throw std::invalid_argument(
            "camera.distortion_coefficients 必须包含 4、5、8、12 或 14 个数值");
    }

    distortion_coefficients_ = cv::Mat(1, static_cast<int>(distortionCount), CV_64F);
    for (std::size_t index = 0; index < distortionCount; ++index) {
        distortion_coefficients_.at<double>(0, static_cast<int>(index)) = distortionValues[index];
    }

    parent_frame_ = declare_parameter<std::string>("tf.parent_frame", "camera_optical_frame");
    child_frame_ = declare_parameter<std::string>("tf.child_frame", "exchange_target");
    show_image_ = declare_parameter<bool>("debug.show_image", true);
    const std::int64_t inferencePeriodMs =
        declare_parameter<std::int64_t>("inference.period_ms", 50);

    if (parent_frame_.empty() || child_frame_.empty() || parent_frame_ == child_frame_) {
        throw std::invalid_argument("TF 父子坐标系名称无效");
    }
    if (inferencePeriodMs <= 0) {
        throw std::invalid_argument("inference.period_ms 必须大于 0");
    }

    visual_detector_ = std::make_unique<VisualDetector>(
        yoloModelPath.string(), litehrModelPath.string(), static_cast<float>(yoloScoreThreshold),
        static_cast<int>(targetClassId), static_cast<float>(bboxPadding),
        static_cast<float>(keypointScoreThreshold), yoloDevice, litehrDevice, useHrnetFallback);

    if (input_source_ == "camera") {
        camera_ = std::make_unique<HkCamera>();
        if (!camera_->initCamera(serialNumber, autoWhiteBalance, static_cast<int>(exposureTimeUs),
                                 gain, gamma, flip, mirror)) {
            throw std::runtime_error("海康相机初始化失败");
        }
    } else if (input_source_ == "video") {
        if (!video_.open(resolvePackagePath(inputPath).string())) {
            throw std::runtime_error("无法打开视频: " + inputPath);
        }
    } else if (input_source_ == "image") {
        image_ = cv::imread(resolvePackagePath(inputPath).string());
        if (image_.empty()) throw std::runtime_error("无法读取图片: " + inputPath);
    } else {
        throw std::invalid_argument("input.source 必须是 camera、video 或 image");
    }

    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

    if (show_image_) {
        cv::namedWindow("visual_detection", cv::WINDOW_NORMAL);
    }

    timer_ = create_wall_timer(std::chrono::milliseconds(inferencePeriodMs),
                               [this]() { processFrame(); });

    RCLCPP_INFO(get_logger(), "视觉节点初始化完成: TF %s -> %s, period=%ld ms",
                parent_frame_.c_str(), child_frame_.c_str(), static_cast<long>(inferencePeriodMs));
}

VisualNode::~VisualNode() {
    if (show_image_) {
        try {
            cv::destroyWindow("visual_detection");
        } catch (const cv::Exception &) {
        }
    }
}

void VisualNode::publishTransform(const PnPResult &result, const rclcpp::Time &stamp) {
    const cv::Matx44d &transform = result.camera_from_target;
    tf2::Matrix3x3 rotation(transform(0, 0), transform(0, 1), transform(0, 2), transform(1, 0),
                            transform(1, 1), transform(1, 2), transform(2, 0), transform(2, 1),
                            transform(2, 2));

    tf2::Quaternion quaternion;
    rotation.getRotation(quaternion);
    quaternion.normalize();

    geometry_msgs::msg::TransformStamped message;
    message.header.stamp = stamp;
    message.header.frame_id = parent_frame_;
    message.child_frame_id = child_frame_;
    message.transform.translation.x = transform(0, 3);
    message.transform.translation.y = transform(1, 3);
    message.transform.translation.z = transform(2, 3);
    message.transform.rotation.x = quaternion.x();
    message.transform.rotation.y = quaternion.y();
    message.transform.rotation.z = quaternion.z();
    message.transform.rotation.w = quaternion.w();

    tf_broadcaster_->sendTransform(message);
}

void VisualNode::showDebugFrame(cv::Mat &frame, const VisualResult &visualResult,
                                const std::optional<PnPResult> &pnpResult) const {
    if (!show_image_) {
        return;
    }

    for (const YoloResult &detection : visualResult.yolo_results) {
        const char *name =
            detection.class_id == 0 ? "pillar" : (detection.class_id == 1 ? "exchange" : "unknown");
        cv::Scalar color =
            detection.class_id == 0 ? cv::Scalar(255, 128, 0) : cv::Scalar(0, 255, 255);
        cv::rectangle(frame, detection.box, color, 1);
        cv::putText(frame, std::string(name) + " " + cv::format("%.2f", detection.confidence),
                    cv::Point(detection.box.x, std::max(15, detection.box.y - 5)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
    }

    if (!visualResult.success) {
        cv::putText(frame, "NO TARGET", cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 1.0,
                    cv::Scalar(0, 0, 255), 2);
    } else {
        cv::rectangle(frame, visualResult.box, cv::Scalar(0, 255, 255), 2);

        for (const VisualKeypoint &keypoint : visualResult.keypoints) {
            const std::size_t index = static_cast<std::size_t>(keypoint.name);
            cv::Scalar color(128, 128, 128);

            if (keypoint.valid) {
                color = cv::Scalar(0, 255, 0);
                if (pnpResult.has_value() && !pnpResult->inlier_mask[index]) {
                    color = cv::Scalar(0, 165, 255);
                }
            }

            const cv::Point center(cvRound(keypoint.point.x), cvRound(keypoint.point.y));
            cv::circle(frame, center, 5, color, -1, cv::LINE_AA);

            std::ostringstream label;
            label << KeypointNameToString(keypoint.name) << ' ' << std::fixed
                  << std::setprecision(2) << keypoint.confidence;
            cv::putText(frame, label.str(), center + cv::Point(6, -6), cv::FONT_HERSHEY_SIMPLEX,
                        0.45, color, 1, cv::LINE_AA);
        }

        std::ostringstream status;
        status << (visualResult.used_fallback ? "HRNET FALLBACK " : "YOLO ")
               << "valid=" << visualResult.valid_keypoint_count;
        if (pnpResult.has_value()) {
            status << " inliers=" << pnpResult->inlier_count << " reproj=" << std::fixed
                   << std::setprecision(2) << pnpResult->mean_reprojection_error_px << "px";
        } else {
            status << " PnP failed";
        }

        cv::putText(frame, status.str(), cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 0.75,
                    pnpResult.has_value() ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255), 2,
                    cv::LINE_AA);
    }

    cv::imshow("visual_detection", frame);
    cv::waitKey(1);
}

void VisualNode::processFrame() {
    cv::Mat frame;
    if (input_source_ == "camera") {
        frame = camera_->getFrame();
    } else if (input_source_ == "image") {
        frame = image_.clone();
    } else if (!video_.read(frame)) {
        RCLCPP_INFO(get_logger(), "视频回放结束");
        timer_->cancel();
        return;
    }
    if (frame.empty()) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "相机取帧失败");
        return;
    }

    // 相机模式暂用主机取帧完成时间；离线模式是回放时间，不是原始曝光时间。
    const rclcpp::Time captureStamp = now();
    VisualResult visualResult;
    std::optional<PnPResult> pnpResult;

    try {
        visualResult = visual_detector_->Detect(frame);

        if (visualResult.success) {
            pnpResult = pnp_solver_->Solve(visualResult, camera_matrix_, distortion_coefficients_);

            if (pnpResult.has_value()) {
                publishTransform(*pnpResult, captureStamp);
                RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000,
                                     "目标位姿已发布: valid=%zu, inliers=%zu, reprojection=%.2f px",
                                     visualResult.valid_keypoint_count, pnpResult->inlier_count,
                                     pnpResult->mean_reprojection_error_px);
            }
        }

        showDebugFrame(frame, visualResult, pnpResult);
    } catch (const std::exception &exception) {
        RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 2000, "视觉处理失败: %s",
                              exception.what());
    }
}

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    int exitCode = 0;

    try {
        rclcpp::spin(std::make_shared<VisualNode>());
    } catch (const std::exception &exception) {
        RCLCPP_FATAL(rclcpp::get_logger("visual_node"), "视觉节点异常退出: %s", exception.what());
        exitCode = 1;
    }

    rclcpp::shutdown();
    return exitCode;
}
