#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_msgs/msg/header.hpp>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/transform_broadcaster.h>

#include "detector.hpp"
#include "pnp_solver.hpp"

namespace enginer_net
{

class NetNode : public rclcpp::Node
{
public:
    NetNode() : Node("net_node")
    {
        const auto image_topic = declare_parameter<std::string>("image_topic", "/image_raw");
        const auto info_topic = declare_parameter<std::string>("camera_info_topic", "/camera_info");
        camera_frame_ = declare_parameter<std::string>("camera_frame", "camera_optical");
        target_frame_ = declare_parameter<std::string>("target_frame", "exchange_target");
        show_window_ = declare_parameter<bool>("show_window", true);
        publish_tf_ = declare_parameter<bool>("publish_tf", true);

        detector_ = std::make_unique<Detector>(loadDetectorConfig());
        pnp_ = std::make_unique<PnpSolver>(loadPnpConfig());

        const auto pose_topic = declare_parameter<std::string>("pose_topic", "/exchange_pose");
        const auto status_topic = declare_parameter<std::string>("status_topic", "/perception/status");
        pose_pub_ = create_publisher<geometry_msgs::msg::PoseStamped>(pose_topic, 10);
        status_pub_ = create_publisher<std_msgs::msg::String>(status_topic, 10);
        if (publish_tf_) {
            tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
        }

        const auto qos = rclcpp::SensorDataQoS().keep_last(1);
        const bool publish_image = declare_parameter<bool>("publish_annotated_image", false);
        const auto debug_topic = declare_parameter<std::string>(
            "annotated_image_topic", "/perception/annotated_image");
        if (publish_image) {
            debug_pub_ = create_publisher<sensor_msgs::msg::Image>(debug_topic, qos);
        }
        info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
            info_topic, qos, [this](sensor_msgs::msg::CameraInfo::ConstSharedPtr msg) {
                camera_info_ = msg;
            });
        image_sub_ = create_subscription<sensor_msgs::msg::Image>(
            image_topic, qos, [this](sensor_msgs::msg::Image::ConstSharedPtr msg) {
                onImage(msg);
            });
        if (show_window_) {
            cv::namedWindow("Engineer perception", cv::WINDOW_NORMAL);
        }
        RCLCPP_INFO(get_logger(), "感知节点已启动，图像=%s，位姿=%s，坐标系=%s",
                    image_topic.c_str(), pose_topic.c_str(), camera_frame_.c_str());
    }

    ~NetNode() override
    {
        if (show_window_) {
            cv::destroyAllWindows();
        }
    }

private:
    DetectorConfig loadDetectorConfig()
    {
        const std::filesystem::path share_dir =
            ament_index_cpp::get_package_share_directory("enginer_net");
        DetectorConfig config;
        const auto yolo_model = declare_parameter<std::string>(
            "yolo_model", "weights/yolo/v10.12/openvino_best/best.xml");
        const auto hrnet_model = declare_parameter<std::string>(
            "hrnet_model", "weights/litehr/litehrnet30_v11.0/openvino_best/hrnet_heatmap.xml");
        config.yolo_model = (share_dir / yolo_model).string();
        config.hrnet.model_path = (share_dir / hrnet_model).string();
        config.yolo_device = declare_parameter<std::string>("yolo_device", config.yolo_device);
        config.yolo_conf = declare_parameter<double>("yolo_conf", config.yolo_conf);
        config.hrnet.device = declare_parameter<std::string>("hrnet_device", config.hrnet.device);

        const auto size = declare_parameter<std::vector<int64_t>>(
            "hrnet_input_size", std::vector<int64_t>{256, 256});
        const auto mean = declare_parameter<std::vector<double>>(
            "hrnet_mean", std::vector<double>{123.675, 116.28, 103.53});
        const auto std_values = declare_parameter<std::vector<double>>(
            "hrnet_std", std::vector<double>{58.395, 57.12, 57.375});
        if (size.size() != 2 || mean.size() != 3 || std_values.size() != 3) {
            throw std::invalid_argument("LiteHRNet 输入尺寸需要 2 个值，均值和标准差各需要 3 个值");
        }
        config.hrnet.input_size = cv::Size(static_cast<int>(size[0]), static_cast<int>(size[1]));
        for (int i = 0; i < 3; ++i) {
            config.hrnet.mean[i] = static_cast<float>(mean[i]);
            config.hrnet.std[i] = static_cast<float>(std_values[i]);
        }
        config.hrnet.bbox_padding = declare_parameter<double>("bbox_padding", config.hrnet.bbox_padding);
        config.bbox_class_id = declare_parameter<int>("bbox_class_id", config.bbox_class_id);
        config.crop_margin = declare_parameter<double>("crop_margin", config.crop_margin);
        config.keypoint_names = declare_parameter<std::vector<std::string>>(
            "keypoint_names", config.keypoint_names);
        config.fallback_score_threshold = declare_parameter<double>(
            "fallback_score_threshold", config.fallback_score_threshold);
        config.fallback_min_keypoints = declare_parameter<int>(
            "fallback_min_keypoints", config.fallback_min_keypoints);
        config.fallback_crop_margin = declare_parameter<double>(
            "fallback_crop_margin", config.fallback_crop_margin);
        config.drop_low_confidence_keypoints = declare_parameter<bool>(
            "drop_low_confidence_keypoints", config.drop_low_confidence_keypoints);
        config.min_keypoint_score = declare_parameter<double>("min_keypoint_score", config.min_keypoint_score);
        config.visibility_score_threshold = declare_parameter<double>(
            "visibility_score_threshold", config.min_keypoint_score);
        config.min_visible_keypoints = declare_parameter<int>("min_visible_keypoints", config.min_visible_keypoints);
        return config;
    }

    PnpConfig loadPnpConfig()
    {
        PnpConfig config;
        const auto schema = declare_parameter<std::string>("pose_solver.schema", "exchange_12pt");
        config.use_ransac = declare_parameter<bool>("pose_solver.use_ransac", config.use_ransac);
        config.ransac_reprojection_error_px = declare_parameter<double>(
            "pose_solver.ransac_reprojection_error_px", config.ransac_reprojection_error_px);
        config.ransac_confidence = declare_parameter<double>(
            "pose_solver.ransac_confidence", config.ransac_confidence);
        config.ransac_iterations = declare_parameter<int>(
            "pose_solver.ransac_iterations", config.ransac_iterations);

        // ROS 将 YAML 中的嵌套字段展开为带点的参数名，按所选 schema 读取全部模型点。
        const std::string prefix = "keypoint_schemas." + schema + ".points.";
        const auto& overrides = get_node_parameters_interface()->get_parameter_overrides();
        for (const auto& item : overrides) {
            if (item.first.compare(0, prefix.size(), prefix) != 0) {
                continue;
            }
            const auto xyz = declare_parameter<std::vector<double>>(item.first);
            if (xyz.size() != 3) {
                throw std::invalid_argument("三维模型点必须包含 3 个坐标：" + item.first);
            }
            config.object_points[item.first.substr(prefix.size())] = cv::Point3d(xyz[0], xyz[1], xyz[2]);
        }
        return config;
    }

    void publishPose(const std_msgs::msg::Header& header, const PnpResult& result)
    {
        const auto& t = result.transform;
        const tf2::Matrix3x3 rotation(t(0, 0), t(0, 1), t(0, 2),
                                     t(1, 0), t(1, 1), t(1, 2),
                                     t(2, 0), t(2, 1), t(2, 2));
        tf2::Quaternion quaternion;
        rotation.getRotation(quaternion);
        quaternion.normalize();

        geometry_msgs::msg::PoseStamped pose;
        pose.header = header;
        pose.pose.position.x = t(0, 3);
        pose.pose.position.y = t(1, 3);
        pose.pose.position.z = t(2, 3);
        pose.pose.orientation.x = quaternion.x();
        pose.pose.orientation.y = quaternion.y();
        pose.pose.orientation.z = quaternion.z();
        pose.pose.orientation.w = quaternion.w();
        pose_pub_->publish(pose);

        if (publish_tf_) {
            geometry_msgs::msg::TransformStamped transform;
            transform.header = header;
            transform.child_frame_id = target_frame_;
            transform.transform.translation.x = pose.pose.position.x;
            transform.transform.translation.y = pose.pose.position.y;
            transform.transform.translation.z = pose.pose.position.z;
            transform.transform.rotation = pose.pose.orientation;
            tf_broadcaster_->sendTransform(transform);
        }
    }

    void onImage(sensor_msgs::msg::Image::ConstSharedPtr msg)
    {
        if (msg->header.frame_id != camera_frame_) {
            throw std::invalid_argument("图像坐标系与 camera_frame 不一致：" + msg->header.frame_id);
        }
        if (!camera_info_) {
            return;
        }
        if (camera_info_->header.frame_id != camera_frame_) {
            throw std::invalid_argument("CameraInfo 坐标系与 camera_frame 不一致：" + camera_info_->header.frame_id);
        }
        const auto start = std::chrono::steady_clock::now();
        double fps = 0.0;
        if (last_frame_time_ != std::chrono::steady_clock::time_point{}) {
            fps = 1.0 / std::chrono::duration<double>(start - last_frame_time_).count();
        }
        last_frame_time_ = start;
        const auto frame = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::BGR8);

        KeypointObservation observation;
        PnpResult pose;
        bool detected = false;
        bool solved = false;
        std::string phase = "detect_failed";
        std_msgs::msg::String status;
        try {
            observation = detector_->detect(frame->image);
            detected = true;
            phase = "pnp_failed";
            cv::Mat camera(3, 3, CV_64F);
            std::copy(camera_info_->k.begin(), camera_info_->k.end(), camera.ptr<double>());
            const cv::Mat distortion(camera_info_->d, true);
            pose = pnp_->estimate(observation, camera, distortion);
            solved = true;
            phase = "ok";
            status.data = "ok";
        } catch (const std::exception& error) {
            status.data = "perception_failed: " + std::string(error.what());
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                                 "本帧感知失败：%s", error.what());
        }
        if (solved) {
            publishPose(msg->header, pose);
        }
        status_pub_->publish(status);
        const double elapsed_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count();
        showDebug(frame->image, msg->header, detected ? &observation : nullptr,
                  solved ? &pose : nullptr, phase, fps, elapsed_ms);
    }

    void showDebug(const cv::Mat& image, const std_msgs::msg::Header& header,
                   const KeypointObservation* observation, const PnpResult* pose,
                   const std::string& phase, double fps, double elapsed_ms)
    {
        if (!show_window_ && !debug_pub_) {
            return;
        }
        cv::Mat display = image.clone();
        if (observation) {
            for (const auto& detection : observation->metadata.yolo_detections.detections) {
                const auto& b = detection.xyxy;
                cv::rectangle(display, cv::Point(cvRound(b[0]), cvRound(b[1])),
                              cv::Point(cvRound(b[2]), cvRound(b[3])), cv::Scalar(255, 180, 0), 1);
            }
            const auto& b = observation->bbox_xyxy;
            cv::rectangle(display, cv::Point(cvRound(b[0]), cvRound(b[1])),
                          cv::Point(cvRound(b[2]), cvRound(b[3])), cv::Scalar(0, 255, 255), 2);
            const auto& metadata = observation->metadata;
            for (size_t i = 0; i < metadata.status_keypoints_2d.size(); ++i) {
                const auto& point = metadata.status_keypoints_2d[i];
                if (!std::isfinite(point.x) || !std::isfinite(point.y)) {
                    continue;
                }
                const double score = metadata.status_keypoint_scores[i];
                const auto color = score >= metadata.visibility_score_threshold ?
                    cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
                const cv::Point position(cvRound(point.x), cvRound(point.y));
                cv::circle(display, position, 4, color, -1);
                cv::putText(display, cv::format("%s %.2f", metadata.status_keypoint_names[i].c_str(), score),
                            position + cv::Point(6, -6), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv::LINE_AA);
            }
        }
        const auto title = cv::format("%s | processing FPS %.1f | %.1f ms", phase.c_str(), fps, elapsed_ms);
        cv::putText(display, title, cv::Point(15, 28), cv::FONT_HERSHEY_SIMPLEX,
                    0.6, cv::Scalar(0, 255, 255), 2, cv::LINE_AA);
        if (pose) {
            const auto& t = pose->transform;
            const auto line = cv::format("xyz(m): %.3f %.3f %.3f | reproj: %.3f px",
                                         t(0, 3), t(1, 3), t(2, 3), pose->reprojection_error_px);
            cv::putText(display, line, cv::Point(15, 55), cv::FONT_HERSHEY_SIMPLEX,
                        0.6, cv::Scalar(0, 255, 255), 2, cv::LINE_AA);
        }
        if (debug_pub_) {
            debug_pub_->publish(*cv_bridge::CvImage(header, sensor_msgs::image_encodings::BGR8, display).toImageMsg());
        }
        if (show_window_) {
            cv::imshow("Engineer perception", display);
            cv::waitKey(1);
        }
    }

    std::unique_ptr<Detector> detector_;
    std::unique_ptr<PnpSolver> pnp_;
    sensor_msgs::msg::CameraInfo::ConstSharedPtr camera_info_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr info_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr status_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr debug_pub_;
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    std::string camera_frame_;
    std::string target_frame_;
    bool show_window_ = true;
    bool publish_tf_ = true;
    std::chrono::steady_clock::time_point last_frame_time_{};
};

}  // namespace enginer_net

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    try {
        rclcpp::spin(std::make_shared<enginer_net::NetNode>());
    } catch (const std::exception& error) {
        RCLCPP_FATAL(rclcpp::get_logger("net_node"), "感知节点启动或运行失败：%s", error.what());
        rclcpp::shutdown();
        return 1;
    }
    rclcpp::shutdown();
    return 0;
}
